"""
TCP_lang v3-a — Fusion MLP 统一语言-状态融合（无 command）。

核心改动 vs v2：
  - state_encoder: 3 → 64（比 v2 的 32 更大的表达能力）
  - 新增 fusion_mlp: lang_embed(128) + state_embed(64) → fused_embed(256) 统一规划表示
  - 空间注意力：只用 lang_embed（感知阶段由语言指导"看哪里"）
  - GRU + Route Head：使用 fused_embed（规划阶段统一语义）
  - 感知与规划分离
"""

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW

from simlingo_training.models.adaptors.adaptors import WaypointInputAdaptor
from simlingo_training.models.utils import summarise_losses
from simlingo_training.utils.custom_types import DrivingExample, DrivingInput, DrivingLabel


class TCPDecoderV3A(nn.Module):
    """v3-a decoder：fusion_mlp 统一融合，感知/规划分离。"""

    def __init__(self, hidden_size=896, lang_dim=128, state_dim=64, fused_dim=256,
                 pred_len=10, num_route_points=20):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_size = hidden_size

        # Join: mm_global + vis_pooled + lang_embed + state_embed（保留原始拼接，不做预融合）
        join_in = hidden_size * 2 + lang_dim + state_dim
        self.join = nn.Sequential(
            nn.Linear(join_in, hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True),
        )

        # GRU: x(2) + tp(2) + fused_embed(fused_dim)
        gru_in = 2 + 2 + fused_dim
        self.traj_gru = nn.GRUCell(gru_in, hidden_size)
        self.traj_out = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        # Route head: j + att_vis + fused_embed
        route_in = hidden_size * 2 + fused_dim
        self.route_head = nn.Sequential(
            nn.Linear(route_in, 512), nn.SiLU(inplace=True),
            nn.Linear(512, 256), nn.SiLU(inplace=True),
            nn.Linear(256, num_route_points * 2),
        )

        # Spatial attention: vis_spatial + lang_embed（只用语言，不用 state）
        att_in = hidden_size + lang_dim
        self.att_fc = nn.Sequential(
            nn.Linear(att_in, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # State encoder: speed(1) + tp(2) = 3 → state_dim (64)
        self.state_encoder = nn.Sequential(
            nn.Linear(3, state_dim), nn.ReLU(inplace=True),
            nn.Linear(state_dim, state_dim), nn.ReLU(inplace=True),
        )

        # Fusion MLP: lang_embed(128) + state_embed(64) → fused_dim(256)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(lang_dim + state_dim, fused_dim), nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim), nn.ReLU(inplace=True),
        )

    def forward(self, mm_global, vis_spatial, lang_embed, speed, tp):
        """
        mm_global:   [B, H]      VLM 多模态全局特征
        vis_spatial: [B, N, H]   ViT 空间特征
        lang_embed:  [B, 128]    VLM 语义向量
        speed:       [B, 1]
        tp:          [B, 2]      target_point
        """
        B, N, H = vis_spatial.shape
        device = vis_spatial.device

        # ---- State embedding ----
        state = torch.cat([speed, tp], dim=1)  # [B, 3]
        state_embed = self.state_encoder(state)  # [B, 64]

        # ---- Fusion: 统一规划表示 ----
        fused_embed = self.fusion_mlp(
            torch.cat([lang_embed, state_embed], dim=1))  # [B, 256]

        # ---- 空间池化 ----
        vis_pooled = vis_spatial.mean(dim=1)  # [B, H]

        # ---- Join: 多模态融合（lang + state 未预融合）----
        j = self.join(torch.cat([mm_global, vis_pooled, lang_embed, state_embed], dim=1))

        # ---- 空间注意力：只用 lang_embed（感知阶段）----
        lang_exp = lang_embed.unsqueeze(1).expand(-1, N, -1)  # [B, N, 128]
        att_in = torch.cat([vis_spatial, lang_exp], dim=-1)
        att_w = self.att_fc(att_in).squeeze(-1).softmax(dim=1)  # [B, N]
        att_vis = (vis_spatial * att_w.unsqueeze(-1)).sum(dim=1)  # [B, H]

        # ---- Trajectory GRU: 每步注入 fused_embed ----
        x = torch.zeros(B, 2, device=device, dtype=j.dtype)
        z = j
        pred_wp_list = []
        for _ in range(self.pred_len):
            gru_in = torch.cat([x, tp, fused_embed], dim=1)
            z = self.traj_gru(gru_in, z)
            dx = self.traj_out(z)
            x = x + dx
            pred_wp_list.append(x)
        pred_wp = torch.stack(pred_wp_list, dim=1)  # [B, pred_len, 2]

        # ---- Route head: 使用 fused_embed ----
        route_feat = torch.cat([j, att_vis, fused_embed], dim=1)
        route_flat = self.route_head(route_feat)  # [B, R*2]
        pred_route = route_flat.view(B, -1, 2).cumsum(dim=1)  # [B, R, 2]

        return pred_wp, pred_route


# ============================================================
# LightningModule
# ============================================================

class TCPLangModelV3A(pl.LightningModule):
    """TCP_lang v3-a：统一融合版，感知/规划分离，无 command。"""

    def __init__(self, cfg_data_module, processor, cache_dir, **cfg):
        super().__init__()
        self.save_hyperparameters()
        for key, value in cfg.items():
            setattr(self, key, value)

        self.processor = processor
        self.predict_language = False
        self.cfg_data_module = cfg_data_module

        # VLM encoder
        self.vision_model = hydra.utils.instantiate(
            self.vision_model, cfg_data_module=cfg_data_module,
            processor=self.processor, cache_dir=cache_dir, _recursive_=False)
        self.language_model = hydra.utils.instantiate(
            self.language_model, cache_dir=cache_dir, _recursive_=False)

        self.wp_encoder = WaypointInputAdaptor(
            token_size=self.language_model.hidden_size,
            hidden_size=256, hidden_size2=512)

        # Config
        lang_dim = getattr(self, 'lang_dim', 128)
        state_dim = getattr(self, 'state_dim', 64)
        fused_dim = getattr(self, 'fused_dim', 256)
        pred_len = getattr(self, 'pred_len', 11)
        route_pts = cfg_data_module.get('num_route_points', 20)

        self.tcp_decoder = TCPDecoderV3A(
            hidden_size=self.language_model.hidden_size,
            lang_dim=lang_dim, state_dim=state_dim, fused_dim=fused_dim,
            pred_len=pred_len - 1, num_route_points=route_pts)

        # lang_proj: mm_global 896 → 256 → lang_dim 128
        self.lang_proj = nn.Sequential(
            nn.Linear(self.language_model.hidden_size, lang_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(lang_dim * 2, lang_dim))

        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

    def forward(self, example, return_language=None, prompt_ids=None):
        try:
            di = example.driving_input
        except AttributeError:
            di = example
        pred_wp, pred_route = self.forward_model(di)
        B = pred_wp.size(0)
        return pred_wp, pred_route, [""] * B

    def forward_model(self, driving_input: DrivingInput):
        B = driving_input.camera_images.size(0)
        device = driving_input.camera_images.device

        # ---- 1) Text embeddings ----
        prompt = driving_input.prompt
        prompt_ids = prompt.phrase_ids.long().to(device)
        prompt_valid = prompt.phrase_valid.to(device)
        embed = self.language_model.model.embed_tokens
        prompt_embeds = embed(prompt_ids.clamp(min=0, max=embed.num_embeddings - 1))

        # ---- 2) ViT image features ----
        pv = driving_input.camera_images  # [B,1,2,3,448,448]
        B_pv, T, NP, C_pv, H_pv, W_pv = pv.shape
        pv_flat = pv.view(B, NP, C_pv, H_pv, W_pv).reshape(B * NP, C_pv, H_pv, W_pv)
        img_feat = self.vision_model.image_encoder.model.extract_feature(pv_flat)
        img_feat_2d = img_feat.reshape(-1, img_feat.shape[-1])  # [B*NP*256, 896]

        # ---- 3) Replace <IMG_CONTEXT> ----
        _, N_emb, C_emb = prompt_embeds.shape
        ctx_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        ids_flat = prompt_ids.reshape(B * N_emb)
        sel = (ids_flat == ctx_id)
        emb_flat = prompt_embeds.reshape(B * N_emb, C_emb)
        emb_flat[sel] = img_feat_2d[:sel.sum()].to(emb_flat.dtype)
        prompt_embeds = emb_flat.reshape(B, N_emb, C_emb)

        # ---- 4) LLM forward → hidden states ----
        am = prompt_valid.to(dtype=prompt_embeds.dtype)
        llm_out = self.language_model.model(
            inputs_embeds=prompt_embeds, attention_mask=am,
            output_hidden_states=True, return_dict=True)
        last_hidden = llm_out.hidden_states[-2]  # [B, L, H]

        # ---- 5) Extract features ----
        mm_global = last_hidden.mean(dim=1)          # [B, 896]  VLM 多模态全局
        lang_embed = self.lang_proj(mm_global)       # [B, 128]
        self._last_lang = lang_embed.detach()        # 供可视化
        del last_hidden, llm_out, prompt_embeds      # 释放显存
        vis_spatial = img_feat_2d[:sel.sum()].reshape(B, -1, img_feat.shape[-1])  # [B, N_i, 896]

        # ---- 6) State inputs（speed + tp，无 command）----
        speed = driving_input.vehicle_speed.view(B, -1)[:, :1]  # [B, 1]
        tp = driving_input.target_point.view(B, -1)[:, :2]       # [B, 2]

        # ---- 7) TCP decoder ----
        pred_wp, pred_route = self.tcp_decoder(
            mm_global, vis_spatial, lang_embed, speed, tp)
        return pred_wp, pred_route

    # ---- Loss & steps ----
    def forward_loss(self, example: DrivingExample):
        pred_wp, pred_route = self.forward_model(example.driving_input)
        gt_wp = example.driving_label.waypoints[:, :pred_wp.size(1)]
        gt_route = example.driving_label.path[:, :pred_route.size(1)]

        wp_loss = torch.nn.functional.smooth_l1_loss(pred_wp, gt_wp, reduction="none").sum(-1)
        route_loss = torch.nn.functional.smooth_l1_loss(pred_route, gt_route, reduction="none").sum(-1)
        lang_loss = torch.zeros_like(wp_loss[:, 0])

        loss_dict = {
            'speed_wps_loss': (wp_loss, torch.ones_like(wp_loss, dtype=torch.long)),
            'route_loss': (route_loss, torch.ones_like(route_loss, dtype=torch.long)),
            'language_loss': (lang_loss, torch.ones_like(lang_loss, dtype=torch.long)),
            'speed_wps_prediction': pred_wp, 'route_prediction': pred_route,
            'speed_wps_label': gt_wp, 'route_label': gt_route,
        }
        loss_only = {k: v for k, v in loss_dict.items() if k.endswith('loss')}
        return summarise_losses(loss_only, weights={'language_loss': 0.1})

    def training_step(self, batch, batch_idx=0):
        output = self.forward_loss(batch)
        self.log("train/loss", output.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in output.loss_averages.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        step = self.trainer.global_step
        if step % 100 == 0 and self.trainer.is_global_zero:
            self._print_step(batch, output)
        if step > 0 and step % 1000 == 0 and self.trainer.is_global_zero:
            self._visualise_step(batch, output, step)
        return {"loss": output.loss}

    def validation_step(self, batch, batch_idx=0):
        output = self.forward_loss(batch)
        self.log("val/loss", output.loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": output.loss}

    @torch.no_grad()
    def _visualise_step(self, batch, output, step):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from pathlib import Path

        pred_wp, pred_route = self.forward_model(batch.driving_input)
        gt_wp = batch.driving_label.waypoints[0].cpu().numpy()
        gt_route = batch.driving_label.path[0].cpu().numpy()
        pw = pred_wp[0].cpu().numpy()
        pr = pred_route[0].cpu().numpy()

        le = getattr(self, '_last_lang', None)
        if le is not None:
            le = le[0].cpu().numpy()

        save_dir = Path("visualise_tcp_lang_v3a")
        save_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ax = axes[0, 0]
        if le is not None:
            ax.bar(range(len(le)), le, color='steelblue', alpha=0.7)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_title(f'lang_embed [{len(le)} dims] (step {step})\n'
                         f'norm={np.linalg.norm(le):.2f} mean={le.mean():.4f} std={le.std():.4f}')
            ax.set_xlabel('Dimension'); ax.set_ylabel('Value')
        else:
            ax.text(0.5, 0.5, 'No embedding', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title('lang_embed (not captured)')

        ax = axes[0, 1]
        if le is not None:
            ax.hist(le, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_title(f'Embedding Distribution\n'
                         f'min={le.min():.4f} max={le.max():.4f} '
                         f'||le||={np.linalg.norm(le):.2f}')
        else:
            ax.text(0.5, 0.5, 'No embedding', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)

        ax = axes[1, 0]
        ax.plot(gt_wp[:, 0], gt_wp[:, 1], 'go-', label='GT', markersize=5, linewidth=2)
        ax.plot(pw[:, 0], pw[:, 1], 'bo-', label='Pred', markersize=5, linewidth=2)
        ax.scatter(gt_wp[0, 0], gt_wp[0, 1], c='green', s=120, marker='*', zorder=5)
        ax.scatter(pw[0, 0], pw[0, 1], c='blue', s=120, marker='*', zorder=5)
        ax.set_title(f'Waypoints (step {step})')
        ax.legend(); ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')

        ax = axes[1, 1]
        ax.plot(gt_route[:, 0], gt_route[:, 1], 'go-', label='GT Route', markersize=4, linewidth=2)
        ax.plot(pr[:, 0], pr[:, 1], 'ro-', label='Pred Route', markersize=4, linewidth=2)
        ax.scatter(gt_route[0, 0], gt_route[0, 1], c='green', s=80, marker='*', zorder=5)
        ax.scatter(pr[0, 0], pr[0, 1], c='red', s=80, marker='*', zorder=5)
        ax.set_title(f'Route (step {step})')
        ax.legend(); ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')

        plt.tight_layout()
        fig.savefig(save_dir / f"step_{step:06d}.png", dpi=100, bbox_inches='tight')
        plt.close(fig)

        if le is not None:
            print(f"  [Embed] dim={len(le)} norm={np.linalg.norm(le):.2f} "
                  f"mean={le.mean():.4f} std={le.std():.4f} "
                  f"first5=[{', '.join(f'{v:.3f}' for v in le[:5])}]")
            print(f"  [Visualise] saved to {save_dir}/step_{step:06d}.png")

    @torch.no_grad()
    def _print_step(self, batch, output):
        step = self.trainer.global_step
        lv = {k: f"{v.item():.4f}" for k, v in output.loss_averages.items()}
        pred_wp, _ = self.forward_model(batch.driving_input)
        gt_wp = batch.driving_label.waypoints[0].cpu().numpy()
        prompt = batch.driving_input.prompt.language_string[0]
        print(f"\n{'='*60}")
        print(f"Step {step:6d} | loss={output.loss.item():.4f} | "
              f"wp={lv.get('speed_wps_loss','?')} | route={lv.get('route_loss','?')}")
        print(f"  [Input] {prompt[:100]}...")
        if pred_wp is not None:
            pw = pred_wp[0].cpu().numpy()
            print(f"  GT  wps[0]: {gt_wp[0].round(3).tolist()}")
            print(f"  Pred wps[0]: {pw[0].round(3).tolist()}")
        print(f"{'='*60}\n")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)
        ms = self.trainer.max_steps if self.trainer.max_steps != -1 else self.trainer.estimated_stepping_batches
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=ms, pct_start=self.pct_start)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "frequency": 1, "interval": "step"}}
