"""
Base class for ResNet-34 TCP models with optional SimLingo language injection.

Subclasses override _build_decoder() to define how language/state are combined.
"""

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from typing import Optional

from simlingo_training.models._resnet_encoder import ResNetEncoder
from simlingo_training.models.adaptors.adaptors import WaypointInputAdaptor
from simlingo_training.models.utils import summarise_losses
from simlingo_training.utils.custom_types import DrivingExample, DrivingInput


class FiLM(nn.Module):
    """Feature-wise Linear Modulation — language-conditioned feature modulation.

    Generates channel-wise gamma/beta from lang_embed to modulate ResNet
    feature maps without changing spatial dimensions.
    """

    def __init__(self, lang_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(lang_dim, channels * 2),
        )

    def forward(self, x, lang_embed):
        # x: [B, C, H, W],  lang_embed: [B, lang_dim]
        gamma_beta = self.to_gamma_beta(lang_embed)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return x * (1.0 + gamma) + beta


class BaseResNetTCP(pl.LightningModule):
    """Shared ResNet-34 backbone + TCP decoder + training logic."""

    def __init__(self, cfg_data_module, processor, cache_dir, **cfg):
        super().__init__()
        self.save_hyperparameters()
        for key, value in cfg.items():
            setattr(self, key, value)

        self.processor = processor
        self.predict_language = False
        self.cfg_data_module = cfg_data_module

        # ── ResNet-34 visual backbone ──
        self.resnet = ResNetEncoder(pretrained=True)

        # ── Language-conditioned ResNet (FiLM adapters) ──
        self.use_resnet_film = getattr(self, 'use_resnet_film', False)
        self.resnet_film_layers = getattr(self, 'resnet_film_layers', [])
        if self.use_resnet_film:
            layer_channels = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
            lang_dim = getattr(self, 'lang_dim', 128)
            for layer_name, ch in layer_channels.items():
                if layer_name in self.resnet_film_layers:
                    setattr(self, f'film_{layer_name}', FiLM(lang_dim, ch))

        # ── SimLingo VLM (language encoder only, NOT visual backbone) ──
        self.use_lang = getattr(self, 'use_lang', False)
        if self.use_lang or self.use_resnet_film:
            self.vision_model = hydra.utils.instantiate(
                self.vision_model, cfg_data_module=cfg_data_module,
                processor=self.processor, cache_dir=cache_dir, _recursive_=False)
            self.language_model = hydra.utils.instantiate(
                self.language_model, cache_dir=cache_dir, _recursive_=False)
            self.wp_encoder = WaypointInputAdaptor(
                token_size=self.language_model.hidden_size,
                hidden_size=256, hidden_size2=512)
            lang_dim = getattr(self, 'lang_dim', 128)

        # ── Dimensions ──
        state_dim = getattr(self, 'state_dim', 128)
        pred_len = getattr(self, 'pred_len', 11)
        route_pts = cfg_data_module.get('num_route_points', 20)
        self.pred_len = pred_len
        self.state_dim = state_dim

        # ── Measurement encoder ──
        self.use_cmd = getattr(self, 'use_cmd', True)
        meas_in = 9 if self.use_cmd else 3  # speed+tp+cmd or speed+tp only
        self.measurements = nn.Sequential(
            nn.Linear(meas_in, state_dim), nn.ReLU(inplace=True),
            nn.Linear(state_dim, state_dim), nn.ReLU(inplace=True),
        )

        # ── Decoder (subclass overrides) ──
        self._build_decoder()

        # ── Tokenizer ──
        if 'tokenizer' in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = self.processor

        # ── Image preprocessing ──
        self.register_buffer('im_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('im_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # ── Subclass hooks ──────────────────────────────────────

    def _build_decoder(self):
        """Override in subclass to create decoder layers."""
        raise NotImplementedError

    def _compute_fused(self, lang_embed, state_embed):
        """Override in subclass to fuse lang+state once. Default: no fusion."""
        return None

    def _get_gru_input(self, x, tp, state_embed, lang_embed, fused_embed):
        """Override in subclass."""
        raise NotImplementedError

    def _get_join_input(self, feature_emb, vis_pooled, state_embed, lang_embed, fused_embed):
        """Override in subclass. Returns tensor for join MLP."""
        raise NotImplementedError

    def _get_att_input(self, cnn_feature, state_embed, lang_embed, fused_embed):
        """Override in subclass. Returns tensor for spatial attention."""
        raise NotImplementedError

    def _get_route_input(self, j, att_vis, state_embed, lang_embed, fused_embed):
        """Override in subclass. Returns tensor for route head."""
        raise NotImplementedError

    # ── Language embedding extraction ───────────────────────

    def _extract_lang_embed(self, driving_input: DrivingInput):
        """Run SimLingo VLM to get language embedding."""
        B = driving_input.camera_images.size(0)
        device = driving_input.camera_images.device

        prompt = driving_input.prompt
        prompt_ids = prompt.phrase_ids.long().to(device)
        prompt_valid = prompt.phrase_valid.to(device)
        embed = self.language_model.model.embed_tokens
        prompt_embeds = embed(prompt_ids.clamp(min=0, max=embed.num_embeddings - 1))

        pv = driving_input.camera_images
        B_pv, T, NP, C_pv, H_pv, W_pv = pv.shape
        pv_flat = pv.view(B, NP, C_pv, H_pv, W_pv).reshape(B * NP, C_pv, H_pv, W_pv)
        img_feat = self.vision_model.image_encoder.model.extract_feature(pv_flat)
        img_feat_2d = img_feat.reshape(-1, img_feat.shape[-1])

        _, N_emb, C_emb = prompt_embeds.shape
        ctx_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        ids_flat = prompt_ids.reshape(B * N_emb)
        sel = (ids_flat == ctx_id)
        emb_flat = prompt_embeds.reshape(B * N_emb, C_emb)
        emb_flat[sel] = img_feat_2d[:sel.sum()].to(emb_flat.dtype)
        prompt_embeds = emb_flat.reshape(B, N_emb, C_emb)

        am = prompt_valid.to(dtype=prompt_embeds.dtype)
        llm_out = self.language_model.model(
            inputs_embeds=prompt_embeds, attention_mask=am,
            output_hidden_states=True, return_dict=True)
        last_hidden = llm_out.hidden_states[-2]

        mask = prompt_valid.to(last_hidden.dtype).unsqueeze(-1)
        mm_global = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        lang_embed = self.lang_proj(mm_global)
        del last_hidden, llm_out, prompt_embeds
        return lang_embed

    # ── ResNet with FiLM conditioning ───────────────────────

    def _run_resnet_film(self, x, lang_embed):
        """Run ResNet-34 with FiLM conditioning after specified stages.

        Args:
            x: [B_total, 3, H, W]  normalized RGB
            lang_embed: [B_total, lang_dim]  language embedding

        Returns:
            feature_emb: [B_total, 1000]
            cnn_feature: [B_total, 512, h, w]
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        if 'layer1' in self.resnet_film_layers:
            x = self.film_layer1(x, lang_embed)

        x = self.resnet.layer2(x)
        if 'layer2' in self.resnet_film_layers:
            x = self.film_layer2(x, lang_embed)

        x = self.resnet.layer3(x)
        if 'layer3' in self.resnet_film_layers:
            x = self.film_layer3(x, lang_embed)

        x_layer4 = self.resnet.layer4(x)
        if 'layer4' in self.resnet_film_layers:
            x_layer4 = self.film_layer4(x_layer4, lang_embed)

        x_pooled = self.resnet.avgpool(x_layer4)
        x_flat = torch.flatten(x_pooled, 1)
        feature_emb = self.resnet.fc(x_flat)

        return feature_emb, x_layer4

    # ── Core forward ────────────────────────────────────────

    def forward_model(self, driving_input: DrivingInput):
        B = driving_input.camera_images.size(0)
        device = driving_input.camera_images.device

        # ── Preprocess images ──
        pv = driving_input.camera_images  # [B, 1, NP, 3, H, W] float [0,1]
        pv_flat = pv.reshape(-1, pv.shape[-3], pv.shape[-2], pv.shape[-1])
        pv_norm = (pv_flat - self.im_mean) / self.im_std
        NP = pv_flat.shape[0] // B

        # ── Language embedding (extract BEFORE ResNet if FiLM is used) ──
        lang_embed = None
        if self.use_lang or self.use_resnet_film:
            lang_embed = self._extract_lang_embed(driving_input)  # [B, lang_dim]

        # ── ResNet visual features (with optional FiLM conditioning) ──
        if self.use_resnet_film and lang_embed is not None:
            lang_exp = lang_embed.unsqueeze(1).expand(-1, NP, -1).reshape(B * NP, -1)
            feature_emb, cnn_feature = self._run_resnet_film(pv_norm, lang_exp)
        else:
            feature_emb, cnn_feature = self.resnet(pv_norm)

        # Average over camera views back to [B, ...]
        feature_emb = feature_emb.view(B, NP, -1).mean(dim=1)  # [B, 1000]
        cnn_feature = cnn_feature.view(B, NP, *cnn_feature.shape[1:]).mean(dim=1)  # [B, 512, h, w]
        vis_pooled = cnn_feature.flatten(2).mean(dim=2)  # [B, 512]

        # ── State encoding ──
        speed = driving_input.vehicle_speed.view(B, -1)[:, :1].to(device)  # [B, 1]
        tp = driving_input.target_point.view(B, -1)[:, :2].to(device)       # [B, 2]

        if self.use_cmd:
            cmd_texts = driving_input.prompt.language_string
            cmd_onehot = torch.zeros(B, 6, device=device)
            for i, s in enumerate(cmd_texts):
                for j, kw in enumerate(['left', 'right', 'straight', 'follow', 'lane change', 'lane']):
                    if kw in s.lower():
                        cmd_onehot[i, min(j, 5)] = 1.0
            state = torch.cat([speed, tp, cmd_onehot], dim=1)  # [B, 9]
        else:
            state = torch.cat([speed, tp], dim=1)  # [B, 3]

        state_embed = self.measurements(state)  # [B, state_dim]

        # ── Language embedding (already extracted above; compute fusion if needed) ──
        fused_embed = None
        if self.use_lang and lang_embed is not None:
            fused_embed = self._compute_fused(lang_embed, state_embed)

        # ── Join ──
        j_in = self._get_join_input(feature_emb, vis_pooled, state_embed, lang_embed, fused_embed)
        j = self.join(j_in)

        # ── Spatial attention ──
        B, C_cnn, H_cnn, W_cnn = cnn_feature.shape
        cnn_flat = cnn_feature.flatten(2).transpose(1, 2)  # [B, N_spatial, 512]
        att_in = self._get_att_input(cnn_flat, state_embed, lang_embed, fused_embed)
        att_w = self.att_fc(att_in).squeeze(-1).softmax(dim=1)
        att_vis = (cnn_flat * att_w.unsqueeze(-1)).sum(dim=1)  # [B, 512]

        # ── GRU trajectory decoder ──
        x = j
        z = torch.zeros(B, 2, device=device, dtype=j.dtype)
        pred_wp_list = []
        for _ in range(self.pred_len - 1):
            gru_in = self._get_gru_input(z, tp, state_embed, lang_embed, fused_embed)
            x = self.traj_gru(gru_in, x)
            dz = self.traj_out(x)
            z = z + dz
            pred_wp_list.append(z)
        pred_wp = torch.stack(pred_wp_list, dim=1)  # [B, pred_len-1, 2]

        # ── Route head ──
        route_in = self._get_route_input(j, att_vis, state_embed, lang_embed, fused_embed)
        route_flat = self.route_head(route_in)
        pred_route = route_flat.view(B, -1, 2).cumsum(dim=1)

        return pred_wp, pred_route

    # ── Training ────────────────────────────────────────────

    def forward(self, example, return_language=None, prompt_ids=None):
        try:
            di = example.driving_input
        except AttributeError:
            di = example
        pred_wp, pred_route = self.forward_model(di)
        B = pred_wp.size(0)
        return pred_wp, pred_route, [""] * B

    def forward_loss(self, example: DrivingExample):
        pred_wp, pred_route = self.forward_model(example.driving_input)
        gt_wp = example.driving_label.waypoints[:, :pred_wp.size(1)]
        gt_route = example.driving_label.path[:, :pred_route.size(1)]

        wp_loss = F.smooth_l1_loss(pred_wp, gt_wp, reduction="none").sum(-1)
        route_loss = F.smooth_l1_loss(pred_route, gt_route, reduction="none").sum(-1)
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

        name = getattr(self, 'version_name', 'tcp')
        save_dir = Path(f"visualise_{name}")
        save_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        ax.plot(gt_wp[:, 0], gt_wp[:, 1], 'go-', label='GT', markersize=5, linewidth=2)
        ax.plot(pw[:, 0], pw[:, 1], 'bo-', label='Pred', markersize=5, linewidth=2)
        ax.scatter(gt_wp[0, 0], gt_wp[0, 1], c='green', s=120, marker='*', zorder=5)
        ax.scatter(pw[0, 0], pw[0, 1], c='blue', s=120, marker='*', zorder=5)
        ax.set_title(f'Waypoints (step {step})')
        ax.legend(); ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')

        ax = axes[1]
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
