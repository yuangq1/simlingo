"""
TCP v5-d — language-conditioned ResNet layer1-4 + v3 fusion decoder（全模型语言条件化）。

  RGB → ResNet-34 with FiLM(lang) after ALL stages (layer1-4)
  Decoder: v3 fusion（lang_embed+state_embed → fused_embed, 感知/规划分离）
  目的: v3 后端融合 + 前端 FiLM 的联合效果
"""

import torch
from torch import nn

from simlingo_training.models._base_tcp import BaseResNetTCP


class TCPModelV5D(BaseResNetTCP):
    """Language-conditioned ResNet layer1-4 + v3 fusion decoder."""

    use_lang = True
    use_cmd = True
    use_resnet_film = True
    resnet_film_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    version_name = 'tcp_v5d'

    def _build_decoder(self):
        H = 1000
        S = self.state_dim  # 128
        L = getattr(self, 'lang_dim', 128)
        F = getattr(self, 'fused_dim', 256)

        self.lang_proj = nn.Sequential(
            nn.Linear(self.language_model.hidden_size, L * 2),
            nn.ReLU(inplace=True),
            nn.Linear(L * 2, L),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(L + S, F), nn.ReLU(inplace=True),
            nn.Linear(F, F), nn.ReLU(inplace=True),
        )

        # Join 使用 lang+state（不预融合）
        self.join = nn.Sequential(
            nn.Linear(H + 512 + L + S, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
        )

        # Attention 仅用 lang_embed（感知阶段）
        self.att_fc = nn.Sequential(
            nn.Linear(512 + L, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.traj_gru = nn.GRUCell(2 + 2 + F, 256)
        self.traj_out = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        self.route_head = nn.Sequential(
            nn.Linear(256 + 512 + F, 512), nn.SiLU(inplace=True),
            nn.Linear(512, 256), nn.SiLU(inplace=True),
            nn.Linear(256, self.cfg_data_module.get('num_route_points', 20) * 2),
        )

    def _compute_fused(self, lang_embed, state_embed):
        return self.fusion_mlp(torch.cat([lang_embed, state_embed], dim=1))

    def _get_join_input(self, feature_emb, vis_pooled, state_embed, lang_embed, fused_embed):
        return torch.cat([feature_emb, vis_pooled, lang_embed, state_embed], dim=1)

    def _get_gru_input(self, x, tp, state_embed, lang_embed, fused_embed):
        return torch.cat([x, tp, fused_embed], dim=1)

    def _get_att_input(self, cnn_flat, state_embed, lang_embed, fused_embed):
        B, N, _ = cnn_flat.shape
        lang_exp = lang_embed.unsqueeze(1).expand(-1, N, -1)
        return torch.cat([cnn_flat, lang_exp], dim=-1)

    def _get_route_input(self, j, att_vis, state_embed, lang_embed, fused_embed):
        return torch.cat([j, att_vis, fused_embed], dim=1)
