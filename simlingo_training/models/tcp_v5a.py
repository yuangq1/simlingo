"""
TCP v5-a — language-conditioned ResNet layer4 + v0-style decoder（无语言注入 decoder）。

  RGB → ResNet-34 with FiLM(lang) after layer4 → v0 TCP decoder
  FiLM: 仅调制 ResNet 最高层语义特征 ([B, 512, H, W])
  Decoder: 沿用 v0，仅使用 state_embed（cmd+speed+tp），不注入语言
"""

import torch
from torch import nn

from simlingo_training.models._base_tcp import BaseResNetTCP


class TCPModelV5A(BaseResNetTCP):
    """Language-conditioned ResNet layer4 only; decoder = v0 (no lang)."""

    use_lang = False
    use_cmd = True
    use_resnet_film = True
    resnet_film_layers = ['layer4']
    version_name = 'tcp_v5a'

    def _build_decoder(self):
        H = 1000  # ResNet FC dim
        S = self.state_dim  # 128

        # lang_proj is required for FiLM (used in _extract_lang_embed)
        self.lang_proj = nn.Sequential(
            nn.Linear(self.language_model.hidden_size, 128),
        )

        self.join = nn.Sequential(
            nn.Linear(H + 512 + S, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
        )

        self.att_fc = nn.Sequential(
            nn.Linear(512 + S, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.traj_gru = nn.GRUCell(2 + 2 + S, 256)
        self.traj_out = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        self.route_head = nn.Sequential(
            nn.Linear(256 + 512 + S, 512), nn.SiLU(inplace=True),
            nn.Linear(512, 256), nn.SiLU(inplace=True),
            nn.Linear(256, self.cfg_data_module.get('num_route_points', 20) * 2),
        )

    def _get_join_input(self, feature_emb, vis_pooled, state_embed, lang_embed, fused_embed):
        return torch.cat([feature_emb, vis_pooled, state_embed], dim=1)

    def _get_gru_input(self, x, tp, state_embed, lang_embed, fused_embed):
        return torch.cat([x, tp, state_embed], dim=1)

    def _get_att_input(self, cnn_flat, state_embed, lang_embed, fused_embed):
        B, N, _ = cnn_flat.shape
        state_exp = state_embed.unsqueeze(1).expand(-1, N, -1)
        return torch.cat([cnn_flat, state_exp], dim=-1)

    def _get_route_input(self, j, att_vis, state_embed, lang_embed, fused_embed):
        return torch.cat([j, att_vis, state_embed], dim=1)
