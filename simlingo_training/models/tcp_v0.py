"""
TCP v0 — 原始 ResNet-34 TCP baseline（无语言注入）。

输入：
  - RGB image → ResNet-34 → feature_emb(1000) + cnn_feature(512,H,W)
  - speed(1) + target_point(2) + command one-hot(6) → measurement(128)
输出：
  - pred_wp: [B, 10, 2]  waypoints
  - pred_route: [B, 20, 2]  route points
"""

import torch
from torch import nn

from simlingo_training.models._base_tcp import BaseResNetTCP


class TCPModelV0(BaseResNetTCP):
    """原始 TCP：ResNet-34 视觉 + state encoder，无语言模块。"""

    use_lang = False
    use_cmd = True
    version_name = 'tcp_v0'

    def _build_decoder(self):
        H = 1000  # ResNet FC dim
        S = self.state_dim  # 128

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
