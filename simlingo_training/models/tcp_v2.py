"""
TCP v2 — ResNet-34 TCP + SimLingo lang_embed 分散注入。

vs v1：lang_embed 同时注入 Join、Attention、GRU、Route Head。
每个模块独立感知语言语义。
"""

import torch
from torch import nn

from simlingo_training.models._base_tcp import BaseResNetTCP


class TCPModelV2(BaseResNetTCP):
    """TCP + language embedding injected into Join, Attention, GRU, and Route."""

    use_lang = True
    use_cmd = True
    version_name = 'tcp_v2'

    def _build_decoder(self):
        H = 1000
        S = self.state_dim  # 128
        L = getattr(self, 'lang_dim', 128)

        self.lang_proj = nn.Sequential(
            nn.Linear(self.language_model.hidden_size, L * 2),
            nn.ReLU(inplace=True),
            nn.Linear(L * 2, L),
        )

        self.join = nn.Sequential(
            nn.Linear(H + 512 + S + L, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
        )

        self.att_fc = nn.Sequential(
            nn.Linear(512 + S + L, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.traj_gru = nn.GRUCell(2 + 2 + S + L, 256)
        self.traj_out = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

        self.route_head = nn.Sequential(
            nn.Linear(256 + 512 + S + L, 512), nn.SiLU(inplace=True),
            nn.Linear(512, 256), nn.SiLU(inplace=True),
            nn.Linear(256, self.cfg_data_module.get('num_route_points', 20) * 2),
        )

    def _get_join_input(self, feature_emb, vis_pooled, state_embed, lang_embed, fused_embed):
        return torch.cat([feature_emb, vis_pooled, state_embed, lang_embed], dim=1)

    def _get_gru_input(self, x, tp, state_embed, lang_embed, fused_embed):
        return torch.cat([x, tp, state_embed, lang_embed], dim=1)

    def _get_att_input(self, cnn_flat, state_embed, lang_embed, fused_embed):
        B, N, _ = cnn_flat.shape
        state_exp = state_embed.unsqueeze(1).expand(-1, N, -1)
        lang_exp = lang_embed.unsqueeze(1).expand(-1, N, -1)
        return torch.cat([cnn_flat, state_exp, lang_exp], dim=-1)

    def _get_route_input(self, j, att_vis, state_embed, lang_embed, fused_embed):
        return torch.cat([j, att_vis, state_embed, lang_embed], dim=1)
