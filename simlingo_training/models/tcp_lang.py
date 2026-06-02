"""
TCP_lang 向后兼容入口 — 默认指向 v3-b2（当前版本）。

如需训练历史版本，请直接使用对应的版本文件：
  - tcp_v0.py        → 原始 TCP（无语言）
  - tcp_lang_v1.py   → VLM + command one-hot
  - tcp_lang_v2.py   → 无 command，state_dim=32
  - tcp_lang_v3a.py  → fusion_mlp 统一融合
  - tcp_lang_v3b1.py → LLM-output-only planning decoder
  - tcp_lang_v3b2.py → planning_embed + raw tp in GRU（当前版本）
"""

from simlingo_training.models.tcp_lang_v3b2 import TCPDecoderV3B2 as TCPDecoder
from simlingo_training.models.tcp_lang_v3b2 import TCPLangModelV3B2 as TCPLangModel
