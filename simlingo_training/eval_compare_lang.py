"""
对比三种语言增强配置的评估结果。
"""

import sys, os
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from hydra.utils import get_original_cwd
from transformers import AutoProcessor

# 触发 config.py 中的 register_configs()
import simlingo_training.config  # noqa


def compute_ade_fde(pred, gt):
    errors = np.linalg.norm(pred - gt, axis=-1)
    return float(np.mean(errors)), float(np.mean(errors[:, -1]))


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor): return obj.to(device)
    if isinstance(obj, dict): return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list): return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(**{f: _to_device(getattr(obj, f), device) for f in obj._fields})
    return obj


def set_rewriter_prob(datasets, prob):
    """递归遍历 datasets 找到所有 ColloquialRewriter 并设 prob。"""
    for ds in datasets:
        if hasattr(ds, 'lang_rewriter') and ds.lang_rewriter is not None:
            ds.lang_rewriter.prob = prob
        # ConcatDataset
        if hasattr(ds, 'datasets'):
            set_rewriter_prob(ds.datasets, prob)


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    ckpt = cfg.checkpoint_path
    if ckpt is None:
        print("ERROR: checkpoint_path=... required")
        return
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(get_original_cwd(), ckpt)
    print(f"Checkpoint: {ckpt}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant, trust_remote_code=True)

    # Build data module once
    dm = hydra.utils.instantiate(
        cfg.data_module, processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False)
    dm.setup()

    # Build model once
    model = hydra.utils.instantiate(
        cfg.model, cfg_data_module=cfg.data_module,
        processor=processor, cache_dir=None, _recursive_=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("state_dict", sd), strict=False)
    model = model.to(device).eval()

    all_r = {}
    for prob, label in [(0.0, "无口语映射"), (0.5, "50%口语映射"), (1.0, "100%口语映射")]:
        print(f"\n{'='*50}\n  {label}\n{'='*50}")

        # 设置所有 dataset 的 rewriter prob
        for ds in [dm.val_dataset] if hasattr(dm, 'val_dataset') else []:
            set_rewriter_prob([ds], prob)

        loader = dm.val_dataloader()
        wp_p, wp_g, rt_p, rt_g = [], [], [], []

        for batch in tqdm(loader, desc=f"prob={prob}", leave=False):
            batch = _to_device(batch, device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                w, r, _ = model.forward(batch, return_language=True)
            if w is not None:
                wp_p.append(w.cpu().numpy())
                wp_g.append(batch.driving_label.waypoints.cpu().numpy())
            if r is not None:
                rt_p.append(r.cpu().numpy())
                rt_g.append(batch.driving_label.path[:, :20].cpu().numpy())

        r = {}
        if wp_p:
            p = np.concatenate(wp_p, axis=0); g = np.concatenate(wp_g, axis=0)
            v = g.max(axis=(1, 2)) > 0.01
            a, f = compute_ade_fde(p[v], g[v])
            r.update({"Waypoints ADE": round(a, 4), "Waypoints FDE": round(f, 4)})
        if rt_p:
            p = np.concatenate(rt_p, axis=0); g = np.concatenate(rt_g, axis=0)
            v = g.max(axis=(1, 2)) > 0.01
            a, f = compute_ade_fde(p[v], g[v])
            r.update({"Route ADE": round(a, 4), "Route FDE": round(f, 4)})

        all_r[label] = r
        for k, v in r.items():
            print(f"  {k}: {v}")

    print(f"\n{'='*65}")
    print("  对比总结")
    print(f"{'='*65}")
    header = f"{'指标':<20}"
    for l in all_r: header += f" {l:<16}"
    print(header)
    print("-" * 65)
    for m in ["Waypoints ADE", "Waypoints FDE", "Route ADE", "Route FDE"]:
        row = f"{m:<20}"
        for l in all_r:
            v = all_r[l].get(m, "-")
            row += f" {str(v):<16}"
        print(row)


if __name__ == "__main__":
    main()
