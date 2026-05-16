"""
Waypoint prediction evaluation script for SimLingo dreamer models.
Computes ADE / FDE for waypoints and route, plus language accuracy.

Usage:
    cd ~/projects/SimLingo/simlingo
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    PYTHONPATH=$PWD python simlingo_training/eval_waypoints.py \
        experiment=dreamer_no_safety \
        checkpoint_path=outputs/2026_05_07_15_39_29_dreamer_no_safety/checkpoints/epoch=000.ckpt
"""
import json
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from hydra.utils import get_original_cwd
from simlingo_training.config import TrainConfig
import pytorch_lightning as pl
from transformers import AutoProcessor


def compute_ade_fde(pred, gt):
    """pred, gt: [N, T, 2]"""
    errors = np.linalg.norm(pred - gt, axis=-1)  # [N, T]
    ade = np.mean(errors)                         # mean over all timesteps and samples
    fde = np.mean(errors[:, -1])                  # mean over final timestep only
    return float(ade), float(fde)


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    pl.seed_everything(cfg.seed, workers=True)

    checkpoint_path = cfg.checkpoint_path
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(get_original_cwd(), checkpoint_path)
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")

    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant, trust_remote_code=True
    )

    data_module = hydra.utils.instantiate(
        cfg.data_module,
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False,
    )
    data_module.setup()

    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=None,
        _recursive_=False,
    )

    # Load checkpoint weights (handles both raw state_dict and PL checkpoint format)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    val_loader = data_module.val_dataloader()
    total = len(val_loader)
    print(f"Validation batches: {total} (batch_size={cfg.data_module.batch_size})")

    all_wps_pred, all_wps_gt = [], []
    all_route_pred, all_route_gt = [], []
    all_lang_pred, all_lang_gt = [], []

    def _to_device(obj):
        """Recursively move all tensors to `device`."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            if hasattr(obj, '_fields'):  # NamedTuple
                return type(obj)(**{f: _to_device(getattr(obj, f)) for f in obj._fields})
            return tuple(_to_device(v) for v in obj)
        return obj

    for batch in tqdm(val_loader, total=total, desc="Evaluating"):
        batch = _to_device(batch)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            speed_wps, route, language = model.forward(batch, return_language=True)

        # Waypoints: [B, 11, 2]
        if speed_wps is not None:
            all_wps_pred.append(speed_wps.cpu().numpy())
            all_wps_gt.append(batch.driving_label.waypoints.cpu().numpy())

        # Route: [B, 20, 2]
        if route is not None:
            all_route_pred.append(route.cpu().numpy())
            all_route_gt.append(batch.driving_label.path[:, :20].cpu().numpy())

        # Language
        if language:
            gt_strings = batch.driving_label.answer.language_string
            for pred_lang, gt_lang in zip(language, gt_strings):
                all_lang_pred.append(pred_lang)
                all_lang_gt.append(gt_lang)

    # --- Aggregate & compute metrics ---
    results = {"checkpoint": checkpoint_path}

    if all_wps_pred:
        pred_wps = np.concatenate(all_wps_pred, axis=0)  # [N, 11, 2]
        gt_wps = np.concatenate(all_wps_gt, axis=0)
        # Skip degenerate GT samples (all zeros)
        valid = gt_wps.max(axis=(1, 2)) > 0.01
        ade_wps, fde_wps = compute_ade_fde(pred_wps[valid], gt_wps[valid])
        results["waypoints_ADE"] = round(ade_wps, 4)
        results["waypoints_FDE"] = round(fde_wps, 4)
        results["waypoints_valid_samples"] = int(valid.sum())
        results["waypoints_total_samples"] = len(valid)
        print(f"Waypoints ADE: {ade_wps:.4f} m, FDE: {fde_wps:.4f} m "
              f"({int(valid.sum())}/{len(valid)} valid samples)")

    if all_route_pred:
        pred_route = np.concatenate(all_route_pred, axis=0)
        gt_route = np.concatenate(all_route_gt, axis=0)
        valid = gt_route.max(axis=(1, 2)) > 0.01
        ade_route, fde_route = compute_ade_fde(pred_route[valid], gt_route[valid])
        results["route_ADE"] = round(ade_route, 4)
        results["route_FDE"] = round(fde_route, 4)
        results["route_valid_samples"] = int(valid.sum())
        results["route_total_samples"] = len(valid)
        print(f"Route ADE: {ade_route:.4f} m, FDE: {fde_route:.4f} m "
              f"({int(valid.sum())}/{len(valid)} valid samples)")

    if all_lang_pred:
        correct = sum(p == g for p, g in zip(all_lang_pred, all_lang_gt))
        lang_acc = correct / len(all_lang_pred)
        results["language_accuracy"] = round(lang_acc, 4)
        results["language_total_samples"] = len(all_lang_pred)
        print(f"Language Accuracy: {lang_acc:.4f} ({correct}/{len(all_lang_pred)})")

    # Save results
    out_dir = os.path.dirname(checkpoint_path)
    out_path = os.path.join(out_dir, f"eval_metrics_{os.path.basename(checkpoint_path)}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Print summary table for easy comparison
    print("\n" + "=" * 50)
    print("SUMMARY (for model comparison)")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
