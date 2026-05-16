"""
Plot GT vs Predicted waypoints and route on validation samples.
Usage:
    cd ~/projects/SimLingo/simlingo
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    PYTHONPATH=$PWD python simlingo_training/plot_val_samples.py \
        experiment=dreamer_no_safety \
        checkpoint_path=outputs/2026_05_07_15_39_29_dreamer_no_safety/checkpoints/last.ckpt

Output: checkpoints/eval_plots/ directory with comparison images.
"""
import os
import sys

import hydra
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import get_original_cwd
from tqdm import tqdm

from simlingo_training.config import TrainConfig
import pytorch_lightning as pl
from transformers import AutoProcessor


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

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    val_loader = data_module.val_dataloader()

    def _to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            if hasattr(obj, '_fields'):
                return type(obj)(**{f: _to_device(getattr(obj, f)) for f in obj._fields})
            return tuple(_to_device(v) for v in obj)
        return obj

    out_dir = os.path.join(os.path.dirname(checkpoint_path), "eval_plots")
    os.makedirs(out_dir, exist_ok=True)

    num_plots = 0
    for batch_idx, batch in enumerate(val_loader):
        if num_plots >= 8:
            break

        batch = _to_device(batch)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            speed_wps, route, language = model.forward(batch, return_language=True)

        gt_wps = batch.driving_label.waypoints.cpu().numpy()       # [B, 11, 2]
        gt_route = batch.driving_label.path[:, :20].cpu().numpy()   # [B, 20, 2]
        pred_wps = speed_wps.cpu().numpy() if speed_wps is not None else None
        pred_route = route.cpu().numpy() if route is not None else None

        bs = gt_wps.shape[0]
        for i in range(bs):
            if num_plots >= 8:
                break

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

            # --- Top: Waypoints ---
            ax1.plot(gt_wps[i, :, 1], gt_wps[i, :, 0], 'g-o', ms=6, label='GT', linewidth=2.5)
            if pred_wps is not None:
                ax1.plot(pred_wps[i, :, 1], pred_wps[i, :, 0], 'b-s', ms=6, label='Pred', linewidth=2.5)
                errs = np.linalg.norm(pred_wps[i] - gt_wps[i], axis=-1)
                ade = float(np.mean(errs))
                fde = float(errs[-1])
                ax1.set_title(f'Waypoints (11 points, 2.75s) — ADE={ade:.2f}m  FDE={fde:.2f}m', fontsize=14, fontweight='bold')
            else:
                ax1.set_title('Waypoints (11 points, 2.75s)', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=12, loc='upper right')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel('lateral (m)', fontsize=12)
            ax1.set_ylabel('longitudinal (m)', fontsize=12)
            # auto-scale lateral axis, fix longitudinal to show full range
            ax1.set_xlim(auto=True)

            # --- Bottom: Route ---
            ax2.plot(gt_route[i, :, 1], gt_route[i, :, 0], 'g-o', ms=6, label='GT', linewidth=2.5)
            if pred_route is not None:
                ax2.plot(pred_route[i, :, 1], pred_route[i, :, 0], 'b-s', ms=6, label='Pred', linewidth=2.5)
                errs_r = np.linalg.norm(pred_route[i] - gt_route[i], axis=-1)
                ade_r = float(np.mean(errs_r))
                fde_r = float(errs_r[-1])
                ax2.set_title(f'Route (20 points) — ADE={ade_r:.2f}m  FDE={fde_r:.2f}m', fontsize=14, fontweight='bold')
            else:
                ax2.set_title('Route (20 points)', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=12, loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel('lateral (m)', fontsize=12)
            ax2.set_ylabel('longitudinal (m)', fontsize=12)
            ax2.set_xlim(auto=True)

            # Text info
            gt_text = batch.driving_label.answer.language_string[i]
            pred_text = language[i] if language else ""
            fig.suptitle(f'Sample {num_plots+1} | GT: {gt_text}\nPred: {pred_text}',
                         fontsize=10, fontfamily='monospace')

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"sample_{num_plots:02d}.png"), dpi=150)
            plt.close(fig)

            num_plots += 1

    print(f"\nSaved {num_plots} plots to: {out_dir}")


if __name__ == "__main__":
    main()
