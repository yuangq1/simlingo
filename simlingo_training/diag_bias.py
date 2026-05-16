"""
Quick diagnostic: lateral vs longitudinal bias in waypoint predictions.
"""
import os, sys, json
import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from tqdm import tqdm
from transformers import AutoProcessor
from simlingo_training.config import TrainConfig
import pytorch_lightning as pl


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    pl.seed_everything(cfg.seed, workers=True)

    checkpoint_path = cfg.checkpoint_path
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(get_original_cwd(), checkpoint_path)

    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    data_module = hydra.utils.instantiate(cfg.data_module, processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant, _recursive_=False)
    data_module.setup()

    model = hydra.utils.instantiate(cfg.model, cfg_data_module=cfg.data_module,
        processor=processor, cache_dir=None, _recursive_=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    def _to_device(obj):
        if isinstance(obj, torch.Tensor): return obj.to(device)
        if isinstance(obj, dict): return {k: _to_device(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            if hasattr(obj, '_fields'):
                return type(obj)(**{f: _to_device(getattr(obj, f)) for f in obj._fields})
            return tuple(_to_device(v) for v in obj)
        return obj

    val_loader = data_module.val_dataloader()

    all_lat_err = []   # lateral error (dim 1)
    all_lon_err = []   # longitudinal error (dim 0)
    all_wps_ade = []
    all_route_ade = []

    for batch in tqdm(val_loader, total=len(val_loader), desc="Diagnosing"):
        batch = _to_device(batch)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            speed_wps, route, _ = model.forward(batch, return_language=True)

        gt_wps = batch.driving_label.waypoints.cpu().numpy()
        gt_route = batch.driving_label.path[:, :20].cpu().numpy()
        pred_wps = speed_wps.cpu().numpy() if speed_wps is not None else None
        pred_route = route.cpu().numpy() if route is not None else None

        if pred_wps is not None:
            err_wps = pred_wps - gt_wps  # [B, 11, 2], dim0=lon, dim1=lat
            all_lon_err.append(err_wps[:, :, 0])  # longitudinal
            all_lat_err.append(err_wps[:, :, 1])  # lateral
            all_wps_ade.append(np.mean(np.linalg.norm(err_wps, axis=-1), axis=-1))

        if pred_route is not None:
            err_route = pred_route - gt_route
            all_route_ade.append(np.mean(np.linalg.norm(err_route, axis=-1), axis=-1))

    lon_err = np.concatenate([e.ravel() for e in all_lon_err])
    lat_err = np.concatenate([e.ravel() for e in all_lat_err])
    wps_ade = np.concatenate(all_wps_ade)
    route_ade = np.concatenate(all_route_ade)

    print("\n" + "=" * 60)
    print("BIAS DIAGNOSIS")
    print("=" * 60)
    print(f"  Longitudinal error (forward):  mean={lon_err.mean():+.3f}m  std={lon_err.std():.3f}m")
    print(f"  Lateral error (left/right):    mean={lat_err.mean():+.3f}m  std={lat_err.std():.3f}m")
    print()
    print(f"  Waypoints ADE distribution: min={wps_ade.min():.2f}  median={np.median(wps_ade):.2f}  max={wps_ade.max():.2f}")
    print(f"  Route ADE distribution:     min={route_ade.min():.2f}  median={np.median(route_ade):.2f}  max={route_ade.max():.2f}")
    print()
    # Interpret bias
    if abs(lat_err.mean()) > 0.2:
        direction = "RIGHT" if lat_err.mean() > 0 else "LEFT"
        print(f"  ⚠ Lateral bias: model steers {abs(lat_err.mean()):.2f}m to the {direction} on average")
    else:
        print(f"  ✓ No significant lateral bias")
    if abs(lon_err.mean()) > 0.3:
        bias_type = "OVERSHOOT" if lon_err.mean() > 0 else "UNDERSHOOT"
        print(f"  ⚠ Longitudinal bias: model predictions {bias_type} by {abs(lon_err.mean()):.2f}m on average")
    else:
        print(f"  ✓ No significant longitudinal bias")
    print("=" * 60)


if __name__ == "__main__":
    main()
