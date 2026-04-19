"""
Grad-CAM visualisation for a single DICOM slice.

Usage:
    python scripts/gradcam.py \
        --input  "data/raw/CQ500CT25 CQ500CT25/Unknown Study/CT Thin Plain/CT000110.dcm" \
        --checkpoint models/checkpoints/best_model.pt \
        --class-index 2          # 0=no_hemorrhage 1=epidural 2=intraparenchymal
                                 # 3=intraventricular 4=subarachnoid 5=subdural
        --output reports/figures/gradcam.png

Or via Makefile:
    make gradcam INPUT="..." CLASS=2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import hu_to_3channel, load_dicom_slice, normalize_image
from src.models.architectures.efficientnet import EfficientNetICH
from src.visualization.visualize import HEMORRHAGE_TYPES, plot_gradcam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM for ICH detection model")
    parser.add_argument("--input", required=True, help="Path to a DICOM (.dcm) file")
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="params.yaml with model config",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=None,
        help=(
            "Class index to explain (0-5). "
            "Defaults to the class with the highest predicted probability."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save figure to this path. If omitted, displays interactively.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (cpu / cuda). Auto-detected if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load params ───────────────────────────────────────────────────────────
    with open(args.params) as f:
        params = yaml.safe_load(f)

    model_cfg = params.get("model", {})
    model_name = model_cfg.get("name", "efficientnet_b4")
    num_classes = params["data"]["num_classes"]
    image_size = params["preprocessing"]["image_size"]
    mean = params["preprocessing"]["normalize_mean"]
    std = params["preprocessing"]["normalize_std"]

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    model = EfficientNetICH(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint: %s", ckpt_path)

    # ── Load & preprocess DICOM ───────────────────────────────────────────────
    dcm_path = Path(args.input)
    if not dcm_path.exists():
        logger.error("DICOM file not found: %s", dcm_path)
        sys.exit(1)

    hu_array, meta = load_dicom_slice(dcm_path)
    logger.info("Loaded DICOM: %s | SliceLocation=%.1f", dcm_path.name, meta.get("slice_location", 0))

    rgb = hu_to_3channel(hu_array)                          # (H, W, 3) in [0,1]
    rgb_norm = normalize_image(rgb, mean=mean, std=std)

    import torch.nn.functional as F
    tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    tensor = F.interpolate(tensor, size=image_size, mode="bilinear", align_corners=False)

    # ── Predict ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    logger.info("Predictions:")
    for cls, p in zip(HEMORRHAGE_TYPES, probs):
        logger.info("  %-22s %.3f", cls, p)

    target_class = args.class_index
    if target_class is None:
        target_class = int(np.argmax(probs))
        logger.info("Auto-selected class: %s (index %d)", HEMORRHAGE_TYPES[target_class], target_class)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    # Rebuild tensor with grad (can't use inference_mode here)
    tensor_grad = tensor.clone().requires_grad_(False)

    save_path = Path(args.output) if args.output else None
    plot_gradcam(
        model=model,
        image_tensor=tensor_grad,
        target_class=target_class,
        device=device,
        original_image=rgb,        # unnormalised for display
        class_name=HEMORRHAGE_TYPES[target_class],
        save_path=save_path,
    )

    if save_path:
        logger.info("Grad-CAM saved → %s", save_path)


if __name__ == "__main__":
    main()
