from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO, settings


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    settings.update({"wandb": False})
    parser = argparse.ArgumentParser(description="Train a small YOLO classifier for card recognition.")
    parser.add_argument("--data", type=Path, default=Path("data/card"), help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=96, help="Image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="YOLO classification model.")
    parser.add_argument("--val", action="store_true", help="Enable validation split if available.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("train/train_card/runs"),
        help="Project directory for training outputs.",
    )
    parser.add_argument("--name", type=str, default="classify", help="Run name.")
    args = parser.parse_args()

    data_dir = args.data
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    val_dir = data_dir / "val"
    use_val = args.val and val_dir.exists()
    split = "val" if use_val else "train"

    model = YOLO(args.model)
    model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        val=use_val,
        split=split,
    )


if __name__ == "__main__":
    main()
