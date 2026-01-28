from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO, settings

DIGIT_CLASSES = [f"{i:02d}" for i in range(10)]


def ensure_digit_dirs(data_dir: Path) -> None:
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")
    missing = [name for name in DIGIT_CLASSES if not (train_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"训练集缺少类别目录: {', '.join(missing)}")


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    settings.update({"wandb": False})
    parser = argparse.ArgumentParser(description="Train a small YOLO classifier for digit recognition.")
    parser.add_argument("--data", type=Path, default=Path("data/number"), help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=64, help="Image size.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size.")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="YOLO classification model.")
    parser.add_argument("--val", action="store_true", help="Enable validation split if available.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("train/train_number/runs"),
        help="Project directory for training outputs.",
    )
    parser.add_argument("--name", type=str, default="classify", help="Run name.")
    args = parser.parse_args()

    data_dir = args.data
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    ensure_digit_dirs(data_dir)

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
