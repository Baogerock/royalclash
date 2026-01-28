from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F

DIGIT_FOLDERS = [f"{i:02d}" for i in range(10)]
DIGIT_WEIGHTS = [(1, 0.05), (2, 0.15), (3, 0.30), (4, 0.50)]


@dataclass
class GeneratedSample:
    image_path: str
    bbox: Tuple[int, int, int, int]
    label: str


class NumberDataset(Dataset):
    def __init__(self, samples: List[GeneratedSample], label_map: Dict[str, int]):
        self.samples = samples
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取训练图片: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image)
        x1, y1, x2, y2 = sample.bbox
        target = {
            "boxes": torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            "labels": torch.tensor([self.label_map[sample.label]], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image_tensor, target


def ensure_digit_dirs(data_dir: Path) -> None:
    missing = [name for name in DIGIT_FOLDERS if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"模板目录缺失: {', '.join(missing)}")


def load_templates(data_dir: Path) -> Dict[str, List[Path]]:
    templates: Dict[str, List[Path]] = {}
    for name in DIGIT_FOLDERS:
        folder = data_dir / name
        images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if not images:
            raise FileNotFoundError(f"模板目录为空: {folder}")
        templates[name] = images
    return templates


def choose_length() -> int:
    lengths, weights = zip(*DIGIT_WEIGHTS, strict=True)
    return random.choices(list(lengths), weights=list(weights), k=1)[0]


def sample_digits(length: int) -> List[str]:
    digits = [str(random.randint(1, 9))]
    for _ in range(1, length):
        digits.append(str(random.randint(0, 9)))
    return digits


def resize_to_height(image: torch.Tensor, target_height: int) -> torch.Tensor:
    height, width = image.shape[:2]
    scale = target_height / height
    new_width = max(int(width * scale), 1)
    return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)


def compose_number(
    digits: List[str],
    templates: Dict[str, List[Path]],
    target_height: int,
    spacing: int,
):
    digit_images = []
    for digit in digits:
        folder = f"{int(digit):02d}"
        template_path = random.choice(templates[folder])
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            raise FileNotFoundError(f"无法读取模板: {template_path}")
        digit_images.append(resize_to_height(template, target_height))

    total_width = sum(img.shape[1] for img in digit_images) + spacing * (len(digit_images) - 1)
    number_strip = 255 * np.ones((target_height, total_width, 3), dtype=np.uint8)
    cursor = 0
    for img in digit_images:
        number_strip[:, cursor : cursor + img.shape[1]] = img
        cursor += img.shape[1] + spacing

    pad_x = random.randint(6, 18)
    pad_y = random.randint(6, 18)
    canvas_h = target_height + pad_y * 2
    canvas_w = total_width + pad_x * 2
    canvas = 255 * np.ones((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + target_height, pad_x : pad_x + total_width] = number_strip
    bbox = (pad_x, pad_y, pad_x + total_width, pad_y + target_height)
    return canvas, bbox


def generate_dataset(
    data_dir: Path,
    output_dir: Path,
    num_samples: int,
    target_height: int,
    spacing: int,
) -> Tuple[List[GeneratedSample], Dict[str, int]]:
    templates = load_templates(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    samples: List[GeneratedSample] = []
    label_map: Dict[str, int] = {}
    for idx in range(num_samples):
        length = choose_length()
        digits = sample_digits(length)
        label = "".join(digits)
        if label not in label_map:
            label_map[label] = len(label_map) + 1
        image, bbox = compose_number(digits, templates, target_height, spacing)
        image_path = image_dir / f"{idx:05d}.png"
        cv2.imwrite(str(image_path), image)
        samples.append(GeneratedSample(str(image_path), bbox, label))

    label_path = output_dir / "label_map.json"
    with label_path.open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, ensure_ascii=False, indent=2)
    annotations_path = output_dir / "annotations.json"
    with annotations_path.open("w", encoding="utf-8") as handle:
        json.dump([sample.__dict__ for sample in samples], handle, ensure_ascii=False, indent=2)
    return samples, label_map


def load_dataset(output_dir: Path) -> Tuple[List[GeneratedSample], Dict[str, int]]:
    annotations_path = output_dir / "annotations.json"
    label_path = output_dir / "label_map.json"
    if not annotations_path.exists() or not label_path.exists():
        raise FileNotFoundError("未找到生成的数据集，请先生成训练数据。")
    with annotations_path.open("r", encoding="utf-8") as handle:
        raw_samples = json.load(handle)
    with label_path.open("r", encoding="utf-8") as handle:
        label_map = json.load(handle)
    samples = [GeneratedSample(**sample) for sample in raw_samples]
    return samples, label_map


def collate_fn(batch):
    return tuple(zip(*batch))


def train(
    samples: List[GeneratedSample],
    label_map: Dict[str, int],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    device: str,
):
    dataset = NumberDataset(samples, label_map)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=len(label_map) + 1)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0005)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for images, targets in progress:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            if torch.isnan(losses) or torch.isinf(losses):
                progress.set_postfix(loss="nan/inf")
                continue
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += losses.item()
            progress.set_postfix(loss=f"{losses.item():.4f}")
        progress.close()
        print(f"Epoch {epoch + 1}/{epochs} Loss: {epoch_loss / len(data_loader):.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "number_rcnn.pt"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model for digit number detection.")
    parser.add_argument("--data", type=Path, default=Path("train/train_number/data"), help="模板数据目录(00-09).")
    parser.add_argument("--output", type=Path, default=Path("train/train_number/generated"), help="生成数据输出目录.")
    parser.add_argument("--num-samples", type=int, default=5000, help="生成的训练样本数量.")
    parser.add_argument("--height", type=int, default=64, help="数字拼接后的目标高度.")
    parser.add_argument("--spacing", type=int, default=0, help="数字之间的间距.")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备.")
    parser.add_argument("--skip-generate", action="store_true", help="跳过数据生成，直接训练.")
    args = parser.parse_args()

    ensure_digit_dirs(args.data)
    if args.skip_generate:
        samples, label_map = load_dataset(args.output)
    else:
        samples, label_map = generate_dataset(
            args.data,
            args.output,
            args.num_samples,
            args.height,
            args.spacing,
        )
    train(samples, label_map, args.output, args.epochs, args.batch, args.device)


if __name__ == "__main__":
    main()
