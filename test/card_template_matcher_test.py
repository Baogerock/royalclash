from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

CARD_REGIONS = [
    ("1", ((157, 34), (292, 197))),
    ("2", ((292, 34), (427, 197))),
    ("3", ((428, 34), (562, 197))),
    ("4", ((562, 34), (697, 197))),
    ("next", ((34, 164), (101, 246))),
    ("water", ((196, 193), (257, 238))),
]

CARD_NAME_MAP = {
    "00": "Awakened Guards",
    "01": "Cage",
    "02": "Hogs",
    "03": "ElecSpirit",
    "04": "Plane",
    "05": "ElecCar",
    "06": "Guards",
    "07": "Barrel",
    "08": "Fireball",
    "09": "Awakened Hogs",
}

WATER_NAME_MAP = {
    "10": "0",
    "11": "1",
    "12": "2",
    "13": "3",
    "14": "4",
    "15": "5",
    "16": "6",
    "17": "7",
    "18": "8",
    "19": "9",
    "20": "10",
}

CLASSIFIER_MODEL_ENV = "CARD_CLASSIFIER_MODEL"
DEFAULT_MODEL_PATH = Path("train/train_card/best.pt")

def iter_videos(video_dir: Path) -> list[Path]:
    return sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def crop_region(frame: np.ndarray, region: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    (x1, y1), (x2, y2) = region
    height, width = frame.shape[:2]
    left = max(0, min(width, x1))
    right = max(0, min(width, x2))
    top = max(0, min(height, y1))
    bottom = max(0, min(height, y2))
    if right <= left or bottom <= top:
        return np.empty((0, 0), dtype=frame.dtype)
    return frame[top:bottom, left:right]


def classify_card(crop: np.ndarray, model: YOLO) -> str:
    if crop.size == 0:
        return "?"
    results = model.predict(crop, verbose=False, task="classify")
    if not results:
        return "?"
    result = results[0]
    probs = result.probs
    if probs is None:
        return "?"
    top_index = int(probs.top1)
    names = result.names or {}
    return str(names.get(top_index, top_index))


def draw_label(canvas: np.ndarray, region: tuple[tuple[int, int], tuple[int, int]], label: str) -> None:
    (x1, y1), (x2, y2) = region
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    cv2.putText(canvas, label, (text_x, text_y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


def format_label(region_name: str, label: str) -> str:
    if region_name in {"1", "2", "3", "4", "next"}:
        return CARD_NAME_MAP.get(label, label)
    if region_name == "water":
        return WATER_NAME_MAP.get(label, label)
    return label


def resolve_model_path(candidate: Path) -> Path | None:
    if not candidate.exists():
        return None
    if candidate.is_file():
        return candidate
    weights_dir = candidate / "weights"
    for name in ("best.pt", "last.pt"):
        path = weights_dir / name
        if path.exists():
            return path
    for path in sorted(candidate.rglob("*.pt")):
        return path
    return None


def load_classifier(repo_root: Path) -> YOLO:
    env_path = Path(os.environ.get(CLASSIFIER_MODEL_ENV, "")).expanduser()
    model_path = resolve_model_path(env_path)
    if model_path is None:
        model_path = resolve_model_path(repo_root / DEFAULT_MODEL_PATH)
    if model_path is None:
        raise FileNotFoundError(
            f"未找到分类模型: {env_path}. 请先训练模型或设置 {CLASSIFIER_MODEL_ENV} 环境变量。"
        )
    return YOLO(str(model_path), task="classify")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    video_dir = repo_root / "output" / "card"

    if not video_dir.exists():
        raise FileNotFoundError(f"未找到视频目录: {video_dir}")

    videos = iter_videos(video_dir)
    if not videos:
        print(f"未找到视频文件: {video_dir}")
        return

    classifier = load_classifier(repo_root)

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] 无法打开视频: {video_path}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            canvas = np.zeros_like(frame)
            for region_name, coords in CARD_REGIONS:
                crop = crop_region(frame, coords)
                label = classify_card(crop, classifier)
                display_label = format_label(region_name, label)
                draw_label(canvas, coords, display_label)

            combined = np.hstack([frame, canvas])
            cv2.imshow(f"card-template-matcher: {video_path.name}", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord("n"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
