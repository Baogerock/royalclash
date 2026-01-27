from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

CARD_REGIONS = [
    ("1", ((157, 34), (292, 197))),
    ("2", ((292, 34), (427, 197))),
    ("3", ((428, 34), (562, 197))),
    ("4", ((562, 34), (697, 197))),
    ("next", ((34, 164), (101, 246))),
    ("water", ((196, 193), (257, 238))),
]

REGION_TEMPLATE_DIRS = {
    "1": "card",
    "2": "card",
    "3": "card",
    "4": "card",
    "next": "next",
    "water": "water",
}

CARD_NAME_MAP = {
    "00": "觉醒卫队",
    "01": "牢笼",
    "02": "卫猪",
    "03": "电豆",
    "04": "飞机",
    "05": "电车",
    "06": "卫队",
    "07": "蛮桶",
    "08": "火球",
    "09": "觉醒卫猪",
}


@dataclass(frozen=True)
class TemplateItem:
    name: str
    image: np.ndarray


def iter_videos(video_dir: Path) -> list[Path]:
    return sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def iter_images(dir_path: Path) -> list[Path]:
    return sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )


def load_templates(template_dir: Path) -> list[TemplateItem]:
    templates: list[TemplateItem] = []
    for path in iter_images(template_dir):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        templates.append(TemplateItem(path.stem, image))
    return templates


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


def match_template(crop: np.ndarray, templates: list[TemplateItem]) -> str:
    if crop.size == 0:
        return "?"
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    best_name = "?"
    best_score = -1.0
    for item in templates:
        template = item.image
        if template is None:
            continue
        if template.shape[0] > crop_gray.shape[0] or template.shape[1] > crop_gray.shape[1]:
            continue
        result = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_name = item.name
    return best_name


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
    return label


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    video_dir = repo_root / "output" / "card"
    template_root = repo_root / "template"

    if not video_dir.exists():
        raise FileNotFoundError(f"未找到视频目录: {video_dir}")

    templates_by_region: dict[str, list[TemplateItem]] = {}
    for region, template_dir in REGION_TEMPLATE_DIRS.items():
        dir_path = template_root / template_dir
        templates_by_region[region] = load_templates(dir_path)

    videos = iter_videos(video_dir)
    if not videos:
        print(f"未找到视频文件: {video_dir}")
        return

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
                label = match_template(crop, templates_by_region.get(region_name, []))
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
