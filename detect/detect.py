from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from battle_detect import (
    GRID_BG_COLOR,
    TEXT_PANEL_WIDTH,
    build_grid_layout,
    find_cell_id,
    iter_videos,
    render_grid_frame,
)
from katacr.constants.label_list import idx2unit
from katacr.yolov8.combo_detect import ComboDetector, path_detectors

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


def render_card_frame(bottom_part: np.ndarray, classifier: YOLO) -> np.ndarray:
    canvas = np.zeros_like(bottom_part)
    for region_name, coords in CARD_REGIONS:
        crop = crop_region(bottom_part, coords)
        label = classify_card(crop, classifier)
        display_label = format_label(region_name, label)
        draw_label(canvas, coords, display_label)
    return canvas


def process_video(path_video: Path, combo: ComboDetector, classifier: YOLO, output_dir: Path) -> None:
    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {path_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if height <= 0 or width <= 0:
        print(f"[WARN] 视频尺寸异常: {path_video}")
        cap.release()
        return

    top_h = int(height * 0.8)
    bottom_h = height - top_h

    output_path = output_dir / f"{path_video.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 3, height))

    if combo.tracker is not None:
        combo.tracker.reset()

    cells, cell_by_id, grid_lines = build_grid_layout()
    pad_width = width - TEXT_PANEL_WIDTH
    if pad_width < 0:
        pad_width = 0
    max_event_lines = max((top_h - 2 * 12) // 22, 1)
    event_lines: list[str] = []
    pending_elixir: list[tuple[int, int]] = []
    last_card_state: dict[str, str] = {}
    last_change_label = "unknown"
    last_change_frame = -9999
    last_clock_frame = -9999

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        top_part = frame[:top_h, :]
        bottom_part = frame[top_h:, :]

        pred = combo.infer(top_part)
        battle_frame = pred.show_box(verbose=False, show_conf=True)
        detections = pred.get_data()
        elixir_cells = []
        clock_cells = []
        for det in detections:
            cls_idx = int(det[-2])
            unit_name = idx2unit.get(cls_idx, str(cls_idx))
            if unit_name not in {"elixir", "clock"}:
                continue
            x0, y0, x1, y1 = det[:4]
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            cell_id = find_cell_id(cells, cx, cy)
            if cell_id is None:
                continue
            if unit_name == "elixir":
                elixir_cells.append(cell_id)
            else:
                clock_cells.append(cell_id)

        if elixir_cells:
            for cell_id in elixir_cells:
                pending_elixir.append((frame_idx, cell_id))

        card_labels: dict[str, str] = {}
        for region_name, coords in CARD_REGIONS:
            if region_name not in {"1", "2", "3", "4"}:
                continue
            crop = crop_region(bottom_part, coords)
            label = classify_card(crop, classifier)
            card_labels[region_name] = label

        changed_label = None
        for region_name, label in card_labels.items():
            if last_card_state.get(region_name) != label:
                changed_label = format_label(region_name, label)
                last_change_label = changed_label
                last_change_frame = frame_idx
                break
        last_card_state = card_labels

        if pending_elixir and changed_label:
            frame_limit = frame_idx - 6
            pending_elixir = [(f, c) for f, c in pending_elixir if f >= frame_limit]
            if pending_elixir:
                _, cell_id = pending_elixir.pop(0)
                event_lines.append(f"me use card {changed_label} place on {cell_id}")
                if len(event_lines) > max_event_lines:
                    event_lines = event_lines[-max_event_lines:]

        if clock_cells and not elixir_cells and frame_idx - last_clock_frame > 6:
            card_name = last_change_label if frame_idx - last_change_frame <= 6 else "unknown"
            event_lines.append(f"enemy use card {card_name} place on {clock_cells[0]}")
            last_clock_frame = frame_idx
            if len(event_lines) > max_event_lines:
                event_lines = event_lines[-max_event_lines:]

        grid_frame = render_grid_frame(
            top_h,
            width,
            cells,
            cell_by_id,
            grid_lines,
            detections,
            event_lines=event_lines,
        )
        if pad_width:
            grid_frame = np.concatenate(
                [grid_frame, np.full((top_h, pad_width, 3), GRID_BG_COLOR, dtype=np.uint8)],
                axis=1,
            )

        card_canvas = render_card_frame(bottom_part, classifier)
        card_frame = np.hstack([card_canvas, card_canvas])
        right_panel = np.concatenate([grid_frame, card_frame], axis=0)
        left_panel = np.concatenate([battle_frame, bottom_part], axis=0)
        output_frame = np.concatenate([left_panel, right_panel], axis=1)

        writer.write(output_frame)

        if frame_idx % 100 == 0:
            print(f"{path_video.name} 已处理 {frame_idx} 帧")

    cap.release()
    writer.release()
    print(f"完成: {path_video.name} -> {output_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    video_dir = repo_root / "video"
    output_dir = repo_root / "output" / "all"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    combo = ComboDetector(path_detectors, show_conf=True, conf=0.7, iou_thre=0.6, tracker="bytetrack")
    classifier = load_classifier(repo_root)

    videos = iter_videos(video_dir)
    if not videos:
        print(f"未找到视频文件: {video_dir}")

    for video in videos:
        process_video(video, combo, classifier, output_dir)
