from pathlib import Path

import cv2
import numpy as np

from katacr.constants.label_list import idx2unit
from katacr.yolov8.combo_detect import ComboDetector, path_detectors

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

GRID_STEP = 34
MIN_CELL_RATIO = 0.5
GRID_LINE_COLOR = (0, 255, 255)
GRID_LINE_WIDTH = 2
GRID_FILL_ALPHA = 0.45
GRID_BG_COLOR = (12, 12, 12)
TEXT_PANEL_WIDTH = 360
TEXT_COLOR = (240, 240, 240)
TEXT_LINE_HEIGHT = 22
TEXT_MARGIN = 12

REGION2_ROWS = [
    (53, 137.0000, 666, 163.5385),
    (53, 163.5385, 666, 190.0769),
    (53, 190.0769, 666, 216.6154),
    (53, 216.6154, 666, 243.1538),
    (53, 243.1538, 666, 269.6923),
    (53, 269.6923, 666, 296.2308),
    (53, 296.2308, 666, 322.7692),
    (53, 322.7692, 666, 349.3077),
    (53, 349.3077, 666, 375.8462),
    (53, 375.8462, 666, 402.3846),
    (53, 402.3846, 666, 428.9231),
    (53, 428.9231, 666, 455.4615),
    (53, 455.4615, 666, 482.0000),
]

REGION5_ROWS = [
    (53, 586.0000, 666, 615),
    (53, 615, 666, 643),
    (53, 643, 666, 672),
    (53, 672, 666, 700),
    (53, 700, 666, 728),
    (53, 728, 666, 756),
    (53, 756, 666, 783),
    (53, 783, 666, 811),
    (53, 811, 666, 840),
    (53, 840, 666, 868),
    (53, 868, 666, 896),
    (53, 896, 666, 924),
    (53, 924, 666, 950.0000),
]

GRID_REGIONS = [
    (258, 112, 461, 137),
    *REGION2_ROWS,
    (85, 481, 633, 507),
    (85, 559, 632, 587),
    *REGION5_ROWS,
    (257, 950, 460, 980),
]


def iter_videos(video_dir: Path):
    return sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def make_lines_drop_small(a: float, b: float, step: float, min_ratio: float) -> list[float]:
    lines = [float(a)]
    x = float(a) + step
    while x < float(b):
        lines.append(x)
        x += step
    remainder = float(b) - lines[-1]
    if remainder >= step * min_ratio:
        lines.append(float(b))
    return lines


def build_grid_layout():
    grid_lines = []
    cells = []
    cell_by_id = {}
    gid = 0
    for (x0, y0, x1, y1) in GRID_REGIONS:
        x_lines = make_lines_drop_small(x0, x1, GRID_STEP, MIN_CELL_RATIO)
        y_lines = make_lines_drop_small(y0, y1, GRID_STEP, MIN_CELL_RATIO)
        if len(x_lines) < 2 or len(y_lines) < 2:
            continue
        for x in x_lines:
            grid_lines.append(((x, y_lines[0]), (x, y_lines[-1])))
        for y in y_lines:
            grid_lines.append(((x_lines[0], y), (x_lines[-1], y)))
        for r in range(len(y_lines) - 1):
            for c in range(len(x_lines) - 1):
                cells.append(
                    {
                        "id": gid,
                        "x0": x_lines[c],
                        "y0": y_lines[r],
                        "x1": x_lines[c + 1],
                        "y1": y_lines[r + 1],
                    }
                )
                cell_by_id[gid] = cells[-1]
                gid += 1
    return cells, cell_by_id, grid_lines


def find_cell_id(cells, x: float, y: float):
    for cell in cells:
        if cell["x0"] <= x < cell["x1"] and cell["y0"] <= y < cell["y1"]:
            return cell["id"]
    return None


def render_grid_frame(
    frame_height: int,
    frame_width: int,
    cells,
    cell_by_id,
    grid_lines,
    detections: np.ndarray,
):
    grid_base = np.full((frame_height, frame_width, 3), GRID_BG_COLOR, dtype=np.uint8)
    overlay = grid_base.copy()
    hit_map = {}

    for det in detections:
        x0, y0, x1, y1 = det[:4]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        cell_id = find_cell_id(cells, cx, cy)
        if cell_id is None:
            continue
        cls_idx = int(det[-2])
        unit_name = idx2unit.get(cls_idx, str(cls_idx))
        if unit_name in {"bar", "bar-level"} or unit_name.endswith("-bar"):
            continue
        hit_map.setdefault(cell_id, set()).add(unit_name)
        cell = cell_by_id.get(cell_id)
        if cell is not None:
            cv2.rectangle(
                overlay,
                (int(cell["x0"]), int(cell["y0"])),
                (int(cell["x1"]), int(cell["y1"])),
                (0, 180, 255),
                thickness=-1,
            )

    blended = cv2.addWeighted(overlay, GRID_FILL_ALPHA, grid_base, 1 - GRID_FILL_ALPHA, 0)
    for (start, end) in grid_lines:
        cv2.line(
            blended,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            GRID_LINE_COLOR,
            GRID_LINE_WIDTH,
        )

    panel = np.full((frame_height, TEXT_PANEL_WIDTH, 3), GRID_BG_COLOR, dtype=np.uint8)
    lines = []
    for cell_id in sorted(hit_map.keys()):
        units = ", ".join(sorted(hit_map[cell_id]))
        lines.append(f"格子 {cell_id}: {units}")
    y = TEXT_MARGIN + TEXT_LINE_HEIGHT
    for text in lines:
        if y >= frame_height - TEXT_MARGIN:
            break
        cv2.putText(
            panel,
            text,
            (TEXT_MARGIN, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += TEXT_LINE_HEIGHT

    return np.concatenate([blended, panel], axis=1)


def process_video(
    path_video: Path,
    combo: ComboDetector,
    battle_dir: Path,
    card_dir: Path,
    grid_dir: Path,
    combined_dir: Path,
):
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

    battle_path = battle_dir / f"{path_video.stem}.mp4"
    card_path = card_dir / f"{path_video.stem}.mp4"
    grid_path = grid_dir / f"{path_video.stem}.mp4"
    combined_path = combined_dir / f"{path_video.stem}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    battle_writer = cv2.VideoWriter(str(battle_path), fourcc, fps, (width, top_h))
    card_writer = cv2.VideoWriter(str(card_path), fourcc, fps, (width, bottom_h))
    grid_writer = cv2.VideoWriter(str(grid_path), fourcc, fps, (width + TEXT_PANEL_WIDTH, top_h))
    combined_writer = cv2.VideoWriter(
        str(combined_path),
        fourcc,
        fps,
        (width + width + TEXT_PANEL_WIDTH, top_h),
    )

    if combo.tracker is not None:
        combo.tracker.reset()

    cells, cell_by_id, grid_lines = build_grid_layout()

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
        grid_frame = render_grid_frame(top_h, width, cells, cell_by_id, grid_lines, detections)
        combined_frame = np.concatenate([battle_frame, grid_frame], axis=1)

        battle_writer.write(battle_frame)
        card_writer.write(bottom_part)
        grid_writer.write(grid_frame)
        combined_writer.write(combined_frame)

        if frame_idx % 100 == 0:
            print(f"{path_video.name} 已处理 {frame_idx} 帧")

    cap.release()
    battle_writer.release()
    card_writer.release()
    grid_writer.release()
    combined_writer.release()
    print(
        "完成: "
        f"{path_video.name} -> battle: {battle_path}, card: {card_path}, "
        f"grid: {grid_path}, combined: {combined_path}"
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    video_dir = repo_root / "video"
    output_dir = repo_root / "output"
    battle_dir = output_dir / "battle"
    card_dir = output_dir / "card"
    grid_dir = output_dir / "grid"
    combined_dir = output_dir / "combined"
    battle_dir.mkdir(parents=True, exist_ok=True)
    card_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    combo = ComboDetector(path_detectors, show_conf=True, conf=0.7, iou_thre=0.6, tracker="bytetrack")

    videos = iter_videos(video_dir)
    if not videos:
        print(f"未找到视频文件: {video_dir}")

    for video in videos:
        process_video(video, combo, battle_dir, card_dir, grid_dir, combined_dir)
