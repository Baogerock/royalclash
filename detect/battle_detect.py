from pathlib import Path

import cv2
import numpy as np

from katacr.yolov8.combo_detect import ComboDetector, path_detectors

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CARD_SAMPLE_THRESHOLD = 0.95
CARD_SAMPLE_FPS = 1

CARD_REGIONS = [
    ("1", ((157, 34), (292, 197))),
    ("2", ((292, 34), (427, 197))),
    ("3", ((428, 34), (562, 197))),
    ("4", ((562, 34), (697, 197))),
    ("next", ((34, 164), (101, 246))),
    ("water", ((196, 193), (257, 238))),
]


def iter_videos(video_dir: Path):
    return sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def iter_samples(pic_dir: Path):
    return sorted([p for p in pic_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def next_sample_index(pic_dir: Path):
    existing = []
    for path in iter_samples(pic_dir):
        if path.stem.isdigit():
            existing.append(int(path.stem))
    return max(existing, default=0) + 1


def is_duplicate(
    sample: np.ndarray,
    existing_samples: list[np.ndarray],
    threshold=CARD_SAMPLE_THRESHOLD,
):
    if sample is None or sample.size == 0:
        return True
    for existing in existing_samples:
        if existing is None or existing.size == 0:
            continue
        if existing.shape[:2] != sample.shape[:2]:
            continue
        result = cv2.matchTemplate(existing, sample, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val >= threshold:
            return True
    return False


def crop_regions(frame: np.ndarray, regions):
    crops = []
    height, width = frame.shape[:2]
    for name, ((x1, y1), (x2, y2)) in regions:
        left = max(0, min(width, x1))
        right = max(0, min(width, x2))
        top = max(0, min(height, y1))
        bottom = max(0, min(height, y2))
        if right <= left or bottom <= top:
            continue
        crops.append((name, frame[top:bottom, left:right]))
    return crops


def process_video(
    path_video: Path,
    combo: ComboDetector,
    battle_dir: Path,
    card_dir: Path,
    pic_dir: Path,
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    battle_writer = cv2.VideoWriter(str(battle_path), fourcc, fps, (width, top_h))
    card_writer = cv2.VideoWriter(str(card_path), fourcc, fps, (width, bottom_h))

    if combo.tracker is not None:
        combo.tracker.reset()

    frame_idx = 0
    sample_stride = max(int(round(fps / CARD_SAMPLE_FPS)), 1)
    sample_state = {}
    for name, _ in CARD_REGIONS:
        region_dir = pic_dir / name
        region_dir.mkdir(parents=True, exist_ok=True)
        existing_samples = []
        for sample_path in iter_samples(region_dir):
            sample = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
            if sample is not None:
                existing_samples.append(sample)
        sample_state[name] = {
            "dir": region_dir,
            "next_index": next_sample_index(region_dir),
            "samples": existing_samples,
        }
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        top_part = frame[:top_h, :]
        bottom_part = frame[top_h:, :]

        pred = combo.infer(top_part)
        battle_frame = pred.show_box(verbose=False, show_conf=True)

        battle_writer.write(battle_frame)
        card_writer.write(bottom_part)

        if frame_idx % sample_stride == 0:
            for name, crop in crop_regions(bottom_part, CARD_REGIONS):
                state = sample_state[name]
                if is_duplicate(crop, state["samples"]):
                    continue
                sample_path = state["dir"] / f"{state['next_index']:02d}.png"
                cv2.imwrite(str(sample_path), crop)
                state["samples"].append(crop)
                state["next_index"] += 1

        if frame_idx % 100 == 0:
            print(f"{path_video.name} 已处理 {frame_idx} 帧")

    cap.release()
    battle_writer.release()
    card_writer.release()
    print(f"完成: {path_video.name} -> battle: {battle_path}, card: {card_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    video_dir = repo_root / "video"
    output_dir = repo_root / "output"
    battle_dir = output_dir / "battle"
    card_dir = output_dir / "card"
    pic_dir = card_dir / "pic"
    battle_dir.mkdir(parents=True, exist_ok=True)
    card_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    combo = ComboDetector(path_detectors, show_conf=True, conf=0.7, iou_thre=0.6, tracker="bytetrack")

    videos = iter_videos(video_dir)
    if not videos:
        print(f"未找到视频文件: {video_dir}")

    for video in videos:
        process_video(video, combo, battle_dir, card_dir, pic_dir)
