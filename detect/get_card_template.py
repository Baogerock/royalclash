from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CARD_SAMPLE_THRESHOLD = 0.95
CARD_SAMPLE_FPS = 5

CARD_REGIONS = [
    ("1", ((157, 34), (292, 197))),
    ("2", ((292, 34), (427, 197))),
    ("3", ((428, 34), (562, 197))),
    ("4", ((562, 34), (697, 197))),
    ("next", ((34, 164), (101, 246))),
    ("water", ((196, 193), (257, 238))),
    ("trophy", ((11, 67), (60, 118))),
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


def build_sample_state(pic_dir: Path):
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
    return sample_state


def process_video(path_video: Path, sample_state):
    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {path_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_stride = max(int(round(fps / CARD_SAMPLE_FPS)), 1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % sample_stride == 0:
            for name, crop in crop_regions(frame, CARD_REGIONS):
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
    print(f"完成卡牌截图: {path_video.name}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    card_dir = repo_root / "output" / "card"
    pic_dir = card_dir / "pic"
    pic_dir.mkdir(parents=True, exist_ok=True)

    if not card_dir.exists():
        raise FileNotFoundError(f"卡牌视频目录不存在: {card_dir}")

    videos = iter_videos(card_dir)
    if not videos:
        print(f"未找到卡牌视频文件: {card_dir}")

    sample_state = build_sample_state(pic_dir)

    for video in videos:
        process_video(video, sample_state)
