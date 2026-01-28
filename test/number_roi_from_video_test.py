from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SAMPLE_FPS = 5
DUPLICATE_THRESHOLD = 0.95
VIS_SAMPLE_INTERVAL = 100

NUMBER_REGIONS = [
    ("top_left", ((140, 162), (211, 192))),
    ("top_right", ((520, 162), (589, 192))),
    ("bottom_left", ((142, 790), (204, 820))),
    ("bottom_right", ((521, 790), (582, 820))),
]


def iter_videos(video_dir: Path):
    return sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


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


def segment_digits(roi: np.ndarray):
    if roi is None or roi.size == 0:
        return []

    scale = 4
    resized = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(enhanced, 50, 150)
    combined = cv2.bitwise_or(bright, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    height, width = cleaned.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 12:
            continue
        if w > width or h > height:
            continue
        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])
    digits = []
    for x, y, w, h in boxes:
        pad = 2
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(width, x + w + pad)
        bottom = min(height, y + h + pad)
        digit = resized[top:bottom, left:right]
        if digit.size == 0:
            continue
        digits.append(digit)
    return digits


def is_duplicate(
    sample: np.ndarray,
    existing_samples: list[np.ndarray],
    threshold=DUPLICATE_THRESHOLD,
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


def build_output_state(output_dir: Path):
    state = {}
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, _ in NUMBER_REGIONS:
        region_dir = output_dir / name
        region_dir.mkdir(parents=True, exist_ok=True)
        existing = [p for p in region_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg"}]
        indices = [int(p.stem) for p in existing if p.stem.isdigit()]
        next_index = max(indices, default=0) + 1
        samples = []
        for sample_path in existing:
            sample = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
            if sample is not None:
                samples.append(sample)
        state[name] = {"dir": region_dir, "next_index": next_index, "samples": samples}
    state["debug_dir"] = debug_dir
    return state


def process_video(path_video: Path, output_state):
    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {path_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_stride = max(int(round(fps / SAMPLE_FPS)), 1)

    frame_idx = 0
    debug_dir = output_state["debug_dir"]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % sample_stride == 0:
            for name, roi in crop_regions(frame, NUMBER_REGIONS):
                digits = segment_digits(roi)
                for digit in digits:
                    state = output_state[name]
                    if is_duplicate(digit, state["samples"]):
                        continue
                    sample_path = state["dir"] / f"{state['next_index']:06d}.png"
                    cv2.imwrite(str(sample_path), digit)
                    state["samples"].append(digit)
                    state["next_index"] += 1

        if frame_idx % VIS_SAMPLE_INTERVAL == 0:
            debug_frame = frame.copy()
            for name, ((x1, y1), (x2, y2)) in NUMBER_REGIONS:
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_frame,
                    name,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            debug_path = debug_dir / f"{path_video.stem}_{frame_idx:06d}.png"
            cv2.imwrite(str(debug_path), debug_frame)
            print(f"{path_video.name} 已处理 {frame_idx} 帧")

    cap.release()
    print(f"完成数字截图: {path_video.name}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    video_dir = repo_root / "video"
    output_dir = repo_root / "output" / "number"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    videos = iter_videos(video_dir)
    if not videos:
        print(f"未找到视频文件: {video_dir}")

    output_state = build_output_state(output_dir)

    for video in videos:
        process_video(video, output_state)
