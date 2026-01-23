from pathlib import Path

import cv2

from katacr.yolov8.combo_detect import ComboDetector, path_detectors

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_videos(video_dir: Path):
    return sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def process_video(
    path_video: Path,
    combo: ComboDetector,
    battle_dir: Path,
    card_dir: Path,
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
    battle_dir.mkdir(parents=True, exist_ok=True)
    card_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    combo = ComboDetector(path_detectors, show_conf=True, conf=0.7, iou_thre=0.6, tracker="bytetrack")

    videos = iter_videos(video_dir)
    if not videos:
        print(f"未找到视频文件: {video_dir}")

    for video in videos:
        process_video(video, combo, battle_dir, card_dir)
