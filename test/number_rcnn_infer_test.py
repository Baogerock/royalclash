from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F

from number_roi_from_video_test import NUMBER_REGIONS, iter_videos


def load_label_map(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        label_map = json.load(handle)
    return {int(v): k for k, v in label_map.items()}


def load_model(model_path: Path, num_classes: int, device: str):
    model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def draw_prediction(frame, roi_offset, prediction, label_map, score_threshold):
    if not prediction["boxes"].numel():
        return
    scores = prediction["scores"].detach().cpu().numpy()
    best_idx = scores.argmax()
    if scores[best_idx] < score_threshold:
        return
    boxes = prediction["boxes"].detach().cpu().numpy()
    label_id = int(prediction["labels"].detach().cpu().numpy()[best_idx])
    label = label_map.get(label_id, str(label_id))
    x1, y1, x2, y2 = boxes[best_idx]
    ox, oy = roi_offset
    pt1 = (int(x1 + ox), int(y1 + oy))
    pt2 = (int(x2 + ox), int(y2 + oy))
    cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"{label}:{scores[best_idx]:.2f}",
        (pt1[0], max(0, pt1[1] - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer numbers on ROIs using Faster R-CNN.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("number_rcnn.pt"),
        help="模型权重路径.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("../train/train_number/generated/label_map.json"),
        help="label_map.json 路径.",
    )
    parser.add_argument("--video", type=Path, default=Path("../video"), help="视频目录.")
    parser.add_argument("--output", type=Path, default=Path("output/number/infer"), help="输出目录.")
    parser.add_argument("--sample-fps", type=int, default=5, help="采样 FPS.")
    parser.add_argument("--save-interval", type=int, default=100, help="保存调试帧间隔.")
    parser.add_argument("--score", type=float, default=0.6, help="置信度阈值.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    if not args.labels.exists():
        raise FileNotFoundError(f"标签文件不存在: {args.labels}")
    if not args.video.exists():
        raise FileNotFoundError(f"视频目录不存在: {args.video}")

    label_map = load_label_map(args.labels)
    model = load_model(args.model, num_classes=len(label_map) + 1, device=args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    videos = iter_videos(args.video)
    if not videos:
        print(f"未找到视频文件: {args.video}")
        return

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[WARN] 无法打开视频: {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        stride = max(int(round(fps / args.sample_fps)), 1)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue

            debug_frame = frame.copy()
            for name, ((x1, y1), (x2, y2)) in NUMBER_REGIONS:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                tensor = F.to_tensor(image).to(args.device)
                with torch.no_grad():
                    prediction = model([tensor])[0]
                draw_prediction(debug_frame, (x1, y1), prediction, label_map, args.score)
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

            cv2.imshow("number_rcnn_infer", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

            if frame_idx % args.save_interval == 0:
                out_path = args.output / f"{video.stem}_{frame_idx:06d}.png"
                cv2.imwrite(str(out_path), debug_frame)
                print(f"{video.name} 已处理 {frame_idx} 帧")
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
