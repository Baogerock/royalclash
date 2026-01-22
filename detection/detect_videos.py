import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO


# 超参配置
VIDEO_DIR = Path(__file__).resolve().parents[1] / "video"  # 输入视频目录
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "detect_result"  # 输出目录
SHOW_PREVIEW = True  # 是否显示可视化窗口
OUTPUT_FPS = 20.0  # 输出视频帧率
ROI_X1 = 0  # 对战区域左上角 X
ROI_Y1 = 0  # 对战区域左上角 Y
ROI_X2 = 720  # 对战区域右下角 X
ROI_Y2 = 1280  # 对战区域右下角 Y

PATH_DETECTORS = [
    Path("C:/0ShitMountain/KataCR/runs/detector1_v0.7.13.pt"),
    Path("C:/0ShitMountain/KataCR/runs/detector2_v0.7.13.pt"),
]

UNIT_LIST = [
    "king-tower",
    "queen-tower",
    "cannoneer-tower",
    "dagger-duchess-tower",
    "dagger-duchess-tower-bar",
    "tower-bar",
    "king-tower-bar",
    "bar",
    "bar-level",
    "clock",
    "emote",
    "text",
    "elixir",
    "selected",
    "skeleton-king-bar",
    "skeleton",
    "skeleton-evolution",
    "electro-spirit",
    "fire-spirit",
    "ice-spirit",
    "heal-spirit",
    "goblin",
    "spear-goblin",
    "bomber",
    "bat",
    "bat-evolution",
    "zap",
    "giant-snowball",
    "ice-golem",
    "barbarian-barrel",
    "barbarian",
    "barbarian-evolution",
    "wall-breaker",
    "rage",
    "the-log",
    "archer",
    "arrows",
    "knight",
    "knight-evolution",
    "minion",
    "cannon",
    "skeleton-barrel",
    "firecracker",
    "firecracker-evolution",
    "royal-delivery",
    "royal-recruit",
    "royal-recruit-evolution",
    "tombstone",
    "mega-minion",
    "dart-goblin",
    "earthquake",
    "elixir-golem-big",
    "elixir-golem-mid",
    "elixir-golem-small",
    "goblin-barrel",
    "guard",
    "clone",
    "tornado",
    "miner",
    "dirt",
    "princess",
    "ice-wizard",
    "royal-ghost",
    "bandit",
    "fisherman",
    "skeleton-dragon",
    "mortar",
    "mortar-evolution",
    "tesla",
    "fireball",
    "mini-pekka",
    "musketeer",
    "goblin-cage",
    "goblin-brawler",
    "valkyrie",
    "battle-ram",
    "battle-ram-evolution",
    "bomb-tower",
    "bomb",
    "flying-machine",
    "hog-rider",
    "battle-healer",
    "furnace",
    "zappy",
    "baby-dragon",
    "dark-prince",
    "freeze",
    "poison",
    "hunter",
    "goblin-drill",
    "electro-wizard",
    "inferno-dragon",
    "phoenix-big",
    "phoenix-egg",
    "phoenix-small",
    "magic-archer",
    "lumberjack",
    "night-witch",
    "mother-witch",
    "hog",
    "golden-knight",
    "skeleton-king",
    "mighty-miner",
    "rascal-boy",
    "rascal-girl",
    "giant",
    "goblin-hut",
    "inferno-tower",
    "wizard",
    "royal-hog",
    "witch",
    "balloon",
    "prince",
    "electro-dragon",
    "bowler",
    "executioner",
    "axe",
    "cannon-cart",
    "ram-rider",
    "graveyard",
    "archer-queen",
    "monk",
    "royal-giant",
    "royal-giant-evolution",
    "elite-barbarian",
    "rocket",
    "barbarian-hut",
    "elixir-collector",
    "giant-skeleton",
    "lightning",
    "goblin-giant",
    "x-bow",
    "sparky",
    "pekka",
    "electro-giant",
    "mega-knight",
    "lava-hound",
    "lava-pup",
    "golem",
    "golemite",
    "little-prince",
    "royal-guardian",
    "archer-evolution",
    "ice-spirit-evolution",
    "valkyrie-evolution",
    "bomber-evolution",
    "wall-breaker-evolution",
    "evolution-symbol",
    "mirror",
    "tesla-evolution",
    "goblin-ball",
    "skeleton-king-skill",
    "tesla-evolution-shock",
    "ice-spirit-evolution-symbol",
    "zap-evolution",
]

IDX2UNIT = dict(enumerate(UNIT_LIST))
UNIT2IDX = {name: idx for idx, name in enumerate(UNIT_LIST)}


class TroopResults:
    def __init__(self, orig_img, boxes, names):
        self.orig_img = orig_img
        self.boxes = boxes
        self.names = names

    def get_data(self):
        if isinstance(self.boxes, torch.Tensor):
            return self.boxes.detach().cpu().numpy()
        return np.asarray(self.boxes)


class TroopDetector:
    def __init__(self, detectors=None, conf=0.7, iou_thre=0.6):
        self.detectors = detectors or PATH_DETECTORS
        if not _has_katacr():
            raise RuntimeError(
                "缺少 katacr 依赖，无法加载自定义权重。请先安装 katacr，"
                "或使用官方 YOLOv8 权重重新训练。"
            )
        self.models = [YOLO(str(p)) for p in self.detectors]
        self.conf = conf
        self.iou_thre = iou_thre

    def infer(self, frame):
        results = [m.predict(frame, verbose=False, conf=self.conf)[0] for m in self.models]
        preds = []
        for result in results:
            if result.boxes is None or result.boxes.xyxy is None:
                continue
            xyxy = result.boxes.xyxy
            confs = result.boxes.conf
            clss = result.boxes.cls
            for i in range(len(xyxy)):
                name = result.names[int(clss[i])]
                cls_idx = UNIT2IDX.get(name, -1)
                if cls_idx < 0:
                    continue
                pred = torch.stack(
                    [
                        xyxy[i, 0],
                        xyxy[i, 1],
                        xyxy[i, 2],
                        xyxy[i, 3],
                        confs[i],
                        torch.tensor(float(cls_idx), device=xyxy.device),
                        torch.tensor(0.0, device=xyxy.device),
                    ]
                )
                preds.append(pred)
        if not preds:
            preds = torch.zeros(0, 7)
        else:
            preds = torch.cat(preds, 0).reshape(-1, 7)
        keep = torchvision.ops.nms(preds[:, :4], preds[:, 4], iou_threshold=self.iou_thre)
        preds = preds[keep]
        return TroopResults(frame, preds, IDX2UNIT)


def draw_boxes(frame, results, show_conf=True):
    data = results.get_data()
    out = frame.copy()
    for box in data:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = float(box[4])
        cls = int(box[5])
        label = IDX2UNIT[cls]
        if show_conf:
            label = f"{label} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return out


def iter_videos(video_dir: Path):
    for path in sorted(video_dir.glob("*.mp4")):
        yield path


def process_video(video_path: Path, detector: TroopDetector):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频：{video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
    output_path = OUTPUT_DIR / f"{video_path.stem}_detect.mp4"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (ROI_X2 - ROI_X1, ROI_Y2 - ROI_Y1),
    )
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        results = detector.infer(roi)
        drawn = draw_boxes(roi, results)
        writer.write(drawn)
        if SHOW_PREVIEW:
            cv2.imshow("Detection Preview", drawn)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"{video_path.name}: 已处理 {frame_idx} 帧")
    cap.release()
    writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    print(f"完成输出：{output_path}")


def main():
    if not VIDEO_DIR.exists():
        print(f"视频目录不存在：{VIDEO_DIR}")
        return
    try:
        detector = TroopDetector()
    except RuntimeError as exc:
        print(f"检测模型加载失败：{exc}")
        return
    for video_path in iter_videos(VIDEO_DIR):
        process_video(video_path, detector)
        time.sleep(0.2)


if __name__ == "__main__":
    main()


def _has_katacr():
    return importlib.util.find_spec("katacr") is not None
