import os
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from ultralytics import YOLO

# 超参配置
BASE_DIR = Path(__file__).resolve().parents[1]
CLASSIFIER_MODEL_ENV = "CARD_CLASSIFIER_MODEL"
DEFAULT_MODEL_PATH = BASE_DIR / "train" / "train_card" / "best.pt"
TROPHY_CLASS_ID = "21"  # trophy 类别
TROPHY_REGION = ((11, 67), (60, 118))  # 基于 720x1280 的截图区域
BASE_FULL_WIDTH = 720
BASE_FULL_HEIGHT = 1280
STOP_GRACE_SECONDS = 3  # trophy 消失后多少秒才停止录制


class AutoRecorder:
    def __init__(self, device, dialog, overlay):
        self.device = device
        self.dialog = dialog
        self.overlay = overlay
        self.last_detect_time = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_rules)
        self.classifier = self._load_classifier()

    def start(self, interval_ms=1000):
        self.timer.start(interval_ms)

    def check_rules(self):
        if self.classifier is None:
            return

        screenshot = self.device.screencap()
        frame_bgr = cv2.imdecode(
            np.frombuffer(screenshot, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame_bgr is None:
            return

        trophy_region = self._build_trophy_region(frame_bgr.shape[1], frame_bgr.shape[0])
        crop = self._crop_region(frame_bgr, trophy_region)
        label = self._classify_card(crop, self.classifier)
        found = label == TROPHY_CLASS_ID

        if found:
            (x1, y1), (x2, y2) = trophy_region
            self.overlay.set_match_rect_emulator(x1, y1, x2 - x1, y2 - y1)
            self.last_detect_time = time.time()
            if not self.dialog.recording:
                print("检测到 trophy(21)，开始录制。")
                self.dialog.start_recording()
        else:
            self.overlay.set_match_rect_emulator(None)
            if self.dialog.recording and self._should_stop():
                print("未检测到 trophy(21)，停止录制。")
                self.dialog.stop_recording()
                self.last_detect_time = None

    def _should_stop(self, grace_seconds=STOP_GRACE_SECONDS):
        if self.last_detect_time is None:
            return False
        return (time.time() - self.last_detect_time) >= grace_seconds

    def _resolve_model_path(self, candidate: Path) -> Path | None:
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

    def _load_classifier(self):
        env_val = os.environ.get(CLASSIFIER_MODEL_ENV, "")
        env_path = Path(env_val).expanduser() if env_val else Path("__missing__")
        model_path = self._resolve_model_path(env_path)
        if model_path is None:
            model_path = self._resolve_model_path(DEFAULT_MODEL_PATH)
        if model_path is None:
            print(f"[WARN] 未找到分类模型，自动录制不可用: {env_path}")
            return None
        return YOLO(str(model_path), task="classify")

    def _classify_card(self, crop: np.ndarray, model: YOLO) -> str:
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

    def _scale_point(
        self,
        x: float,
        y: float,
        src_w: float,
        src_h: float,
        dst_w: float,
        dst_h: float,
    ) -> tuple[int, int]:
        return int(round(x * dst_w / src_w)), int(round(y * dst_h / src_h))

    def _build_trophy_region(self, frame_width: int, frame_height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        (x1, y1), (x2, y2) = TROPHY_REGION
        p1 = self._scale_point(x1, y1, BASE_FULL_WIDTH, BASE_FULL_HEIGHT, frame_width, frame_height)
        p2 = self._scale_point(x2, y2, BASE_FULL_WIDTH, BASE_FULL_HEIGHT, frame_width, frame_height)
        return p1, p2

    def _crop_region(self, frame: np.ndarray, region: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
        (x1, y1), (x2, y2) = region
        height, width = frame.shape[:2]
        left = max(0, min(width, x1))
        right = max(0, min(width, x2))
        top = max(0, min(height, y1))
        bottom = max(0, min(height, y2))
        if right <= left or bottom <= top:
            return np.empty((0, 0), dtype=frame.dtype)
        return frame[top:bottom, left:right]
