import os
import time

import cv2
import numpy as np
from PySide6.QtCore import QTimer

# 超参配置
TEMPLATE_PATH = "trophy.png"  # 模板图片路径
MATCH_THRESHOLD = 0.8  # 模板匹配阈值
ROI_LEFT = 0  # 模板匹配区域左上角 X
ROI_TOP = 0  # 模板匹配区域左上角 Y
ROI_RIGHT = 300  # 模板匹配区域右下角 X
ROI_BOTTOM = 300  # 模板匹配区域右下角 Y
STOP_GRACE_SECONDS = 3  # 模板消失后多少秒才停止录制


class AutoRecorder:
    def __init__(self, device, dialog, overlay, template_path=TEMPLATE_PATH):
        self.device = device
        self.dialog = dialog
        self.overlay = overlay
        self.last_detect_time = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_rules)
        self.template_path = template_path
        self.template = self._load_template(template_path)

    def start(self, interval_ms=1000):
        self.timer.start(interval_ms)

    def check_rules(self):
        if self.template is None:
            return

        screenshot = self.device.screencap()
        frame_bgr = cv2.imdecode(
            np.frombuffer(screenshot, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame_bgr is None:
            return

        roi = frame_bgr[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
        found = self._match_template(roi, self.template)
        if found:
            x, y, w, h, _ = found
            self.overlay.set_match_rect_emulator(x, y, w, h)
            self.last_detect_time = time.time()
            if not self.dialog.recording:
                print("检测到模板，开始录制。")
                self.dialog.start_recording()
        else:
            self.overlay.set_match_rect_emulator(None)
            if self.dialog.recording and self._should_stop():
                print("未检测到模板，停止录制。")
                self.dialog.stop_recording()
                self.last_detect_time = None

    def _should_stop(self, grace_seconds=STOP_GRACE_SECONDS):
        if self.last_detect_time is None:
            return False
        return (time.time() - self.last_detect_time) >= grace_seconds

    def _load_template(self, path):
        if not os.path.exists(path):
            return None
        template = cv2.imread(path, cv2.IMREAD_COLOR)
        return template

    def _match_template(self, frame, template, threshold=MATCH_THRESHOLD):
        if template is None or frame is None:
            return None
        h, w = template.shape[:2]
        if frame.shape[0] < h or frame.shape[1] < w:
            return None
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            x, y = max_loc
            return x, y, w, h, max_val
        return None
