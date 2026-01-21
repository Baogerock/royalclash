import os

import cv2
import numpy as np
from PIL import ImageGrab
from PySide6.QtCore import QTimer

from video_window import SCRCPY_TITLE, get_window_rect


class AutoRecorder:
    def __init__(self, device, dialog, overlay, template_path="trophy.png"):
        self.device = device
        self.dialog = dialog
        self.overlay = overlay
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_rules)
        self.template_path = template_path
        self.template = self._load_template(template_path)

    def start(self, interval_ms=1000):
        self.timer.start(interval_ms)

    def check_rules(self):
        if self.template is None:
            return

        rect = get_window_rect(SCRCPY_TITLE)
        if not rect:
            return

        left, top, right, bottom = rect
        frame = ImageGrab.grab(bbox=(left, top, right, bottom))
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        roi = frame_bgr[:300, :300]
        found = self._match_template(roi, self.template)
        if found:
            x, y, w, h, _ = found
            self.overlay.set_match_rect(x, y, w, h)
            if not self.dialog.recording:
                print("检测到模板，开始录制。")
                self.dialog.start_recording()
        else:
            self.overlay.set_match_rect(None)
            if self.dialog.recording:
                print("未检测到模板，停止录制。")
                self.dialog.stop_recording()

    def _load_template(self, path):
        if not os.path.exists(path):
            return None
        template = cv2.imread(path, cv2.IMREAD_COLOR)
        return template

    def _match_template(self, frame, template, threshold=0.8):
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
