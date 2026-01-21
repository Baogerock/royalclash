import io
import time

from PIL import Image
import pytesseract
from PySide6.QtCore import QTimer


class AutoRecorder:
    def __init__(self, device, dialog):
        self.device = device
        self.dialog = dialog
        self.cooldown_start_until = 0.0
        self.cooldown_stop_until = 0.0
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_rules)

    def start(self, interval_ms=1000):
        self.timer.start(interval_ms)

    def check_rules(self):
        now = time.time()
        screenshot = self.device.screencap()
        image = Image.open(io.BytesIO(screenshot))

        if not self.dialog.recording and now >= self.cooldown_start_until:
            if self._has_text(image, (300, 530, 440, 610), "对战"):
                self.dialog.start_recording()
                self.cooldown_start_until = now + 10
                return

        if self.dialog.recording and now >= self.cooldown_stop_until:
            if self._has_text(image, (300, 450, 420, 510), "对战"):
                self.dialog.stop_recording()
                self.cooldown_stop_until = now + 10

    def _has_text(self, image, box, target):
        crop = image.crop(box)
        text = pytesseract.image_to_string(crop, lang="chi_sim")
        return target in text.replace(" ", "")
