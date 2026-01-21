import time
import subprocess

from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QPen

import win32gui


SCRCPY = r"C:\0ShitMountain\royalclash\scrcpy-win64-v3.3.4\scrcpy.exe"
SCRCPY_TITLE = "LD Stream"   # 要和启动参数 --window-title 一致


def start_scrcpy(device_id: str):
    subprocess.Popen([
        SCRCPY, "-s", device_id,
        "--no-control",
        "--max-fps", "60",
        "--video-bit-rate", "8M",
        "--window-title", SCRCPY_TITLE,
    ])
    time.sleep(0.5)


def get_window_rect(title: str):
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        return None
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    right, bottom = win32gui.ClientToScreen(
        hwnd, win32gui.GetClientRect(hwnd)[2:4]
    )
    return left, top, right, bottom


class MarkerOverlay(QMainWindow):
    def __init__(self, emu_w, emu_h):
        super().__init__()
        self.emu_w = emu_w
        self.emu_h = emu_h
        self.rect_scrcpy = None
        self.marker_pos = None
        self.match_rect = None
        self.match_rect_emulator = None

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sync_to_scrcpy)
        self.timer.start(100)

    def sync_to_scrcpy(self):
        r = get_window_rect(SCRCPY_TITLE)
        if not r:
            return
        if r != self.rect_scrcpy:
            self.rect_scrcpy = r
            l, t, rr, bb = r
            scale = self.devicePixelRatioF()
            self.setGeometry(
                int(l / scale),
                int(t / scale),
                int((rr - l) / scale),
                int((bb - t) / scale),
            )
            self.update()

    def set_marker(self, x, y):
        self.marker_pos = QPointF(x, y)
        self.update()

    def set_match_rect(self, x, y=None, w=None, h=None):
        if x is None:
            self.match_rect = None
        else:
            self.match_rect = (x, y, w, h)
        self.update()

    def set_match_rect_emulator(self, x, y=None, w=None, h=None):
        if x is None:
            self.match_rect_emulator = None
        else:
            self.match_rect_emulator = (x, y, w, h)
        self.update()

    def _map_from_emulator(self, pos: QPointF):
        scale = self.devicePixelRatioF()
        w = self.width() * scale
        h = self.height() * scale

        view_scale = min(w / self.emu_w, h / self.emu_h)
        disp_w = self.emu_w * view_scale
        disp_h = self.emu_h * view_scale
        off_x = (w - disp_w) / 2.0
        off_y = (h - disp_h) / 2.0

        x = off_x + pos.x() * view_scale
        y = off_y + pos.y() * view_scale
        return QPointF(x / scale, y / scale)

    def paintEvent(self, e):
        if not self.rect_scrcpy:
            return
        p = QPainter(self)
        if self.marker_pos:
            pen = QPen()
            pen.setWidth(3)
            p.setPen(pen)
            center = self._map_from_emulator(self.marker_pos)
            size = 12
            p.drawLine(
                center.x() - size,
                center.y(),
                center.x() + size,
                center.y(),
            )
            p.drawLine(
                center.x(),
                center.y() - size,
                center.x(),
                center.y() + size,
            )

        if self.match_rect:
            x, y, w, h = self.match_rect
            scale = self.devicePixelRatioF()
            pen = QPen(Qt.red)
            pen.setWidth(2)
            p.setPen(pen)
            p.drawRect(
                int(x / scale),
                int(y / scale),
                int(w / scale),
                int(h / scale),
            )

        if self.match_rect_emulator:
            x, y, w, h = self.match_rect_emulator
            top_left = self._map_from_emulator(QPointF(x, y))
            bottom_right = self._map_from_emulator(QPointF(x + w, y + h))
            pen = QPen(Qt.red)
            pen.setWidth(2)
            p.setPen(pen)
            p.drawRect(
                int(top_left.x()),
                int(top_left.y()),
                int(bottom_right.x() - top_left.x()),
                int(bottom_right.y() - top_left.y()),
            )
