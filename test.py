import sys
import re
import time
import subprocess
from ppadb.client import Client as AdbClient
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QPen

import win32gui

DEVICE = "emulator-5556"
SCRCPY = r"C:\0ShitMountain\royalclash\scrcpy-win64-v3.3.4\scrcpy.exe"
SCRCPY_TITLE = "LD Stream"   # 要和启动参数 --window-title 一致


def start_scrcpy():
    subprocess.Popen([
        SCRCPY, "-s", DEVICE,
        "--no-control",
        "--max-fps", "60",
        "--video-bit-rate", "8M",   # ✅ 新参数
        "--window-title", SCRCPY_TITLE
    ])



def get_wm_size(device):
    out = device.shell("wm size")
    m = re.search(r"(\d+)x(\d+)", out)
    if not m:
        raise RuntimeError(f"wm size parse failed: {out}")
    return int(m.group(1)), int(m.group(2))


def get_window_rect(title: str):
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        return None
    # 返回客户端区域 (left, top, right, bottom)
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    right, bottom = win32gui.ClientToScreen(
        hwnd, win32gui.GetClientRect(hwnd)[2:4]
    )
    return left, top, right, bottom


class Overlay(QMainWindow):
    def __init__(self, device, emu_w, emu_h):
        super().__init__()
        self.device = device
        self.emu_w = emu_w
        self.emu_h = emu_h
        self.rect_scrcpy = None

        # 透明、无边框、置顶、鼠标可点
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # 定时跟随 scrcpy 窗口
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

    def paintEvent(self, e):
        # 画个很淡的边框，方便确认覆盖层对齐（你不想要可删）
        if not self.rect_scrcpy:
            return
        p = QPainter(self)
        pen = QPen()
        pen.setWidth(2)
        p.setPen(pen)
        p.drawRect(1, 1, self.width() - 2, self.height() - 2)

    def _map_to_emulator(self, pos: QPointF):
        # overlay 窗口就是 scrcpy 客户区的近似映射
        # 这里做一个 KeepAspectRatio 的黑边处理，避免 scrcpy 拉伸时点偏
        scale = self.devicePixelRatioF()
        x, y = pos.x() * scale, pos.y() * scale
        w, h = self.width() * scale, self.height() * scale

        scale = min(w / self.emu_w, h / self.emu_h)
        disp_w = self.emu_w * scale
        disp_h = self.emu_h * scale
        off_x = (w - disp_w) / 2.0
        off_y = (h - disp_h) / 2.0

        x_in = x - off_x
        y_in = y - off_y
        if x_in < 0 or y_in < 0 or x_in > disp_w or y_in > disp_h:
            return None

        real_x = int(x_in / scale)
        real_y = int(y_in / scale)
        real_x = max(0, min(self.emu_w - 1, real_x))
        real_y = max(0, min(self.emu_h - 1, real_y))
        return real_x, real_y

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        mapped = self._map_to_emulator(event.position())
        if not mapped:
            return
        x, y = mapped
        self.device.shell(f"input tap {x} {y}")


def main():
    client = AdbClient(host="127.0.0.1", port=5037)
    device = client.device(DEVICE)
    if device is None:
        raise RuntimeError("未找到设备 emulator-5556")

    start_scrcpy()
    time.sleep(0.5)  # 给 scrcpy 起窗口一点时间（不想 sleep 可做轮询）

    emu_w, emu_h = get_wm_size(device)
    print("device:", emu_w, emu_h)

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    overlay = Overlay(device, emu_w, emu_h)
    overlay.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
