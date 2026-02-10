import sys

from ppadb.client import Client as AdbClient
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from auto_record import AutoRecorder
from tap_dialog import DEVICE, get_wm_size, TapDialog
from video_window import start_scrcpy, MarkerOverlay

# 超参配置
TEMPLATE_CHECK_INTERVAL_MS = 1000  # trophy 分类识别间隔（毫秒）


def main():
    client = AdbClient(host="127.0.0.1", port=5037)
    device = client.device(DEVICE)
    if device is None:
        raise RuntimeError("未找到设备 emulator-5556")

    emu_w, emu_h = get_wm_size(device)
    start_scrcpy(DEVICE)

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    overlay = MarkerOverlay(emu_w, emu_h)
    overlay.show()

    dialog = TapDialog(device, emu_w, emu_h, overlay)
    dialog.show()

    auto_recorder = AutoRecorder(device, dialog, overlay)
    auto_recorder.start(interval_ms=TEMPLATE_CHECK_INTERVAL_MS)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
