import sys

from ppadb.client import Client as AdbClient
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from auto_record import AutoRecorder
from tap_dialog import DEVICE, get_wm_size, TapDialog
from video_window import start_scrcpy, MarkerOverlay


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

    auto_recorder = AutoRecorder(device, dialog)
    auto_recorder.start(interval_ms=500)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
