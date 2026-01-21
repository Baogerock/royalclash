import sys
import re

from ppadb.client import Client as AdbClient
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
)


DEVICE = "emulator-5556"


def get_wm_size(device):
    out = device.shell("wm size")
    m = re.search(r"(\d+)x(\d+)", out)
    if not m:
        raise RuntimeError(f"wm size parse failed: {out}")
    return int(m.group(1)), int(m.group(2))


class TapDialog(QDialog):
    def __init__(self, device, emu_w, emu_h):
        super().__init__()
        self.device = device
        self.emu_w = emu_w
        self.emu_h = emu_h
        self.setWindowTitle("模拟器点击")

        layout = QVBoxLayout()
        layout.addWidget(
            QLabel(f"请输入坐标范围：x 0-{emu_w - 1}，y 0-{emu_h - 1}")
        )

        row_x = QHBoxLayout()
        row_x.addWidget(QLabel("X:"))
        self.input_x = QLineEdit()
        row_x.addWidget(self.input_x)
        layout.addLayout(row_x)

        row_y = QHBoxLayout()
        row_y.addWidget(QLabel("Y:"))
        self.input_y = QLineEdit()
        row_y.addWidget(self.input_y)
        layout.addLayout(row_y)

        self.confirm_btn = QPushButton("确认点击")
        self.confirm_btn.clicked.connect(self.on_confirm)
        layout.addWidget(self.confirm_btn)

        self.setLayout(layout)

    def on_confirm(self):
        try:
            x = int(self.input_x.text().strip())
            y = int(self.input_y.text().strip())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入整数坐标。")
            return

        if not (0 <= x < self.emu_w) or not (0 <= y < self.emu_h):
            QMessageBox.warning(
                self,
                "输入错误",
                f"坐标超出范围：x 0-{self.emu_w - 1}，y 0-{self.emu_h - 1}",
            )
            return

        self.device.shell(f"input tap {x} {y}")
        QMessageBox.information(self, "完成", f"已点击坐标 ({x}, {y})。")


def main():
    client = AdbClient(host="127.0.0.1", port=5037)
    device = client.device(DEVICE)
    if device is None:
        raise RuntimeError("未找到设备 emulator-5556")

    emu_w, emu_h = get_wm_size(device)

    app = QApplication(sys.argv)
    dialog = TapDialog(device, emu_w, emu_h)
    dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
