import re

from PySide6.QtWidgets import (
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
    def __init__(self, device, emu_w, emu_h, overlay):
        super().__init__()
        self.device = device
        self.emu_w = emu_w
        self.emu_h = emu_h
        self.overlay = overlay
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

        btn_row = QHBoxLayout()
        self.locate_btn = QPushButton("定位")
        self.locate_btn.clicked.connect(self.on_locate)
        btn_row.addWidget(self.locate_btn)

        self.confirm_btn = QPushButton("确认点击")
        self.confirm_btn.clicked.connect(self.on_confirm)
        btn_row.addWidget(self.confirm_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def _read_coords(self):
        try:
            x = int(self.input_x.text().strip())
            y = int(self.input_y.text().strip())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入整数坐标。")
            return None

        if not (0 <= x < self.emu_w) or not (0 <= y < self.emu_h):
            QMessageBox.warning(
                self,
                "输入错误",
                f"坐标超出范围：x 0-{self.emu_w - 1}，y 0-{self.emu_h - 1}",
            )
            return None
        return x, y

    def on_locate(self):
        coords = self._read_coords()
        if not coords:
            return
        x, y = coords
        self.overlay.set_marker(x, y)

    def on_confirm(self):
        coords = self._read_coords()
        if not coords:
            return
        x, y = coords
        self.device.shell(f"input tap {x} {y}")
        QMessageBox.information(self, "完成", f"已点击坐标 ({x}, {y})。")
