import os
import re
import time
from pathlib import Path

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
# 超参配置
VIDEO_BITRATE = "8000000"  # 录制码率
REMOTE_SAVE_DIR = "/sdcard"  # 设备端保存目录
BASE_DIR = Path(__file__).resolve().parents[1]
LOCAL_SAVE_DIR = BASE_DIR / "video"  # 本地保存目录
WAIT_REMOTE_TIMEOUT_S = 10  # 等待远端文件稳定超时（秒）


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
        self.recording = False
        self.record_pid = None
        self.record_remote_path = None
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

        self.record_btn = QPushButton("开始录制")
        self.record_btn.clicked.connect(self.on_record_toggle)
        btn_row.addWidget(self.record_btn)

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

    def on_record_toggle(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.recording:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        remote_path = f"{REMOTE_SAVE_DIR}/record_{timestamp}.mp4"
        cmd = (
            "sh -c "
            f"'nohup screenrecord --bit-rate {VIDEO_BITRATE} {remote_path} "
            "> /dev/null 2>&1 & echo $!'"
        )
        pid_out = self.device.shell(cmd).strip()
        try:
            self.record_pid = int(pid_out)
        except ValueError:
            QMessageBox.warning(self, "录制失败", f"无法获取录制进程ID：{pid_out}")
            return
        if not self._is_pid_alive(self.record_pid):
            print("录制进程未启动成功，请检查设备是否支持 screenrecord。")
            self.record_pid = None
            return
        self.record_remote_path = remote_path
        self.recording = True
        self.record_btn.setText("停止录制")
        print(f"开始录制，保存到设备路径：{remote_path}")

    def stop_recording(self):
        if not self.recording:
            return
        if self.record_pid is None or self.record_remote_path is None:
            QMessageBox.warning(self, "录制失败", "录制状态异常。")
            return
        self.device.shell(f"kill -2 {self.record_pid}")
        self._wait_for_remote_file()
        LOCAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        local_path = str(LOCAL_SAVE_DIR / f"{self._next_sequence():03d}.mp4")
        print(f"保存中：{local_path}")
        self.device.pull(self.record_remote_path, local_path)
        self.recording = False
        self.record_pid = None
        self.record_remote_path = None
        self.record_btn.setText("开始录制")
        print(f"录制已保存：{local_path}")

    def _wait_for_remote_file(self, timeout_s=WAIT_REMOTE_TIMEOUT_S):
        start = time.time()
        last_size = -1
        stable_checks = 0
        while time.time() - start < timeout_s:
            size = self._get_remote_size(self.record_remote_path)
            if size is None:
                time.sleep(0.5)
                continue
            if size == last_size and size > 0:
                stable_checks += 1
                if stable_checks >= 2:
                    return
            else:
                stable_checks = 0
            last_size = size
            print(f"保存中：远端大小 {size} bytes")
            time.sleep(0.5)
        print("保存中：等待远端文件完成超时，继续拉取。")

    def _get_remote_size(self, path):
        out = self.device.shell(f"ls -l {path}").strip()
        parts = out.split()
        if len(parts) < 5:
            return None
        try:
            return int(parts[4])
        except ValueError:
            return None

    def _is_pid_alive(self, pid):
        out = self.device.shell("ps -A").strip()
        return str(pid) in out

    def _next_sequence(self):
        existing = []
        for path in LOCAL_SAVE_DIR.iterdir():
            if not path.is_file():
                continue
            match = re.match(r"^(\\d{3})\\.mp4$", path.name)
            if match:
                existing.append(int(match.group(1)))
        return max(existing, default=0) + 1
