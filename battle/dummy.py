from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ppadb.client import Client as AdbClient

from detect.detect import CARD_REGIONS, classify_card, crop_region, load_classifier

# 参考 test/grid_test.py 的区域划分与网格参数
REGION2_ROWS = [
    (53, 137.0000, 666, 163.5385),
    (53, 163.5385, 666, 190.0769),
    (53, 190.0769, 666, 216.6154),
    (53, 216.6154, 666, 243.1538),
    (53, 243.1538, 666, 269.6923),
    (53, 269.6923, 666, 296.2308),
    (53, 296.2308, 666, 322.7692),
    (53, 322.7692, 666, 349.3077),
    (53, 349.3077, 666, 375.8462),
    (53, 375.8462, 666, 402.3846),
    (53, 402.3846, 666, 428.9231),
    (53, 428.9231, 666, 455.4615),
    (53, 455.4615, 666, 482.0000),
]

REGION5_ROWS = [
    (53, 586.0000, 666, 615),
    (53, 615, 666, 643),
    (53, 643, 666, 672),
    (53, 672, 666, 700),
    (53, 700, 666, 728),
    (53, 728, 666, 756),
    (53, 756, 666, 783),
    (53, 783, 666, 811),
    (53, 811, 666, 840),
    (53, 840, 666, 868),
    (53, 868, 666, 896),
    (53, 896, 666, 924),
    (53, 924, 666, 950.0000),
]

GRID_REGIONS = [
    (258, 112, 461, 137),
    *REGION2_ROWS,
    (85, 481, 633, 507),
    (120, 506, 221, 559),
    (499, 506, 597, 558),
    (85, 559, 632, 587),
    *REGION5_ROWS,
    (257, 950, 460, 980),
]

GRID_STEP = 34
GRID_MIN_CELL_RATIO = 0.5

CARD_COSTS = {
    "00": 7,
    "01": 4,
    "02": 5,
    "03": 1,
    "04": 4,
    "05": 4,
    "06": 7,
    "07": 2,
    "08": 4,
    "09": 5,
}

CARD_TARGETS = {
    "00": [520, 310],
    "01": [346, 347],
    "02": [275],
    "03": [520, 521],
    "04": [520, 521],
    "05": [346, 347],
    "06": [520, 310],
    "07": [449, 460],
    "08": [99, 110],
    "09": [275],
}

BASE_PRIORITY = [
    "00",
    "06",
    "09",
    "02",
    "03",
    "05",
    "04",
    "01",
    "08",
    "07",
]


@dataclass
class BattleState:
    fireball_priority: bool = False
    hogs_played_since_barrel: bool = True
    guards_played: int = 0


@dataclass
class DetectedHand:
    slots: dict[str, str] = field(default_factory=dict)
    water: int = 0


class GridMapper:
    def __init__(self) -> None:
        self.centers = self._build_centers()

    @staticmethod
    def _make_lines_drop_small(a: float, b: float, step: float, min_ratio: float) -> list[float]:
        lines = [float(a)]
        x = float(a) + step
        while x < float(b):
            lines.append(x)
            x += step
        remainder = float(b) - lines[-1]
        if remainder >= step * min_ratio:
            lines.append(float(b))
        return lines

    def _build_centers(self) -> dict[int, tuple[int, int]]:
        centers: dict[int, tuple[int, int]] = {}
        cell_id = 0
        for x0, y0, x1, y1 in GRID_REGIONS:
            x_lines = self._make_lines_drop_small(x0, x1, GRID_STEP, GRID_MIN_CELL_RATIO)
            y_lines = self._make_lines_drop_small(y0, y1, GRID_STEP, GRID_MIN_CELL_RATIO)
            if len(x_lines) < 2 or len(y_lines) < 2:
                continue
            for r in range(len(y_lines) - 1):
                for c in range(len(x_lines) - 1):
                    cx = int((x_lines[c] + x_lines[c + 1]) / 2.0)
                    cy = int((y_lines[r] + y_lines[r + 1]) / 2.0)
                    centers[cell_id] = (cx, cy)
                    cell_id += 1
        return centers

    def get_center(self, cell_id: int) -> tuple[int, int]:
        if cell_id not in self.centers:
            raise ValueError(f"未知网格ID: {cell_id}")
        return self.centers[cell_id]


class TapController:
    def __init__(self, device_id: str) -> None:
        client = AdbClient(host="127.0.0.1", port=5037)
        device = client.device(device_id)
        if device is None:
            raise RuntimeError(f"未找到设备 {device_id}")
        self.device = device

    def tap(self, x: int, y: int) -> None:
        self.device.shell(f"input tap {x} {y}")


def detect_hand_and_water(frame_bottom: np.ndarray, classifier) -> DetectedHand:
    cards: dict[str, str] = {}
    water = 0
    for region_name, coords in CARD_REGIONS:
        crop = crop_region(frame_bottom, coords)
        label = classify_card(crop, classifier)
        if region_name in {"1", "2", "3", "4"}:
            cards[region_name] = label
        elif region_name == "water":
            if label.isdigit():
                water = int(label)
    return DetectedHand(slots=cards, water=water)


def build_priority(state: BattleState) -> list[str]:
    priority = list(BASE_PRIORITY)
    if state.fireball_priority:
        priority = ["08"] + [cid for cid in priority if cid != "08"]
    else:
        priority = [cid for cid in priority if cid != "08"] + ["08"]
    return priority


def select_card(hand: DetectedHand, state: BattleState) -> tuple[str, str] | None:
    priority = build_priority(state)
    for card_id in priority:
        if card_id not in hand.slots.values():
            continue
        cost = CARD_COSTS.get(card_id, 99)
        if hand.water < cost:
            continue
        if card_id == "07" and not state.hogs_played_since_barrel:
            continue
        slot = next(slot for slot, cid in hand.slots.items() if cid == card_id)
        return card_id, slot
    return None


def choose_target(card_id: str) -> int:
    options = CARD_TARGETS[card_id]
    return random.choice(options)


def update_state_after_play(card_id: str, state: BattleState) -> None:
    if card_id == "06":
        state.fireball_priority = True
        state.guards_played += 1
    elif card_id == "08":
        state.fireball_priority = False
    elif card_id == "02":
        state.hogs_played_since_barrel = True
    elif card_id == "07":
        state.hogs_played_since_barrel = False


def play_card(
    tapper: TapController,
    grid: GridMapper,
    slot: str,
    card_id: str,
) -> None:
    region = next(coords for name, coords in CARD_REGIONS if name == slot)
    (x1, y1), (x2, y2) = region
    card_x = int((x1 + x2) / 2)
    card_y = int((y1 + y2) / 2)
    tapper.tap(card_x, card_y)

    target_id = choose_target(card_id)
    target_x, target_y = grid.get_center(target_id)
    tapper.tap(target_x, target_y)


def process_frame(
    frame: np.ndarray,
    classifier,
    tapper: TapController,
    grid: GridMapper,
    state: BattleState,
) -> bool:
    height = frame.shape[0]
    top_h = int(height * 0.8)
    bottom_part = frame[top_h:, :]
    hand = detect_hand_and_water(bottom_part, classifier)

    selection = select_card(hand, state)
    if selection is None:
        return False
    card_id, slot = selection
    play_card(tapper, grid, slot, card_id)
    update_state_after_play(card_id, state)
    return True


def main(video_path: str, device_id: str = "emulator-5556") -> None:
    classifier = load_classifier(Path(__file__).resolve().parents[1])
    tapper = TapController(device_id)
    grid = GridMapper()
    state = BattleState()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame, classifier, tapper, grid, state)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        raise SystemExit("用法: python battle/dummy.py <video_path>")
    main(sys.argv[1])
