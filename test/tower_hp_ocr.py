import argparse
from pathlib import Path

import cv2
import numpy as np

LEFT_ROI = (140, 155, 210, 183)
RIGHT_ROI = (518, 156, 589, 181)


def load_digit_templates(template_dir: Path) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for digit in range(10):
        path = template_dir / f"{digit}.png"
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        templates[str(digit)] = img
    if not templates:
        raise FileNotFoundError(
            f"模板为空：请在 {template_dir} 放置 0.png~9.png 的数字模板图。"
        )
    return templates


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def recognize_digits(binary_roi: np.ndarray, templates: dict[str, np.ndarray]) -> str:
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = binary_roi.shape
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 8:
            continue
        if bh < h * 0.4:
            continue
        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: b[0])
    if not boxes:
        return ""

    best_chars = []
    for x, y, bw, bh in boxes:
        digit_roi = binary_roi[y : y + bh, x : x + bw]
        best_digit = ""
        best_score = -1.0
        for digit, tmpl in templates.items():
            resized = cv2.resize(digit_roi, (tmpl.shape[1], tmpl.shape[0]))
            score = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best_score = score
                best_digit = digit
        if best_digit:
            best_chars.append(best_digit)

    return "".join(best_chars)


def draw_roi(frame: np.ndarray, roi: tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="公主塔血量识别测试")
    parser.add_argument(
        "--source",
        default=0,
        help="视频流来源：摄像头索引(默认0)或视频文件路径",
    )
    parser.add_argument(
        "--templates",
        default="test/tower_digit_templates",
        help="数字模板目录，包含 0.png~9.png",
    )
    args = parser.parse_args()

    template_dir = Path(args.templates)
    templates = load_digit_templates(template_dir)

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频流: {args.source}")

    window_name = "tower-hp-ocr"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left_crop = frame[LEFT_ROI[1] : LEFT_ROI[3], LEFT_ROI[0] : LEFT_ROI[2]]
        right_crop = frame[RIGHT_ROI[1] : RIGHT_ROI[3], RIGHT_ROI[0] : RIGHT_ROI[2]]

        left_bin = preprocess_roi(left_crop)
        right_bin = preprocess_roi(right_crop)

        left_digits = recognize_digits(left_bin, templates)
        right_digits = recognize_digits(right_bin, templates)

        print(f"\r左塔: {left_digits or '-'}  右塔: {right_digits or '-'}", end="")

        draw_roi(frame, LEFT_ROI, f"L:{left_digits or '-'}")
        draw_roi(frame, RIGHT_ROI, f"R:{right_digits or '-'}")

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
