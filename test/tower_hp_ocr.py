import argparse

import cv2
import numpy as np
import pytesseract

LEFT_ROI = (140, 155, 210, 183)
RIGHT_ROI = (518, 156, 589, 181)


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def recognize_digits(binary_roi: np.ndarray) -> str:
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(binary_roi, config=config)
    return "".join(ch for ch in text if ch.isdigit())


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
    args = parser.parse_args()

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

        left_digits = recognize_digits(left_bin)
        right_digits = recognize_digits(right_bin)

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
