import cv2
import numpy as np

def overlay_rgba(background_bgr: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    bg = background_bgr.copy();

    h, w = overlay_rgba.shape[:2]
    H, W = bg.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x+w), min(H, y+h)
    if x1 >= x2 or y1 >= y2:
        return bg

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 -x1), oy1 + (y2 -y1)

    overlay_crop = overlay_rgba[oy1:oy2, ox1:ox2]
    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32);

    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0

    bg_crop = bg[y1:y2, x1:x2].astype(np.float32)

    blended = alpha * overlay_rgb + (1 - alpha) * bg_crop
    bg[y1:y2, x1:x2].astype(np.float32)

    blended = alpha * overlay_rgb + (1 - alpha) * bg_crop
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)

    return bg

def main():
    cap = cv2.VideoCapture(0)

    overlay = cv2.imread("assets/fish.png", cv2.IMREAD_UNCHANGED)
    if overlay is None or overlay.shape[2] != 4:
        raise RuntimeError("overlay must be a PNG with alpha (RGBA).")

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out = overlay_rgba(frame, overlay, x=30, y=30)
        cv2.imshow("overlay", out)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()