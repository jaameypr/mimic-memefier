import time
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/blaze_face_short_range.tflite"

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
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)

    return bg

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.6,
    )
    detector = vision.FaceDetector.create_from_options(options)
    t0 = time.time()

    hamster = cv2.imread("assets/fish.png", cv2.IMREAD_UNCHANGED)
    if hamster is None or hamster.shape[2] != 4:
        raise RuntimeError("hamster must be a PNG with alpha (RGBA).")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        ts_ms = int((time.time() - t0) * 1000) # timestamp in milliseconds
        result = detector.detect_for_video(mp_image, ts_ms)

        out = frame_bgr

        if result.detections:
            det = result.detections[0]
            box = det.bounding_box
            H, W = frame_bgr.shape[:2]

            x = int(box.origin_x)
            y = int(box.origin_y)
            w = int(box.width)
            h = int(box.height)

            scale = w / hamster.shape[1]
            new_w = int(hamster.shape[1] * scale)
            new_h = int(hamster.shape[0] * scale)

            resized = cv2.resize(hamster, (new_w, new_h), interpolation=cv2.INTER_AREA)

            hx = x + w // 2 - new_w // 2
            hy = y - new_h - 10

            out = overlay_rgba(frame_bgr, resized, hx, hy)

            #debug
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)



        cv2.imshow("face anchor", out)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()