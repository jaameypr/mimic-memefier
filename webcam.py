import time
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FACE_DETECTOR_MODEL = "models/blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = "models/face_landmarker.task"

MEME_MAP = {
    "blushing": "assets/blushing.png",
    "creepedout": "assets/creepedout.png",
    "disgusted": "assets/disgusted.png",
    "fish": "assets/fish.png",
    "funny": "assets/funny.png",
    "neutral": "assets/neutral.png",
    "sleeping": "assets/sleeping.png",
    "ultra_happy": "assets/ultra_happy.png",
    "ultra_laugh": "assets/ultra_laugh.png",
}


class EMA:
    """Exponential moving average smoothing."""
    def __init__(self, alpha=0.25):
        self.alpha = float(alpha)
        self.value = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def overlay_rgba(background_bgr: np.ndarray, overlay_rgba_img: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-blend an RGBA image onto a BGR background at top-left (x,y)."""
    bg = background_bgr.copy()
    if overlay_rgba_img is None or overlay_rgba_img.ndim != 3 or overlay_rgba_img.shape[2] != 4:
        return bg

    h, w = overlay_rgba_img.shape[:2]
    H, W = bg.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return bg

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    overlay_crop = overlay_rgba_img[oy1:oy2, ox1:ox2]
    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32)
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0

    bg_crop = bg[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * overlay_rgb + (1 - alpha) * bg_crop
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)
    return bg


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# Landmark indices (MediaPipe Face Mesh topology)
IDX_UPPER_LIP = 13
IDX_LOWER_LIP = 14
IDX_MOUTH_L = 61
IDX_MOUTH_R = 291
IDX_CHEEK_L = 234
IDX_CHEEK_R = 454
IDX_BROW_L = 105
IDX_EYE_L = 33
IDX_BROW_R = 334
IDX_EYE_R = 263

# Optional eye openness indices (simple approximation)
IDX_EYE_L_TOP = 159
IDX_EYE_L_BOTTOM = 145
IDX_EYE_R_TOP = 386
IDX_EYE_R_BOTTOM = 374


def compute_features(face_landmarks, W: int, H: int) -> dict:
    """
    Compute a few simple, interpretable facial features from landmarks.
    Returns normalized ratios (scale-invariant-ish).
    """
    pts = np.array([(p.x * W, p.y * H) for p in face_landmarks], dtype=np.float32)

    upper = pts[IDX_UPPER_LIP]
    lower = pts[IDX_LOWER_LIP]
    mouth_l = pts[IDX_MOUTH_L]
    mouth_r = pts[IDX_MOUTH_R]
    cheek_l = pts[IDX_CHEEK_L]
    cheek_r = pts[IDX_CHEEK_R]
    brow_l = pts[IDX_BROW_L]
    eye_l = pts[IDX_EYE_L]
    brow_r = pts[IDX_BROW_R]
    eye_r = pts[IDX_EYE_R]

    face_width = dist(cheek_l, cheek_r) + 1e-6
    mouth_width = dist(mouth_l, mouth_r) + 1e-6

    mouth_open = dist(upper, lower) / mouth_width
    smile = mouth_width / face_width
    brow_raise = (dist(brow_l, eye_l) + dist(brow_r, eye_r)) / (2.0 * face_width)

    # Eye openness (useful for "sleeping")
    eye_l_open = dist(pts[IDX_EYE_L_TOP], pts[IDX_EYE_L_BOTTOM]) / face_width
    eye_r_open = dist(pts[IDX_EYE_R_TOP], pts[IDX_EYE_R_BOTTOM]) / face_width
    eye_open = (eye_l_open + eye_r_open) / 2.0

    return {
        "mouth_open": float(mouth_open),
        "smile": float(smile),
        "brow_raise": float(brow_raise),
        "eye_open": float(eye_open),
    }


def choose_meme(features: dict) -> str:
    """
    Rule-based mapper from features to meme key (matches MEME_MAP keys).
    Tune thresholds by watching live numbers on screen.
    """
    mouth_open = features["mouth_open"]
    smile = features["smile"]
    brow = features["brow_raise"]
    eye_open = features["eye_open"]

    if eye_open < 0.018:
        return "sleeping"

    if mouth_open > 0.22 and smile > 0.40:
        return "ultra_happy"

    if smile < 0.45 and mouth_open < 0.35 and brow < 0.15:
        return "disgusted"

    if mouth_open > 0.20:
        return "funny"

    if smile > 0.42:
        return "fish"

    if smile > 0.35:
        return "neutral"

    if brow > 0.070:
        return "creepedout"

    if smile > 0.30:
        return "blushing"

    return "neutral"


def main():
    # pre convert all images to rgba
    for key, path in MEME_MAP.items():
        img = Image.open(path)
        img = img.convert("RGBA")
        img.save(path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # Load all meme assets
    MEMES = {}
    for key, path in MEME_MAP.items():
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            raise RuntimeError(f"Could not load {path} due to missing alpha channel")

        MEMES[key] = img


    det_base = python.BaseOptions(model_asset_path=FACE_DETECTOR_MODEL)
    det_opts = vision.FaceDetectorOptions(
        base_options=det_base,
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.6,
    )
    face_detector = vision.FaceDetector.create_from_options(det_opts)

    lm_base = python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL)
    lm_opts = vision.FaceLandmarkerOptions(
        base_options=lm_base,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(lm_opts)

    ema_x, ema_y, ema_w, ema_h = EMA(0.35), EMA(0.35), EMA(0.35), EMA(0.35)
    ema_mouth = EMA(0.25)
    ema_smile = EMA(0.25)
    ema_brow = EMA(0.25)
    ema_eye = EMA(0.25)

    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        ts_ms = int((time.time() - t0) * 1000)

        #both mnodels
        det_res = face_detector.detect_for_video(mp_image, ts_ms)
        lm_res = face_landmarker.detect_for_video(mp_image, ts_ms)

        out = frame_bgr.copy()

        meme_key = "neutral"
        features_smoothed = None

        if lm_res.face_landmarks:
            feats = compute_features(lm_res.face_landmarks[0], W=W, H=H)

            feats["mouth_open"] = ema_mouth.update(feats["mouth_open"])
            feats["smile"] = ema_smile.update(feats["smile"])
            feats["brow_raise"] = ema_brow.update(feats["brow_raise"])
            feats["eye_open"] = ema_eye.update(feats["eye_open"])
            features_smoothed = feats

            meme_key = choose_meme(feats)

        # Bounding box (anchor overlay) - use most confident detection
        if det_res.detections:
            det = max(det_res.detections, key=lambda d: d.categories[0].score if d.categories else 0.0)
            box = det.bounding_box

            x = int(ema_x.update(box.origin_x))
            y = int(ema_y.update(box.origin_y))
            w = int(ema_w.update(box.width))
            h = int(ema_h.update(box.height))

            meme_img = MEMES.get(meme_key, MEMES["neutral"])

            scale = w / meme_img.shape[1]
            new_w = int(meme_img.shape[1] * scale)
            new_h = int(meme_img.shape[0] * scale)

            new_w = clamp(new_w, 40, 700)
            new_h = clamp(new_h, 40, 700)

            resized = cv2.resize(meme_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # above head
            hx = x + w // 2 - new_w // 2
            hy = y - new_h - 10

            out = overlay_rgba(out, resized, hx, hy)

            # debug
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(out, f"meme: {meme_key}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if features_smoothed is not None:
            cv2.putText(out, f"mouth_open: {features_smoothed['mouth_open']:.3f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(out, f"smile:     {features_smoothed['smile']:.3f}", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(out, f"brow:      {features_smoothed['brow_raise']:.3f}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(out, f"eye_open:  {features_smoothed['eye_open']:.3f}", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            cv2.putText(out, "Nothign detected", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("mimics-memefied", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()