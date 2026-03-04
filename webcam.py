import time
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FACE_DETECTOR_MODEL = "models/blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = "models/face_landmarker.task"
HAND_LANDMARKER_MODEL = "models/hand_landmarker.task"

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


def compute_hand_signals(hand_landmarks, face_center_y: float | None, W: int, H: int) -> dict:
    """
    Compute hand signals from hand landmarks.
    Returns dict with hands_up (bool) and hand_center (x, y) in pixels.
    """
    # Wrist = landmark 0, Middle finger MCP = landmark 9 (rough palm center)
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]

    hand_cx = (wrist.x + middle_mcp.x) / 2.0 * W
    hand_cy = (wrist.y + middle_mcp.y) / 2.0 * H

    hands_up = False
    if face_center_y is not None:
        hands_up = hand_cy < face_center_y  # hand above face center

    # Open palm vs fist: compare distances between fingertips and wrist
    # Fingertip indices: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    fingertips = [hand_landmarks[i] for i in [4, 8, 12, 16, 20]]
    wrist_pt = np.array([wrist.x * W, wrist.y * H])
    tip_dists = [dist(np.array([ft.x * W, ft.y * H]), wrist_pt) for ft in fingertips]
    avg_tip_dist = sum(tip_dists) / len(tip_dists)

    # MCP indices for reference scale
    mcp_pts = [hand_landmarks[i] for i in [5, 9, 13, 17]]
    mcp_dists = [dist(np.array([m.x * W, m.y * H]), wrist_pt) for m in mcp_pts]
    avg_mcp_dist = sum(mcp_dists) / len(mcp_dists) + 1e-6

    open_palm = (avg_tip_dist / avg_mcp_dist) > 1.6  # fingers extended

    return {
        "hands_up": hands_up,
        "hand_cx": hand_cx,
        "hand_cy": hand_cy,
        "open_palm": open_palm,
    }


def choose_meme(features: dict, hand_signals: dict | None = None) -> str:
    """
    Rule-based mapper from features to meme key (matches MEME_MAP keys).
    Hand signals can override face-based meme selection.
    """
    mouth_open = features["mouth_open"]
    smile = features["smile"]
    brow = features["brow_raise"]
    eye_open = features["eye_open"]

    # Hand gesture overrides
    if hand_signals is not None:
        hands_up = hand_signals.get("hands_up", False)
        open_palm = hand_signals.get("open_palm", False)

        if hands_up and open_palm:
            return "ultra_happy"

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

    # Hand landmarker
    hand_base = python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL)
    hand_opts = vision.HandLandmarkerOptions(
        base_options=hand_base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

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

        # All three models
        det_res = face_detector.detect_for_video(mp_image, ts_ms)
        lm_res = face_landmarker.detect_for_video(mp_image, ts_ms)
        hand_res = hand_landmarker.detect_for_video(mp_image, ts_ms)

        out = frame_bgr.copy()

        meme_key = "neutral"
        features_smoothed = None
        face_center_y = None

        if lm_res.face_landmarks:
            feats = compute_features(lm_res.face_landmarks[0], W=W, H=H)

            feats["mouth_open"] = ema_mouth.update(feats["mouth_open"])
            feats["smile"] = ema_smile.update(feats["smile"])
            feats["brow_raise"] = ema_brow.update(feats["brow_raise"])
            feats["eye_open"] = ema_eye.update(feats["eye_open"])
            features_smoothed = feats

            # Compute face center Y for hands_up detection
            nose_tip = lm_res.face_landmarks[0][1]  # landmark 1 = nose tip
            face_center_y = nose_tip.y * H

        # Hand signals
        hand_signals = None
        num_hands = 0
        if hand_res.hand_landmarks:
            num_hands = len(hand_res.hand_landmarks)
            # Use the first detected hand for signals
            h_sigs = compute_hand_signals(hand_res.hand_landmarks[0], face_center_y, W, H)

            hand_signals = h_sigs

            # Draw hand landmarks for debug
            for hand_lms in hand_res.hand_landmarks:
                for lm in hand_lms:
                    px, py = int(lm.x * W), int(lm.y * H)
                    cv2.circle(out, (px, py), 3, (255, 0, 255), -1)

        # Choose meme (face features + hand signals)
        if features_smoothed is not None:
            meme_key = choose_meme(features_smoothed, hand_signals)
        elif hand_signals is not None:
            # No face but hand detected – use default features with hand override
            default_feats = {"mouth_open": 0, "smile": 0.35, "brow_raise": 0.05, "eye_open": 0.03}
            meme_key = choose_meme(default_feats, hand_signals)

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

        # Stabilize meme selection (majority vote + hold time)
        meme_key = meme_stabilizer.update(meme_key)

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
            cv2.putText(out, "Nothing detected", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Hand debug info
        hand_y = 165
        cv2.putText(out, f"hands: {num_hands}", (20, hand_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
        if hand_signals is not None:
            cv2.putText(out, f"hands_up:  {hand_signals['hands_up']}", (20, hand_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
            cv2.putText(out, f"open_palm: {hand_signals['open_palm']}", (20, hand_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        cv2.imshow("mimics-memefied", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

