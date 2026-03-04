import time
import json
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from collections import deque, Counter

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


with open("config.json", "r") as _f:
    CONFIG = json.load(_f)

FACE_DETECTOR_MODEL = CONFIG["face_detector_model"]
FACE_LANDMARKER_MODEL = CONFIG["face_landmarker_model"]
HAND_LANDMARKER_MODEL = CONFIG["hand_landmarker_model"]

MEME_MAP = CONFIG["meme_map"]
THRESHOLDS = CONFIG["thresholds"]
SMOOTHING = CONFIG["smoothing"]
DETECTION = CONFIG["detection"]
OVERLAY = CONFIG["overlay"]
DISPLAY = CONFIG["display"]


class EMA:
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


class MemeStabilizer:
    def __init__(self, window=12, hold_ms=450):
        self.window = deque(maxlen=window)
        self.hold_ms = hold_ms
        self.current = "neutral"
        self.last_switch = 0.0

    def update(self, candidate: str) -> str:
        self.window.append(candidate)
        most_common = Counter(self.window).most_common(1)[0][0]

        now = time.time() * 1000
        if most_common != self.current and (now - self.last_switch) >= self.hold_ms:
            self.current = most_common
            self.last_switch = now
        return self.current


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def overlay_rgba(background_bgr: np.ndarray, overlay_rgba_img: np.ndarray, x: int, y: int) -> np.ndarray:
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

IDX_EYE_L_TOP = 159
IDX_EYE_L_BOTTOM = 145
IDX_EYE_R_TOP = 386
IDX_EYE_R_BOTTOM = 374


def compute_features(face_landmarks, W: int, H: int) -> dict:
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
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]

    hand_cx = (wrist.x + middle_mcp.x) / 2.0 * W
    hand_cy = (wrist.y + middle_mcp.y) / 2.0 * H

    hands_up = False
    if face_center_y is not None:
        hands_up = hand_cy < face_center_y

    fingertips = [hand_landmarks[i] for i in [4, 8, 12, 16, 20]]
    wrist_pt = np.array([wrist.x * W, wrist.y * H])
    tip_dists = [dist(np.array([ft.x * W, ft.y * H]), wrist_pt) for ft in fingertips]
    avg_tip_dist = sum(tip_dists) / len(tip_dists)

    mcp_pts = [hand_landmarks[i] for i in [5, 9, 13, 17]]
    mcp_dists = [dist(np.array([m.x * W, m.y * H]), wrist_pt) for m in mcp_pts]
    avg_mcp_dist = sum(mcp_dists) / len(mcp_dists) + 1e-6

    open_palm = (avg_tip_dist / avg_mcp_dist) > DETECTION["open_palm_ratio"]

    return {
        "hands_up": hands_up,
        "hand_cx": hand_cx,
        "hand_cy": hand_cy,
        "open_palm": open_palm,
    }


def choose_meme(features: dict, hand_signals: dict | None = None) -> str:
    mouth_open = features["mouth_open"]
    smile = features["smile"]
    brow = features["brow_raise"]
    eye_open = features["eye_open"]

    if hand_signals is not None:
        hands_up = hand_signals.get("hands_up", False)
        open_palm = hand_signals.get("open_palm", False)
        if hands_up and open_palm:
            return "ultra_happy"

    if eye_open < THRESHOLDS["eye_closed"]:
        return "sleeping"

    if mouth_open > THRESHOLDS["mouth_open_happy"] and smile > THRESHOLDS["smile_happy"]:
        return "ultra_happy"

    if smile < THRESHOLDS["smile_disgusted_max"] and mouth_open < THRESHOLDS["mouth_open_disgusted_max"] and brow < THRESHOLDS["brow_disgusted_max"]:
        return "disgusted"

    if mouth_open > THRESHOLDS["mouth_open_funny"]:
        return "funny"

    if smile > THRESHOLDS["smile_fish"]:
        return "fish"

    if smile > THRESHOLDS["smile_neutral"]:
        return "neutral"

    if brow > THRESHOLDS["brow_creepedout"]:
        return "creepedout"

    if smile > THRESHOLDS["smile_blushing"]:
        return "blushing"

    return "neutral"


def render_debug(frame_bgr, meme_img_resized, meme_key, features_smoothed,
                 hand_signals, num_hands, det_box, ema_x, ema_y, ema_w, ema_h):
    out = frame_bgr.copy()
    H, W = out.shape[:2]

    if det_box is not None:
        box = det_box
        x = int(ema_x.update(box.origin_x))
        y = int(ema_y.update(box.origin_y))
        w = int(ema_w.update(box.width))
        h = int(ema_h.update(box.height))

        if meme_img_resized is not None:
            mh, mw = meme_img_resized.shape[:2]
            hx = x + w // 2 - mw // 2
            hy = y - mh + OVERLAY["y_offset"]
            out = overlay_rgba(out, meme_img_resized, hx, hy)

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

    hand_y = 165
    cv2.putText(out, f"hands: {num_hands}", (20, hand_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    if hand_signals is not None:
        cv2.putText(out, f"hands_up:  {hand_signals['hands_up']}", (20, hand_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
        cv2.putText(out, f"open_palm: {hand_signals['open_palm']}", (20, hand_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

    return out


def render_overlay(frame_bgr, meme_img_resized, meme_key,
                   det_box, ema_x, ema_y, ema_w, ema_h,
                   face_landmarks, MEMES, W, H):
    out = frame_bgr.copy()

    if det_box is not None and face_landmarks is not None:
        box = det_box
        x = int(ema_x.update(box.origin_x))
        y = int(ema_y.update(box.origin_y))
        w = int(ema_w.update(box.width))
        h = int(ema_h.update(box.height))

        meme_raw = MEMES.get(meme_key, MEMES["neutral"])

        face_scale = max(w, h) / max(meme_raw.shape[1], meme_raw.shape[0])
        new_w = int(meme_raw.shape[1] * face_scale * 1.3)
        new_h = int(meme_raw.shape[0] * face_scale * 1.3)
        new_w = clamp(new_w, OVERLAY["min_size"], OVERLAY["max_size"])
        new_h = clamp(new_h, OVERLAY["min_size"], OVERLAY["max_size"])

        resized = cv2.resize(meme_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cx = x + w // 2
        cy = y + h // 2
        ox = cx - new_w // 2
        oy = cy - new_h // 2

        out = overlay_rgba(out, resized, ox, oy)

    return out


def render_split(frame_bgr, meme_key, MEMES):
    H, W = frame_bgr.shape[:2]

    meme_raw = MEMES.get(meme_key, MEMES["neutral"])
    mh, mw = meme_raw.shape[:2]

    scale = H / mh
    scaled_w = int(mw * scale)
    scaled_h = H
    meme_resized = cv2.resize(meme_raw, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    meme_bgr = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
    if meme_resized.shape[2] == 4:
        rgb = meme_resized[:, :, :3].astype(np.float32)
        alpha = meme_resized[:, :, 3:4].astype(np.float32) / 255.0
        bg = meme_bgr.astype(np.float32)
        meme_bgr = (alpha * rgb + (1 - alpha) * bg).astype(np.uint8)
    else:
        meme_bgr = meme_resized[:, :, :3]

    out = np.hstack([frame_bgr, meme_bgr])
    return out


def main():
    for key, path in MEME_MAP.items():
        img = Image.open(path)
        img = img.convert("RGBA")
        img.save(path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

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
        min_detection_confidence=DETECTION["min_face_detection_confidence"],
    )
    face_detector = vision.FaceDetector.create_from_options(det_opts)

    lm_base = python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL)
    lm_opts = vision.FaceLandmarkerOptions(
        base_options=lm_base,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=DETECTION["num_faces"],
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(lm_opts)

    hand_base = python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL)
    hand_opts = vision.HandLandmarkerOptions(
        base_options=hand_base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=DETECTION["num_hands"],
        min_hand_detection_confidence=DETECTION["min_hand_detection_confidence"],
        min_hand_presence_confidence=DETECTION["min_hand_presence_confidence"],
        min_tracking_confidence=DETECTION["min_hand_tracking_confidence"],
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

    pos_alpha = SMOOTHING["ema_position_alpha"]
    feat_alpha = SMOOTHING["ema_feature_alpha"]

    ema_x, ema_y, ema_w, ema_h = EMA(pos_alpha), EMA(pos_alpha), EMA(pos_alpha), EMA(pos_alpha)
    ema_mouth = EMA(feat_alpha)
    ema_smile = EMA(feat_alpha)
    ema_brow = EMA(feat_alpha)
    ema_eye = EMA(feat_alpha)

    meme_stabilizer = MemeStabilizer(
        window=SMOOTHING["stabilizer_window"],
        hold_ms=SMOOTHING["stabilizer_hold_ms"],
    )

    modes = DISPLAY["modes"]
    mode_idx = modes.index(DISPLAY["default_mode"]) if DISPLAY["default_mode"] in modes else 0
    display_mode = modes[mode_idx]

    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        ts_ms = int((time.time() - t0) * 1000)

        det_res = face_detector.detect_for_video(mp_image, ts_ms)
        lm_res = face_landmarker.detect_for_video(mp_image, ts_ms)
        hand_res = hand_landmarker.detect_for_video(mp_image, ts_ms)

        meme_key = "neutral"
        features_smoothed = None
        face_center_y = None
        face_landmarks = None

        if lm_res.face_landmarks:
            face_landmarks = lm_res.face_landmarks[0]
            feats = compute_features(face_landmarks, W=W, H=H)

            feats["mouth_open"] = ema_mouth.update(feats["mouth_open"])
            feats["smile"] = ema_smile.update(feats["smile"])
            feats["brow_raise"] = ema_brow.update(feats["brow_raise"])
            feats["eye_open"] = ema_eye.update(feats["eye_open"])
            features_smoothed = feats

            nose_tip = face_landmarks[1]
            face_center_y = nose_tip.y * H

        hand_signals = None
        num_hands = 0
        if hand_res.hand_landmarks:
            num_hands = len(hand_res.hand_landmarks)
            h_sigs = compute_hand_signals(hand_res.hand_landmarks[0], face_center_y, W, H)
            hand_signals = h_sigs

        if features_smoothed is not None:
            meme_key = choose_meme(features_smoothed, hand_signals)
        elif hand_signals is not None:
            default_feats = {"mouth_open": 0, "smile": 0.35, "brow_raise": 0.05, "eye_open": 0.03}
            meme_key = choose_meme(default_feats, hand_signals)

        meme_key = meme_stabilizer.update(meme_key)

        det_box = None
        if det_res.detections:
            det = max(det_res.detections, key=lambda d: d.categories[0].score if d.categories else 0.0)
            det_box = det.bounding_box

        meme_img_resized = None
        if det_box is not None:
            meme_raw = MEMES.get(meme_key, MEMES["neutral"])
            box_w = int(ema_w.value if ema_w.value else det_box.width)
            scale = box_w / meme_raw.shape[1]
            new_w = clamp(int(meme_raw.shape[1] * scale), OVERLAY["min_size"], OVERLAY["max_size"])
            new_h = clamp(int(meme_raw.shape[0] * scale), OVERLAY["min_size"], OVERLAY["max_size"])
            meme_img_resized = cv2.resize(meme_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_with_hands = frame_bgr.copy()
        if hand_res.hand_landmarks and display_mode == "debug":
            for hand_lms in hand_res.hand_landmarks:
                for lm in hand_lms:
                    px, py = int(lm.x * W), int(lm.y * H)
                    cv2.circle(frame_with_hands, (px, py), 3, (255, 0, 255), -1)

        if display_mode == "debug":
            out = render_debug(
                frame_with_hands, meme_img_resized, meme_key, features_smoothed,
                hand_signals, num_hands, det_box, ema_x, ema_y, ema_w, ema_h,
            )
        elif display_mode == "overlay":
            out = render_overlay(
                frame_bgr, meme_img_resized, meme_key,
                det_box, ema_x, ema_y, ema_w, ema_h,
                face_landmarks, MEMES, W, H,
            )
        elif display_mode == "split":
            out = render_split(frame_bgr, meme_key, MEMES)
        elif display_mode == "none":
            out = frame_bgr.copy()
        else:
            out = frame_bgr.copy()

        if display_mode == "debug":
            oh, ow = out.shape[:2]
            mode_text = f"[TAB] mode: {display_mode}"
            cv2.putText(out, mode_text, (ow - 300, oh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("mimics-memefied", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 9:
            mode_idx = (mode_idx + 1) % len(modes)
            display_mode = modes[mode_idx]
            print(f"Display mode: {display_mode}")
        elif key == ord("1"):
            display_mode = "overlay"
            mode_idx = modes.index("overlay")
            print(f"Display mode: {display_mode}")
        elif key == ord("2"):
            display_mode = "debug"
            mode_idx = modes.index("debug")
            print(f"Display mode: {display_mode}")
        elif key == ord("3"):
            display_mode = "split"
            mode_idx = modes.index("split")
            print(f"Display mode: {display_mode}")
        elif key == ord("4"):
            display_mode = "none"
            mode_idx = modes.index("none")
            print(f"Display mode: {display_mode}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
