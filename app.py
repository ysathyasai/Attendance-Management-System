from flask import (Flask, Response, request, jsonify, session,
                   redirect, url_for, render_template, send_file)
from werkzeug.middleware.proxy_fix import ProxyFix

import cv2 as cv
import numpy as np
import os, sys, re, time, threading, socket, smtplib, base64, json
import pymysql
import pymysql.cursors
from datetime import datetime
from email.message import EmailMessage
from functools import wraps
from collections import deque, defaultdict

try:
    from ultralytics import YOLO
    import supervision as sv
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics / supervision not installed — YOLO disabled")

# ================================================================
#  PATHS
# ================================================================
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ALERT_DIR      = os.path.join(BASE_DIR, "alerts")
SAMPLES_DIR    = os.path.join(BASE_DIR, "photo samples")
MODEL_PATH     = os.path.join(BASE_DIR, "classifier.xml")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")
ALARM_PATH     = os.path.join(BASE_DIR, "alarm.wav")
HAAR_PATH      = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
YOLO_PATH      = os.path.join(BASE_DIR, "yolo26.pt")
os.makedirs(ALERT_DIR,   exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ================================================================
#  CONFIG
# ================================================================
USERNAME   = "sai"
PASSWORD   = "9866"
EMAIL_FROM = "attedancemanagementsystem012@gmail.com"
EMAIL_PASS = "gngb tnrc pztb trrj"
NGROK_LINK = "https://corsage-polo-lumping.ngrok-free.dev"

DB_CONFIG = dict(
    host       = "localhost",
    user       = "root",
    password   = "SAICHARANTEJA@143...",
    database   = "attendance_management",
    autocommit = True,
    charset    = "utf8mb4",
)

# ================================================================
#  LBPH THRESHOLDS — v6 FIXED
#
#  ROOT CAUSE OF WRONG-NAME BUG (v5 leftover):
#    Training always used for_dark=False preprocessing.
#    Prediction sometimes used for_dark=True (different CLAHE clip,
#    different gamma) → distance inflated by 15-25 points → wrong winner.
#
#  FIX v6:
#    1. ONE preprocessing path: normalize_face() — used identically
#       at training AND prediction time. No for_dark branch at all.
#    2. bilateralFilter moved BEFORE CLAHE (correct order):
#       denoise first, then enhance contrast.
#    3. CLAHE clipLimit fixed at 2.0 everywhere (no dark/bright split).
#    4. Voting: 5 crops, need ≥3 votes + dist ≤ KNOWN_MAX + margin ≥ MIN_MARGIN.
#    5. DETECT_EVERY_N lowered to 1 so weapons are never skipped.
# ================================================================

# ── LBPH thresholds ──────────────────────────────────────────────
KNOWN_MAX      = 72     # accept as KNOWN only if mean dist ≤ 72
UNKNOWN_MIN    = 90
KNOWN_VOTE_MIN = 3      # majority of 5 crops
MIN_MARGIN     = 10.0   # winner must beat runner-up by 10 pts

# ── Alert / consecutive ──────────────────────────────────────────
ALERT_COOLDOWN     = 15
CONSECUTIVE_NEEDED = 3

# ── YOLO thresholds — FIXED ─────────────────────────────────────
#  OLD: YOLO ran at conf=0.10 then filtered per-class.
#  FIX: Run at conf=0.25 (cuts false positives at model level),
#       then apply tighter per-class thresholds below.
DETECT_EVERY_N       = 1          # FIX: was 2 — never skip weapon frames
YOLO_RUN_CONF        = 0.25       # FIX: was 0.10 — less noise from YOLO
WEAPON_CONF_DEFAULT  = 0.42       # knife / blade / gun etc.
SCISSORS_CONF        = 0.40       # FIX: was 0.60 — too high; nail cutters
                                  #      are classified as scissors by YOLO
MASK_CONF            = 0.38       # FIX: slightly lower to catch partial masks
PERSON_CONF          = 0.40

# ── Dark clothing ────────────────────────────────────────────────
DARK_PIXEL_RATIO = 0.65

# ── Stream ───────────────────────────────────────────────────────
MAX_VIDEO_MB        = 25
FRAME_BUFFER_MAX    = 200
STREAM_JPEG_QUALITY = 72
STALE_THRESHOLD     = 1.0
MIN_FRAME_GAP       = 0.040
MIN_DETECT_W        = 80
MIN_DETECT_H        = 80

# ================================================================
#  YOLO KEYWORDS — FIXED
#
#  OLD: used substring match ("fork" in "pitchfork")
#  FIX: exact word-boundary match via _weapon_match() helper.
#       Added "nail cutter", "cutter", "nail_cutter" explicitly.
# ================================================================
WEAPON_KEYWORDS = {
    "knife", "scissors", "baseball bat", "fork",
    "gun", "pistol", "rifle", "weapon", "handgun", "sword",
    "blade", "firearm", "shotgun", "revolver", "dagger",
    "cleaver", "bat", "nail", "cutter", "sharp",
    "nail cutter", "nail_cutter", "box cutter", "box_cutter",
    "letter opener", "sickle", "machete", "axe", "hatchet",
}
MASK_KEYWORDS = {
    "mask", "balaclava", "face_mask", "facemask",
    "with_mask", "covered_face", "ski_mask", "gas_mask", "surgical_mask",
}
IGNORE_ALERT_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "teddy bear", "hair drier",
    "toothbrush",
}


def _weapon_match(cls_lower: str) -> bool:
    """
    FIX: exact-token match prevents 'fork' matching 'pitchfork', etc.
    Checks both exact equality and token membership.
    """
    if cls_lower in WEAPON_KEYWORDS:
        return True
    tokens = re.split(r'[\s_\-]+', cls_lower)
    return any(tok in WEAPON_KEYWORDS for tok in tokens)


def _mask_match(cls_lower: str) -> bool:
    if cls_lower in MASK_KEYWORDS:
        return True
    tokens = re.split(r'[\s_\-]+', cls_lower)
    return any(tok in MASK_KEYWORDS for tok in tokens)


# ================================================================
#  RUNTIME STATE
# ================================================================
alarm_active     = False
last_alert_ts    = None
frame_lock       = threading.Lock()
latest_frame     = None
latest_raw_frame = None
frame_buffer     = deque(maxlen=FRAME_BUFFER_MAX)
frame_counter    = 0
_consec_count    = 0
_consec_lock     = threading.Lock()
_new_frame_event = threading.Event()

# ================================================================
#  YOLO
# ================================================================
yolo_model    = None
box_annotator = None
lbl_annotator = None

def _load_yolo():
    global yolo_model, box_annotator, lbl_annotator
    if not YOLO_AVAILABLE:
        return
    if not os.path.exists(YOLO_PATH):
        print(f"[WARN] yolo26.pt not found at {YOLO_PATH}")
        return
    try:
        print("[INFO] Loading yolo26.pt …")
        yolo_model    = YOLO(YOLO_PATH)
        box_annotator = sv.BoxAnnotator(thickness=2)
        lbl_annotator = sv.LabelAnnotator(text_scale=0.55, text_padding=4)
        print("\n" + "="*65)
        print("  CLASS NAMES IN yolo26.pt")
        print("="*65)
        weapon_found = []
        for idx, name in yolo_model.names.items():
            nl  = name.lower()
            is_weapon = _weapon_match(nl)
            is_mask   = _mask_match(nl)
            tag = "  ← ⚠ WEAPON" if is_weapon else \
                  "  ← 🎭 MASK"  if is_mask   else \
                  "  ← 👤 PERSON" if nl == "person" else ""
            print(f"  [{idx:>3}]  {name}{tag}")
            if is_weapon:
                weapon_found.append(name)
        print("="*65)
        if weapon_found:
            print(f"  ✅ Weapon classes active: {weapon_found}")
        print("="*65 + "\n")
    except Exception as e:
        print(f"[ERROR] Could not load yolo26.pt: {e}")
        yolo_model = None

_load_yolo()

# ================================================================
#  THREAD-LOCAL CASCADE
# ================================================================
_cascade_local = threading.local()

def _get_cascade():
    if not hasattr(_cascade_local, 'cascade'):
        path = HAAR_PATH if os.path.exists(HAAR_PATH) else \
               cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        c = cv.CascadeClassifier(path)
        if c.empty():
            print("[ERROR] Haar cascade failed to load!")
        _cascade_local.cascade = c
    return _cascade_local.cascade

# ================================================================
#  FACE PREPROCESSING — v6 UNIFIED
#
#  THE BUG IN v5:
#    Training:   _preprocess_face(img, for_dark=False)
#                → CLAHE clip=3.0, bilateral AFTER CLAHE, no gamma
#    Prediction: _preprocess_face(roi, for_dark=True) for dark frames
#                → CLAHE clip=4.0, gamma before CLAHE, different output
#    Any preprocessing difference at prediction vs training time
#    inflates LBPH distances by 15-30 points → wrong names.
#
#  FIX v6:
#    Single function normalize_face() with NO dark/bright branching.
#    Used identically at training AND prediction.
#    Steps: resize → bilateral (denoise) → CLAHE (enhance) → normalize
#    Bilateral BEFORE CLAHE is correct: denoise the raw input,
#    then enhance contrast on the clean signal.
# ================================================================
_clahe_shared = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def normalize_face(gray_roi: np.ndarray) -> np.ndarray:
    """
    SINGLE unified preprocessing. Called identically at train + predict.
    No for_dark branching — that was the v5 mismatch bug.
    """
    if gray_roi is None or gray_roi.size == 0:
        return np.zeros((200, 200), dtype=np.uint8)

    img = cv.resize(gray_roi, (200, 200))

    # Step 1: denoise FIRST (bilateral preserves edges, removes noise)
    img = cv.bilateralFilter(img, d=7, sigmaColor=45, sigmaSpace=45)

    # Step 2: CLAHE — consistent clip=2.0 everywhere
    img = _clahe_shared.apply(img)

    # Step 3: stretch to full [0,255] range
    mn, mx = int(img.min()), int(img.max())
    if mx > mn:
        img = np.clip(
            (img.astype(np.int32) - mn) * 255 // (mx - mn),
            0, 255
        ).astype(np.uint8)

    return img


def _safe_detect_faces(gray_img, is_dark=False):
    if gray_img is None:
        return []
    h, w = gray_img.shape[:2]
    if w < MIN_DETECT_W or h < MIN_DETECT_H:
        return []
    cascade = _get_cascade()
    if cascade.empty():
        return []

    # For dark images, apply gamma + CLAHE before detection only
    detect_img = gray_img
    if is_dark:
        inv_gamma = 1.0 / 1.8
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        detect_img = cv.LUT(gray_img, table)
        tmp_clahe  = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        detect_img = tmp_clahe.apply(detect_img)

    min_neigh = 4 if is_dark else 5
    for scale in [1.05, 1.1, 1.15, 1.2]:
        try:
            faces = cascade.detectMultiScale(
                detect_img, scaleFactor=scale, minNeighbors=min_neigh,
                minSize=(50, 50), flags=cv.CASCADE_SCALE_IMAGE)
            if len(faces) > 0:
                return faces
        except cv.error:
            continue
        except Exception as e:
            print(f"   [CASCADE error] {e}")
            return []
    return []

# ================================================================
#  NAME LOOKUP
# ================================================================
_name_cache      : dict = {}
_name_cache_lock = threading.Lock()

def _get_owner_name(owner_id: int) -> str:
    with _name_cache_lock:
        cached = _name_cache.get(owner_id)
    if cached:
        return cached
    conn = None
    try:
        conn = _db()
        cur  = conn.cursor()
        cur.execute("SELECT name FROM owner_details WHERE owner_id = %s LIMIT 1", (owner_id,))
        row = cur.fetchone()
        if row and row[0]:
            name = str(row[0]).strip()
            with _name_cache_lock:
                _name_cache[owner_id] = name
            return name
    except Exception as e:
        print(f"   [NAME LOOKUP] DB error owner_id={owner_id}: {e}")
    finally:
        if conn:
            try: conn.close()
            except Exception: pass
    fallback = label_map.get(owner_id)
    if fallback:
        with _name_cache_lock:
            _name_cache[owner_id] = fallback
        return fallback
    generic = f"Owner_{owner_id}"
    with _name_cache_lock:
        _name_cache[owner_id] = generic
    return generic

def _invalidate_name_cache():
    with _name_cache_lock:
        _name_cache.clear()

# ================================================================
#  LBPH RECOGNIZER
# ================================================================
_recognizer_lock = threading.RLock()
recognizer = None
label_map  = {}

def _load_label_map():
    global label_map
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
        print("✅ Label map:", label_map)
    else:
        label_map = {}

def _load_recognizer():
    global recognizer
    if os.path.exists(MODEL_PATH):
        r = cv.face.LBPHFaceRecognizer_create()
        r.read(MODEL_PATH)
        with _recognizer_lock:
            recognizer = r
        print("✅ LBPH model loaded")
    else:
        with _recognizer_lock:
            recognizer = None
        print("⚠  No classifier.xml — face recognition disabled until trained")
    _load_label_map()
    _invalidate_name_cache()


# ================================================================
#  CORE VOTING ENGINE — v6
#
#  CHANGES FROM v5:
#  1. Crops now use center-crop + 4 corner-shifted crops (not % offsets
#     that could go negative or zero-size for small ROIs).
#  2. normalize_face() replaces _preprocess_face() — unified pipeline.
#  3. Margin check unchanged but threshold lowered to 10 (was 12).
#  4. Returns (best_mean_dist, best_label, is_confident).
# ================================================================
def _predict_face_v6(rec, gray_roi):
    """
    Returns: (best_mean_dist, best_label, is_confident)
    Uses normalize_face() — same preprocessing as training.
    """
    h, w = gray_roi.shape[:2]
    if h < 20 or w < 20:
        return 9999.0, -1, False

    # Build 5 crops: center + 4 directional shifts (8% of size)
    crops = []
    shift_h = max(1, int(h * 0.08))
    shift_w = max(1, int(w * 0.08))
    offsets = [(0, 0), (-shift_h, 0), (shift_h, 0), (0, -shift_w), (0, shift_w)]
    for dy, dx in offsets:
        y1 = max(0, dy if dy > 0 else 0)
        y2 = min(h, h + dy if dy < 0 else h)
        x1 = max(0, dx if dx > 0 else 0)
        x2 = min(w, w + dx if dx < 0 else w)
        if (y2 - y1) >= 20 and (x2 - x1) >= 20:
            crops.append(gray_roi[y1:y2, x1:x2])
        else:
            crops.append(gray_roi)  # fallback to full ROI

    # Predict all crops using unified normalize_face()
    all_preds = []
    for crop in crops:
        try:
            processed = normalize_face(crop)
            lbl, dist = rec.predict(processed)
            all_preds.append((int(lbl), float(dist)))
        except Exception as e:
            print(f"   [LBPH predict error] {e}")

    if not all_preds:
        return 9999.0, -1, False

    # Group by label → vote count + mean distance
    label_dists = defaultdict(list)
    for lbl, dist in all_preds:
        label_dists[lbl].append(dist)

    scored = sorted(
        [(lbl, len(dists), float(np.mean(dists))) for lbl, dists in label_dists.items()],
        key=lambda x: (-x[1], x[2])
    )

    best_lbl, best_votes, best_mean = scored[0]
    print(f"   [LBPH v6] candidates={[(s[0], s[1], f'{s[2]:.1f}') for s in scored]}")

    # Criterion 1: distance tight enough
    if best_mean > KNOWN_MAX:
        print(f"   [LBPH v6] ❌ dist {best_mean:.1f} > KNOWN_MAX {KNOWN_MAX}")
        return best_mean, best_lbl, False

    # Criterion 2: majority vote
    if best_votes < KNOWN_VOTE_MIN:
        print(f"   [LBPH v6] ❌ votes {best_votes} < KNOWN_VOTE_MIN {KNOWN_VOTE_MIN}")
        return best_mean, best_lbl, False

    # Criterion 3: margin over runner-up
    if len(scored) > 1:
        _, _, runner_mean = scored[1]
        margin = runner_mean - best_mean
        if margin < MIN_MARGIN:
            print(f"   [LBPH v6] ❌ margin {margin:.1f} < MIN_MARGIN {MIN_MARGIN} "
                  f"(winner {best_mean:.1f} vs runner {runner_mean:.1f})")
            return best_mean, best_lbl, False

    print(f"   [LBPH v6] ✅ CONFIDENT: lbl={best_lbl} mean={best_mean:.1f} votes={best_votes}")
    return best_mean, best_lbl, True


def _auto_train_on_startup():
    def _do_train():
        time.sleep(2)
        print("🔄 Auto-training LBPH from existing samples…")
        faces, ids, trained_ids, new_label_map = [], [], set(), {}

        def _scan(directory):
            for entry in os.scandir(directory):
                if entry.is_dir():
                    _scan(entry.path)
                elif entry.name.lower().endswith(".jpg"):
                    parts = entry.name.split(".")
                    if len(parts) < 3:
                        return
                    try:
                        pid = int(parts[1])
                    except ValueError:
                        pid = abs(hash(parts[1])) % 100_000
                    img = cv.imread(entry.path, cv.IMREAD_GRAYSCALE)
                    if img is not None:
                        # FIX: use normalize_face() — same as prediction
                        faces.append(normalize_face(img))
                        ids.append(pid)
                        trained_ids.add(pid)
                        if pid not in new_label_map:
                            new_label_map[pid] = _get_owner_name(pid)
        try:
            _scan(SAMPLES_DIR)
        except Exception as e:
            print(f"⚠  Auto-train scan error: {e}"); return

        if not faces:
            print("ℹ️  Auto-train: no samples found, skipping."); return

        try:
            clf = cv.face.LBPHFaceRecognizer_create()
            clf.train(faces, np.array(ids))
            clf.write(MODEL_PATH)
            with open(LABEL_MAP_PATH, "w") as f:
                json.dump({str(k): v for k, v in new_label_map.items()}, f, indent=2)
            _load_recognizer()
            print(f"✅ Auto-train: {len(faces)} samples, owners={new_label_map}")
        except Exception as e:
            print(f"❌ Auto-train failed: {e}")

    threading.Thread(target=_do_train, daemon=True).start()


_load_recognizer()
_auto_train_on_startup()


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except Exception: return "127.0.0.1"
    finally: s.close()

SERVER_IP = get_local_ip()

# ================================================================
#  FLASK APP
# ================================================================
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = "ai-security-fixed-key-2024"
app.config.update(
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_DOMAIN=None,
    MAX_CONTENT_LENGTH=64 * 1024 * 1024,
)

def _db():
    return pymysql.connect(**DB_CONFIG)

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        valid = f"{USERNAME}:{PASSWORD}"
        if "user" in session:
            return f(*args, **kwargs)
        if request.headers.get("X-Auth-Token", "") == valid:
            return f(*args, **kwargs)
        is_api = (request.path.startswith("/api/") or request.path in ("/video_feed",))
        if is_api and request.args.get("token", "") == valid:
            return f(*args, **kwargs)
        if request.path.startswith("/api/"):
            return jsonify({"error": "unauthorized"}), 401
        return redirect(url_for("login_page"))
    return decorated

# ================================================================
#  DETECTION HELPERS
# ================================================================

def _yolo_detect(jpeg):
    """
    FIXED v6:
    1. Run YOLO at conf=YOLO_RUN_CONF (0.25) not 0.10 — reduces false positives.
    2. Collect ALL weapon/mask alerts, not just first one.
    3. Use _weapon_match() / _mask_match() for exact-token matching.
    4. Label list built in same order as detections (no index mismatch).
    5. Scissors/nail-cutter threshold lowered to 0.40 (was 0.60 — missed real ones).
    6. Returns all_alert_reasons list so caller can pick the most critical.
    """
    if yolo_model is None:
        return jpeg, None
    try:
        arr   = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv.imdecode(arr, cv.IMREAD_COLOR)
        if frame is None:
            return jpeg, None

        results    = yolo_model(frame, verbose=False,
                                conf=YOLO_RUN_CONF, iou=0.45, imgsz=640)[0]
        detections = sv.Detections.from_ultralytics(results)

        alert_reasons  = []   # FIX: collect ALL alerts, pick most critical
        display_labels = []   # one label per detection, same order as detections

        if len(results.boxes) > 0:
            det_list = [(yolo_model.names[int(b.cls)], float(b.conf)) for b in results.boxes]
            print(f"[YOLO] {len(det_list)} detections: " +
                  ", ".join(f"{n}={c:.2f}" for n, c in det_list))

        for box in results.boxes:
            cls_name  = yolo_model.names[int(box.cls)]
            cls_lower = cls_name.lower()
            conf      = float(box.conf)

            if _weapon_match(cls_lower):
                # Per-class threshold selection
                if "scissors" in cls_lower or "nail" in cls_lower or "cutter" in cls_lower:
                    threshold    = SCISSORS_CONF
                    display_name = "NAIL CUTTER / SCISSORS"
                elif "baseball bat" in cls_lower or ("bat" == cls_lower):
                    threshold    = WEAPON_CONF_DEFAULT
                    display_name = "BAT / BLUNT WEAPON"
                elif "fork" == cls_lower:
                    threshold    = WEAPON_CONF_DEFAULT
                    display_name = "FORK / SHARP OBJECT"
                else:
                    threshold    = WEAPON_CONF_DEFAULT
                    display_name = cls_name.upper()

                if conf >= threshold:
                    display_labels.append(f"⚠ {display_name} {conf:.0%}")
                    alert_reasons.append(f"WEAPON DETECTED: {display_name} ({conf:.0%})")
                    print(f"   [YOLO] 🔴 WEAPON: {display_name} conf={conf:.2f}")
                else:
                    display_labels.append(f"{cls_name} {conf:.0%} (low conf)")
                    print(f"   [YOLO] ⚠ weapon low-conf: {cls_name}={conf:.2f} < {threshold}")

            elif _mask_match(cls_lower) and conf >= MASK_CONF:
                display_labels.append(f"🎭 MASK {conf:.0%}")
                alert_reasons.append(f"MASKED PERSON DETECTED ({conf:.0%})")
                print(f"   [YOLO] 🎭 MASK conf={conf:.2f}")

            else:
                if cls_lower == "person" and conf >= PERSON_CONF:
                    display_labels.append(f"👤 PERSON {conf:.0%}")
                else:
                    display_labels.append(f"{cls_name} {conf:.0%}")

        # Annotate frame
        if len(detections) > 0 and len(display_labels) == len(results.boxes):
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = lbl_annotator.annotate(scene=frame, detections=detections,
                                           labels=display_labels)

        _, buf = cv.imencode(".jpg", frame)

        # Pick most critical alert: weapon > mask > None
        final_alert = None
        if alert_reasons:
            weapon_alerts = [r for r in alert_reasons if r.startswith("WEAPON")]
            if weapon_alerts:
                final_alert = " + ".join(weapon_alerts)
            else:
                final_alert = alert_reasons[0]

        return buf.tobytes(), final_alert

    except Exception as e:
        print(f"[YOLO ERROR] {e}")
        return jpeg, None


def _check_face_known(jpeg):
    """
    v6 — uses _predict_face_v6() with unified normalize_face().
    No more for_dark preprocessing split between train and predict.

    Returns: (is_known: bool, name: str, dist: float)
    """
    with _recognizer_lock:
        rec = recognizer

    if rec is None:
        return True, "NO_MODEL", 0.0

    try:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            return True, "DECODE_ERROR", 0.0

        h_img, w_img = img.shape[:2]
        frame_area   = h_img * w_img
        gray         = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        brightness   = float(np.mean(gray))
        is_dark      = brightness < 80   # only truly dark frames

        # Face detection — try normal, then dark-enhanced
        faces = _safe_detect_faces(gray, is_dark=False)
        if len(faces) == 0 and is_dark:
            faces = _safe_detect_faces(gray, is_dark=True)
        if len(faces) == 0:
            return True, "NO_FACE", 0.0

        for (x, y, w, h) in faces:
            face_area = w * h
            if face_area < frame_area * 0.003:
                continue

            x1 = max(0, x);  y1 = max(0, y)
            x2 = min(w_img, x + w);  y2 = min(h_img, y + h)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            roi = gray[y1:y2, x1:x2]

            # v6: single predict call, unified preprocessing
            best_dist, best_lbl, is_confident = _predict_face_v6(rec, roi)

            print(f"   [LBPH v6] final → dist={best_dist:.1f} lbl={best_lbl} "
                  f"confident={is_confident} brightness={brightness:.0f}")

            if is_confident and best_lbl >= 0:
                name = _get_owner_name(best_lbl)
                print(f"   [LBPH v6] ✅ KNOWN: {name} (dist={best_dist:.1f}, id={best_lbl})")
                return True, name, best_dist

            print(f"   [LBPH v6] ❌ UNKNOWN (dist={best_dist:.1f}, not confident)")
            return False, "UNKNOWN", best_dist

        return True, "NO_VALID_FACE", 0.0

    except Exception as e:
        print(f"⚠  _check_face_known error: {e}")
        return True, "ERROR", 0.0


def _check_dark_clothing(jpeg):
    try:
        arr   = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv.imdecode(arr, cv.IMREAD_COLOR)
        if frame is None:
            return False, jpeg
        h_fr, w_fr = frame.shape[:2]
        gray        = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces       = _safe_detect_faces(gray)
        flagged     = False
        for (fx, fy, fw, fh) in faces:
            ty1 = fy + fh
            ty2 = min(fy + fh * 4, h_fr)
            tx1 = max(fx - fw // 2, 0)
            tx2 = min(fx + fw + fw // 2, w_fr)
            torso = frame[ty1:ty2, tx1:tx2]
            if torso.size == 0:
                continue
            hsv        = cv.cvtColor(torso, cv.COLOR_BGR2HSV)
            dark_ratio = float(np.sum(hsv[:, :, 2] < 60)) / hsv[:, :, 2].size
            if dark_ratio > DARK_PIXEL_RATIO:
                flagged = True
                cv.rectangle(frame, (tx1, ty1), (tx2, ty2), (128, 0, 128), 2)
                cv.putText(frame, "SUSPICIOUS CLOTHING",
                           (tx1, ty1 - 5), cv.FONT_HERSHEY_SIMPLEX,
                           0.65, (128, 0, 128), 2)
        _, buf = cv.imencode(".jpg", frame)
        return flagged, buf.tobytes()
    except Exception as e:
        print(f"[DARK CLOTHING ERROR] {e}")
        return False, jpeg


# ================================================================
#  CORE DETECTION PIPELINE
# ================================================================
def _full_detection_pipeline(jpeg):
    alert_reason  = None
    original_jpeg = jpeg

    annotated_jpeg, yolo_alert = _yolo_detect(jpeg)
    if yolo_alert:
        alert_reason = yolo_alert

    is_known, face_label, dist = _check_face_known(original_jpeg)
    if not is_known and alert_reason is None:
        alert_reason = f"UNKNOWN PERSON DETECTED (dist={dist:.0f})"

    clothing_flagged, final_jpeg = _check_dark_clothing(annotated_jpeg)
    if clothing_flagged and alert_reason is None:
        alert_reason = "SUSPICIOUS PERSON (DARK CLOTHING)"

    return final_jpeg, alert_reason


# ================================================================
#  FRAME INGESTION
# ================================================================
def _ingest_frame(jpeg):
    global latest_frame, latest_raw_frame, frame_counter
    if not jpeg or len(jpeg) < 4 or jpeg[:2] != b'\xff\xd8':
        return
    with frame_lock:
        latest_frame     = jpeg
        latest_raw_frame = jpeg
        frame_buffer.append((time.time(), jpeg))
        frame_counter += 1
        run_detect = (frame_counter % DETECT_EVERY_N == 0)
    _new_frame_event.set()
    if run_detect:
        threading.Thread(target=_detect_and_alert, args=(jpeg,), daemon=True).start()


def _detect_and_alert(jpeg):
    global last_alert_ts, _consec_count
    now = datetime.now()
    if last_alert_ts and (now - last_alert_ts).total_seconds() < ALERT_COOLDOWN:
        return
    annotated_jpeg, alert_reason = _full_detection_pipeline(jpeg)
    if annotated_jpeg != jpeg:
        with frame_lock:
            latest_frame = annotated_jpeg
    if alert_reason is None:
        with _consec_lock:
            _consec_count = 0
        return
    with _consec_lock:
        _consec_count += 1
        count = _consec_count
    if count < CONSECUTIVE_NEEDED:
        return
    now = datetime.now()
    if last_alert_ts and (now - last_alert_ts).total_seconds() < ALERT_COOLDOWN:
        with _consec_lock: _consec_count = 0
        return
    with _consec_lock:
        _consec_count = 0
    last_alert_ts = now
    print(f"🚨 ALERT: {alert_reason} at {now.strftime('%H:%M:%S')}")
    ts_str   = now.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(ALERT_DIR, f"alert_{ts_str}.jpg")
    with open(img_path, "wb") as fh:
        fh.write(annotated_jpeg)
    threading.Thread(target=_play_alarm, daemon=True).start()
    threading.Thread(target=_build_video_and_email,
                     args=(img_path, now, alert_reason), daemon=True).start()


# ================================================================
#  MJPEG STREAM
# ================================================================
_blank_frame: bytes = b""

def _build_blank_frame():
    global _blank_frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv.putText(blank, "Waiting for camera...", (120, 230),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    cv.putText(blank, "Start stream on your phone", (100, 270),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1)
    _, buf = cv.imencode(".jpg", blank, [cv.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
    _blank_frame = buf.tobytes()

_build_blank_frame()

def _mjpeg_generator():
    last_sent_ts = 0.0
    while True:
        _new_frame_event.wait(timeout=0.5)
        _new_frame_event.clear()
        now = time.monotonic()
        gap = MIN_FRAME_GAP - (now - last_sent_ts)
        if gap > 0:
            time.sleep(gap)
        with frame_lock:
            frame = latest_frame
            if frame_buffer:
                if time.time() - frame_buffer[-1][0] > STALE_THRESHOLD:
                    frame = None
            else:
                frame = None
        payload      = frame if frame is not None else _blank_frame
        last_sent_ts = time.monotonic()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" +
            payload + b"\r\n"
        )

# ================================================================
#  ALARM
# ================================================================
def _play_alarm():
    global alarm_active
    if alarm_active:
        return
    alarm_active = True
    try:
        if not os.path.exists(ALARM_PATH):
            print("[WARN] alarm.wav not found")
            return
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            time.sleep(30)
            winsound.PlaySound(None, winsound.SND_PURGE)
        else:
            import subprocess
            p = subprocess.Popen(["aplay", ALARM_PATH])
            time.sleep(30)
            p.terminate()
    except Exception as e:
        print(f"[ALARM ERROR] {e}")
    finally:
        alarm_active = False

# ================================================================
#  VIDEO CLIP + EMAIL
# ================================================================
def _build_video_and_email(img_path, now, reason="SECURITY ALERT"):
    video_path = _save_video_clip(now)
    _send_email(img_path, video_path, now, reason)

def _save_video_clip(now):
    ts     = now.strftime("%Y%m%d_%H%M%S")
    path   = os.path.join(ALERT_DIR, f"video_{ts}.mp4")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out    = None
    with frame_lock:
        buffered = list(frame_buffer)
    for _, jpeg in buffered:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None: continue
        img = cv.resize(img, (640, 480))
        if out is None:
            out = cv.VideoWriter(path, fourcc, 10.0, (640, 480))
        out.write(img)
    deadline = time.time() + 8.0
    while time.time() < deadline:
        with frame_lock:
            jpeg = latest_raw_frame
        if jpeg:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv.imdecode(arr, cv.IMREAD_COLOR)
            if img is not None:
                img = cv.resize(img, (640, 480))
                if out is None:
                    out = cv.VideoWriter(path, fourcc, 10.0, (640, 480))
                out.write(img)
        time.sleep(0.1)
    if out: out.release()
    return path

def _send_email(img_path, video_path, now, reason="SECURITY ALERT"):
    conn = None
    try:
        conn = _db()
        cur  = conn.cursor()
        cur.execute("SELECT email FROM owner_details WHERE email IS NOT NULL AND email != ''")
        recipients = [r[0] for r in cur.fetchall()]
    except Exception as e:
        print(f"[EMAIL DB ERROR] {e}"); recipients = []
    finally:
        if conn:
            try: conn.close()
            except Exception: pass
    if not recipients:
        return
    token = f"{USERNAME}:{PASSWORD}"
    msg   = EmailMessage()
    msg["Subject"] = f"🚨 AI SECURITY ALERT: {reason}"
    msg["From"]    = EMAIL_FROM
    msg["To"]      = ", ".join(recipients)
    msg.set_content(f"""\
🚨 ALERT TYPE : {reason}
📅 DATE/TIME  : {now.strftime('%d-%m-%Y %H:%M:%S')}

🎥 Live stream  : {NGROK_LINK}/api/stream/feed?token={token}
🌐 Dashboard    : {NGROK_LINK}/home
📸 Photos       : {NGROK_LINK}/api/alerts/photos?token={token}
🎬 Videos       : {NGROK_LINK}/api/alerts/videos?token={token}

Local: http://{SERVER_IP}:5000/home
""")
    if img_path and os.path.exists(img_path):
        with open(img_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                               filename="alert_snapshot.jpg")
    if video_path and os.path.exists(video_path):
        mb = os.path.getsize(video_path) / (1024 * 1024)
        if mb <= MAX_VIDEO_MB:
            with open(video_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="video", subtype="mp4",
                                   filename="alert_video.mp4")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(EMAIL_FROM, EMAIL_PASS)
            srv.send_message(msg)
        print(f"✅ Email sent to {recipients}  ({reason})")
    except Exception as e:
        print(f"❌ Email error: {e}")

# ================================================================
#  MISC HELPERS
# ================================================================
def _parse_ts_from_filename(filename):
    m = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', filename)
    if m:
        Y, Mo, D, H, Mi, S = m.groups()
        return f"{Y}-{Mo}-{D} {H}:{Mi}:{S}"
    return filename

def _video_duration_seconds(path):
    try:
        cap = cv.VideoCapture(path)
        fps = cap.get(cv.CAP_PROP_FPS) or 10
        fc  = cap.get(cv.CAP_PROP_FRAME_COUNT)
        cap.release()
        return max(0, int(fc / fps))
    except Exception: return 0

def _fmt_duration(s):
    return f"{s // 60:02d}:{s % 60:02d}"

def _read_jpeg_from_request():
    ct = request.content_type or ""
    if "image/jpeg" in ct and request.data:
        return request.data
    for field in ("frame", "image", "snapshot"):
        if field in request.files:
            return request.files[field].read()
    for field in ("frame", "image", "snapshot"):
        if field in request.form:
            try: return base64.b64decode(request.form[field])
            except Exception: pass
    return None

def _send_file_partial(path, mimetype):
    file_size = os.path.getsize(path)
    rh = request.headers.get("Range")
    if rh:
        m = re.match(r'bytes=(\d+)-(\d*)', rh)
        if m:
            start  = int(m.group(1))
            end    = int(m.group(2)) if m.group(2) else file_size - 1
            end    = min(end, file_size - 1)
            length = end - start + 1
            def gen():
                with open(path, "rb") as fh:
                    fh.seek(start); rem = length
                    while rem > 0:
                        d = fh.read(min(65536, rem))
                        if not d: break
                        rem -= len(d); yield d
            resp = Response(gen(), 206, mimetype=mimetype, direct_passthrough=True)
            resp.headers.update({
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(length),
            })
            return resp
    resp = Response(open(path, "rb").read(), 200, mimetype=mimetype)
    resp.headers.update({"Accept-Ranges": "bytes", "Content-Length": str(file_size)})
    return resp

# ================================================================
#  FLASK ROUTES — Auth
# ================================================================
@app.route("/", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if u.lower() == USERNAME and p == PASSWORD:
            session["user"] = u
            return redirect(url_for("home_page"))
        return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/home")
@require_auth
def home_page():
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

# ================================================================
#  FLASK ROUTES — Stream
# ================================================================
@app.route("/video_feed")
@require_auth
def video_feed():
    resp = Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Cache-Control"]     = "no-cache, no-store"
    return resp

@app.route("/api/stream/feed")
@require_auth
def api_stream_feed():
    resp = Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Cache-Control"]     = "no-cache, no-store"
    return resp

# ================================================================
#  FLASK ROUTES — Frame ingestion
# ================================================================
@app.route("/api/frame", methods=["POST"])
@require_auth
def api_receive_frame():
    jpeg = _read_jpeg_from_request()
    if jpeg:
        _ingest_frame(jpeg)
    return "", 204

@app.route("/api/stream/push", methods=["POST"])
@require_auth
def api_stream_push():
    jpeg = _read_jpeg_from_request()
    if jpeg:
        _ingest_frame(jpeg)
    return "", 204

# ================================================================
#  FLASK ROUTES — Alert from phone
# ================================================================
@app.route("/api/alert", methods=["POST"])
@require_auth
def api_receive_alert():
    global last_alert_ts
    now  = datetime.now()
    jpeg = _read_jpeg_from_request()

    if not jpeg:
        return jsonify({"success": False, "message": "No image"}), 400

    annotated_jpeg, yolo_alert = _yolo_detect(jpeg)
    is_known, face_label, dist = _check_face_known(jpeg)

    alert_reason = None
    if yolo_alert:
        alert_reason = yolo_alert
    if not is_known and alert_reason is None:
        alert_reason = f"UNKNOWN PERSON DETECTED (dist={dist:.0f})"

    if is_known and alert_reason is None:
        print(f"ℹ️  KNOWN ({face_label}, dist={dist:.0f}) — suppressed")
        return jsonify({
            "success": True,
            "known":   True,
            "label":   face_label,
            "dist":    round(dist, 1),
        }), 200

    if last_alert_ts and (now - last_alert_ts).total_seconds() < ALERT_COOLDOWN:
        return jsonify({
            "success": True,
            "skipped": True,
            "known":   is_known,
            "label":   face_label if is_known else "UNKNOWN",
            "dist":    round(dist, 1),
        }), 200

    last_alert_ts = now
    img_path = None
    if annotated_jpeg:
        ts_str   = now.strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(ALERT_DIR, f"phone_alert_{ts_str}.jpg")
        with open(img_path, "wb") as fh:
            fh.write(annotated_jpeg)

    reason = alert_reason or "UNKNOWN PERSON (phone-triggered)"
    print(f"🚨 Phone alert — {reason} at {now.strftime('%H:%M:%S')}")
    threading.Thread(target=_play_alarm, daemon=True).start()
    threading.Thread(target=_build_video_and_email,
                     args=(img_path, now, reason), daemon=True).start()

    return jsonify({
        "success": True,
        "known":   False,
        "label":   "UNKNOWN",
        "dist":    round(dist, 1),
    }), 200

# ================================================================
#  FLASK ROUTES — Status / Login
# ================================================================
@app.route("/api/login", methods=["POST"])
def api_login():
    d = request.get_json(silent=True) or {}
    if d.get("username", "").lower() == USERNAME and d.get("password", "") == PASSWORD:
        return jsonify({"success": True, "token": f"{USERNAME}:{PASSWORD}",
                        "server_ip": SERVER_IP, "ngrok": NGROK_LINK})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route("/api/status")
def api_status():
    with frame_lock:
        has_frame = latest_frame is not None
    photo_count = sum(1 for f in os.listdir(ALERT_DIR) if f.lower().endswith(".jpg"))
    video_count = sum(1 for f in os.listdir(ALERT_DIR) if f.lower().endswith((".mp4", ".avi")))
    return jsonify({
        "server": "online", "alarm": alarm_active, "stream": has_frame,
        "model_trained": os.path.exists(MODEL_PATH),
        "yolo_loaded": yolo_model is not None,
        "server_ip": SERVER_IP, "ngrok": NGROK_LINK,
        "alert_photos": photo_count, "alert_videos": video_count,
    })

# ================================================================
#  FLASK ROUTES — Alert media
# ================================================================
@app.route("/api/alerts/photos")
@require_auth
def api_list_photos():
    files = [{"filename": f, "timestamp": _parse_ts_from_filename(f),
               "url": f"/api/alerts/photos/{f}"}
             for f in sorted(os.listdir(ALERT_DIR), reverse=True)
             if f.lower().endswith(".jpg")]
    return jsonify({"photos": files, "count": len(files)})

@app.route("/api/alerts/photos/<path:filename>")
@require_auth
def api_serve_photo(filename):
    path = os.path.join(ALERT_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/jpeg")

@app.route("/api/alerts/photos/all", methods=["DELETE"])
@require_auth
def api_delete_all_photos():
    deleted = 0
    for f in os.listdir(ALERT_DIR):
        if f.lower().endswith(".jpg"):
            os.remove(os.path.join(ALERT_DIR, f)); deleted += 1
    return jsonify({"success": True, "deleted": deleted})

@app.route("/api/alerts/photos/<path:filename>", methods=["DELETE"])
@require_auth
def api_delete_photo(filename):
    path = os.path.join(ALERT_DIR, os.path.basename(filename))
    if os.path.exists(path): os.remove(path)
    return jsonify({"success": True})

@app.route("/api/alerts/videos")
@require_auth
def api_list_videos():
    files = []
    for f in sorted(os.listdir(ALERT_DIR), reverse=True):
        if f.lower().endswith((".mp4", ".avi")):
            p = os.path.join(ALERT_DIR, f)
            files.append({"filename": f, "timestamp": _parse_ts_from_filename(f),
                          "url": f"/api/alerts/videos/{f}",
                          "size_mb": str(round(os.path.getsize(p)/(1024*1024), 1)),
                          "duration": _fmt_duration(_video_duration_seconds(p))})
    return jsonify({"videos": files, "count": len(files)})

@app.route("/api/alerts/videos/upload", methods=["POST"])
@require_auth
def api_upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "message": "No video file"}), 400
    file     = request.files["video"]
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith((".mp4", ".avi")):
        return jsonify({"success": False, "message": "Only .mp4 or .avi accepted"}), 400
    file.seek(0, 2); size_mb = file.tell()/(1024*1024); file.seek(0)
    if size_mb > MAX_VIDEO_MB:
        return jsonify({"success": False,
                        "message": f"File too large ({size_mb:.1f} MB, max {MAX_VIDEO_MB} MB)"}), 413
    save_path = os.path.join(ALERT_DIR, filename)
    if os.path.exists(save_path):
        base, ext = os.path.splitext(filename)
        save_path = os.path.join(ALERT_DIR, f"{base}_dup{ext}")
    try:
        file.save(save_path)
        return jsonify({"success": True, "filename": os.path.basename(save_path),
                        "size_mb": round(size_mb, 1)})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/alerts/videos/<path:filename>")
@require_auth
def api_serve_video(filename):
    path = os.path.join(ALERT_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    mime = "video/mp4" if filename.lower().endswith(".mp4") else "video/x-msvideo"
    return _send_file_partial(path, mime)

@app.route("/api/alerts/videos/all", methods=["DELETE"])
@require_auth
def api_delete_all_videos():
    deleted = 0
    for f in os.listdir(ALERT_DIR):
        if f.lower().endswith((".mp4", ".avi")):
            os.remove(os.path.join(ALERT_DIR, f)); deleted += 1
    return jsonify({"success": True, "deleted": deleted})

@app.route("/api/alerts/videos/<path:filename>", methods=["DELETE"])
@require_auth
def api_delete_video(filename):
    path = os.path.join(ALERT_DIR, os.path.basename(filename))
    if os.path.exists(path): os.remove(path)
    return jsonify({"success": True})

# ================================================================
#  FLASK ROUTES — Owner registration + face capture + training
# ================================================================
@app.route("/api/register", methods=["POST"])
@require_auth
def api_register():
    d     = request.get_json(silent=True) or {}
    name  = d.get("name",  "").strip()
    phone = d.get("phone", "").strip()
    email = d.get("email", "").strip()
    if not all([name, phone, email]):
        return jsonify({"success": False, "message": "name, phone, email required"}), 400
    conn = None
    try:
        conn = _db(); cur = conn.cursor()
        cur.execute("SELECT owner_id FROM owner_details WHERE phone = %s", (phone,))
        row = cur.fetchone()
        if row:
            _invalidate_name_cache()
            return jsonify({"success": True, "owner_id": row[0], "message": "Already registered"})
        cur.execute(
            "INSERT INTO owner_details (name, phone, email, sample_status) VALUES (%s,%s,%s,'No')",
            (name, phone, email))
        conn.commit(); oid = cur.lastrowid
        _invalidate_name_cache()
        return jsonify({"success": True, "owner_id": oid})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/capture_sample", methods=["POST"])
@require_auth
def api_capture_sample():
    owner_id_raw = request.form.get("owner_id", "").strip()
    phone        = request.form.get("phone",    "").strip()
    count        = int(request.form.get("count", 0))
    raw = None
    for field in ("image", "frame", "snapshot"):
        if field in request.files:
            raw = request.files[field].read(); break
    if raw is None:
        raw = _read_jpeg_from_request()
    if not raw:
        return jsonify({"success": False, "message": "No image data"}), 400
    arr  = np.frombuffer(raw, np.uint8)
    gray = cv.imdecode(arr, cv.IMREAD_GRAYSCALE)
    if gray is None:
        return jsonify({"success": False, "message": "Image decode failed"}), 400

    brightness  = float(np.mean(gray))
    is_dark     = brightness < 80

    # Use same detection pipeline as prediction — consistent
    faces = _safe_detect_faces(gray, is_dark=False)
    if len(faces) == 0 and is_dark:
        faces = _safe_detect_faces(gray, is_dark=True)

    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        h_img, w_img = gray.shape[:2]
        x1 = max(0, x);  y1 = max(0, y)
        x2 = min(w_img, x + w);  y2 = min(h_img, y + h)
        face_to_save = gray[y1:y2, x1:x2]
        print(f"   [CAPTURE] Face {w}×{h} px — saving raw ROI")
    else:
        face_to_save = gray
        print(f"   [CAPTURE] No face detected — saving full frame as fallback")

    resolved_id = owner_id_raw
    safe_phone  = "".join(c for c in phone if c.isdigit() or c == "+")
    if not resolved_id or resolved_id in ("-1", "0", ""):
        if safe_phone:
            conn2 = None
            try:
                conn2 = _db(); cur2 = conn2.cursor()
                cur2.execute("SELECT owner_id FROM owner_details WHERE phone = %s", (safe_phone,))
                r = cur2.fetchone()
                resolved_id = str(r[0]) if r else "0"
            except Exception as e:
                print(f"⚠  Phone→owner_id lookup failed: {e}"); resolved_id = "0"
            finally:
                if conn2:
                    try: conn2.close()
                    except Exception: pass
    folder = os.path.join(SAMPLES_DIR, safe_phone) if safe_phone else SAMPLES_DIR
    os.makedirs(folder, exist_ok=True)
    fn        = f"user.{resolved_id}.{count}.jpg"
    save_path = os.path.join(folder, fn)
    if not cv.imwrite(save_path, face_to_save):
        return jsonify({"success": False, "message": "Failed to write image"}), 500
    return jsonify({"success": True, "saved": fn, "count": count + 1})


@app.route("/api/train", methods=["POST"])
@require_auth
def api_train():
    faces, ids    = [], []
    trained_ids   = set()
    new_label_map = {}

    def _scan(directory):
        for entry in os.scandir(directory):
            if entry.is_dir():
                _scan(entry.path)
            elif entry.name.lower().endswith(".jpg"):
                parts = entry.name.split(".")
                if len(parts) < 3: return
                try: pid = int(parts[1])
                except ValueError: pid = abs(hash(parts[1])) % 100_000
                img = cv.imread(entry.path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    # FIX: use normalize_face() — unified preprocessing
                    faces.append(normalize_face(img))
                    ids.append(pid); trained_ids.add(pid)
                    if pid not in new_label_map:
                        new_label_map[pid] = _get_owner_name(pid)

    _scan(SAMPLES_DIR)
    if not faces:
        return jsonify({"success": False,
                        "message": "No samples found — capture face samples first"}), 400

    clf = cv.face.LBPHFaceRecognizer_create()
    clf.train(faces, np.array(ids))
    clf.write(MODEL_PATH)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump({str(k): v for k, v in new_label_map.items()}, f, indent=2)
    print(f"✅ Trained: {len(faces)} samples, owners={new_label_map}")

    if trained_ids:
        conn = None
        try:
            conn = _db(); cur = conn.cursor()
            ph = ",".join(["%s"] * len(trained_ids))
            cur.execute(
                f"UPDATE owner_details SET sample_status='Yes' WHERE owner_id IN ({ph})",
                tuple(trained_ids))
            conn.commit()
        except Exception as e: print(f"⚠  UPDATE sample_status failed: {e}")
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    _load_recognizer()
    return jsonify({"success": True,
                    "message": f"Trained {len(faces)} samples / {len(trained_ids)} owner(s)",
                    "owners": new_label_map})

@app.route("/api/model/classifier.xml")
@require_auth
def serve_model():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "No model yet"}), 404
    return send_file(MODEL_PATH, mimetype="application/xml",
                     as_attachment=True, download_name="classifier.xml")

# ================================================================
#  FLASK ROUTES — Owners CRUD
# ================================================================
@app.route("/api/owners")
@require_auth
def api_owners():
    conn = None
    try:
        conn = _db(); cur = conn.cursor()
        cur.execute("SELECT owner_id, name, phone, email, sample_status FROM owner_details")
        rows = cur.fetchall()
        return jsonify({"success": True, "owners": [
            {"id": r[0], "name": r[1], "phone": r[2],
             "email": r[3], "address": "", "status": r[4]}
            for r in rows]})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        if conn:
            try: conn.close()
            except Exception: pass

@app.route("/api/owners/<int:owner_id>", methods=["DELETE"])
@require_auth
def api_delete_owner(owner_id):
    conn = None
    try:
        conn = _db(); cur = conn.cursor()
        cur.execute("SELECT phone FROM owner_details WHERE owner_id = %s", (owner_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({"success": False, "message": "Not found"}), 404
        safe = "".join(c for c in (row[0] or "") if c.isdigit() or c == "+")
        cur.execute("DELETE FROM owner_details WHERE owner_id = %s", (owner_id,))
        conn.commit()
        if safe:
            folder = os.path.join(SAMPLES_DIR, safe)
            if os.path.isdir(folder):
                import shutil; shutil.rmtree(folder)
        _invalidate_name_cache()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        if conn:
            try: conn.close()
            except Exception: pass

# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    print(f"\n{'='*62}")
    print(f"  AI SECURITY SYSTEM — server v6.0")
    print(f"{'='*62}")
    print(f"  Local  → http://{SERVER_IP}:5000")
    print(f"  Ngrok  → {NGROK_LINK}")
    print(f"  YOLO   → {'✅ loaded' if yolo_model else '❌ not loaded'}")
    print(f"  LBPH   → {'✅ loaded' if recognizer else '⚠  not loaded yet'}")
    print(f"  Alarm  → {'✅ found' if os.path.exists(ALARM_PATH) else '⚠  alarm.wav missing'}")
    print(f"{'='*62}")
    print(f"\n  v6.0 — fixes:")
    print(f"  • normalize_face() — single unified preprocessing")
    print(f"    (was: for_dark=True/False mismatch train vs predict)")
    print(f"  • bilateralFilter BEFORE CLAHE (was: after — wrong order)")
    print(f"  • CLAHE clipLimit fixed at 2.0 everywhere (was: 3.0/4.0)")
    print(f"  • DETECT_EVERY_N=1 — all frames checked (was: 2, skipped weapons)")
    print(f"  • YOLO_RUN_CONF=0.25 (was: 0.10 — too noisy)")
    print(f"  • SCISSORS_CONF=0.40 (was: 0.60 — missed nail cutters)")
    print(f"  • _weapon_match() exact-token match (was: substring → false matches)")
    print(f"  • All weapon alerts collected, not just first one")
    print(f"  ⚠  DELETE old samples, re-capture faces & retrain model!")
    print(f"{'='*62}\n")
    app.run(host="0.0.0.0", port=5000, debug=False,
            use_reloader=False, threaded=True)