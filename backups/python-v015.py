"""
run_simulation.py  v015  -  Pose JSON  ->  MS-Human-700 (MuJoCo)  ->  musculoskeletal_output.mp4

INPUT (pose_motion_data.json, written by index-v015.html; older files still work)
  * PoseNet  : 17 flat 2D points per frame (v013/v014 files, a bare list of frames or {"frames": [...]})
  * BlazePose: 33 landmarks per frame. Each has image pixels (position) AND "world" = metres with the
               origin at the hips. The world numbers give real depth, so the solver no longer goes blind
               when the person turns sideways, and shoulders, hips, wrists, ankles and the trunk twist
               can be measured in 3D.
  Both file types go through the same smoothing, coupling, foot-lock and muscle code below.

3D SOLVER IN ONE PARAGRAPH
  Trunk landmarks give a body frame per frame: F (forward), L (person's left), U (up). Limb angles are
  measured inside that frame, so turning the body does not change them. Gravity comes from the legs
  (ankles below hips) and "forward" is checked with the nose, so the solver does not depend on the sign
  conventions of the world axes. Body turn (pelvis_rotation) and trunk twist come from the yaw of the hip
  and shoulder lines.

MUSCLE REPORT (v015)
  Still kinematic only. Muscle length changes are real; "engagement" is an estimate, not a force.
  Per muscle and frame it combines shortening depth, shortening/lengthening speed and the model's own
  force-length-velocity capacity (MuJoCo's muscle curves from MS-Human-700). A small non-negative
  matrix factorisation then groups muscles that work together into synergies, in the spirit of the
  DynSyn muscle-synergy idea. Real activation still needs inverse dynamics + optimisation.
"""
import os
import sys
import csv
import json
import time
import subprocess

# Headless Linux (no display, e.g. Colab/servers) needs EGL for offscreen rendering.
# Must be set BEFORE importing mujoco. Harmless on Windows/macOS (skipped there).
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import mujoco

# =============================================================================
# SETTINGS
# =============================================================================
MODEL_DIR = "MS-Human-700"
XML_PATH = os.path.join(MODEL_DIR, "MS-Human-700.xml")
JSON_FILENAME = "pose_motion_data.json"
OUTPUT_VIDEO = "musculoskeletal_output.mp4"
OUTPUT_ANGLES_CSV = "joint_angles.csv"
OUTPUT_MUSCLES_CSV = "muscle_report.csv"
OUTPUT_SYNERGY_CSV = "muscle_synergies.csv"

WIDTH, HEIGHT = 1280, 720

# --- Timing ------------------------------------------------------------------
OUT_FPS = 30                       # video frame rate. The JSON is resampled onto this grid.
PLAYBACK_SPEED = 1.0               # 1.0 = real time, 0.5 = half speed slow motion

# --- Pose source (v015) ------------------------------------------------------
USE_DEPTH = True                   # use the 3D world coordinates when the JSON has them (BlazePose)
USE_MEASURED_TURN = True           # 3D: the model turns like the person (False = slow auto-spin as before)
TURN_ZERO_AT_START = False         # True: the person's first pose counts as "facing the camera"
WORLD_Z_AWAY = True                # BlazePose world z grows away from the camera. If the model turns the
                                   #   wrong way when you turn, set this to False
USE_WRIST = True                   # 3D: wrist flexion + deviation from the hand landmarks
USE_ANKLE = True                   # 3D: ankle dorsiflexion from heel + toe landmarks
USE_TRUNK_TWIST = True             # 3D: shoulders turning against the hips
ANKLE_NEUTRAL_DEG = 90.0           # angle between shank and foot (heel -> toe) when standing flat
WRIST_FLEX_GAIN = 0.5              # MS-Human-700 turns the hand twice per "flexion" joint value (checked)
WRIST_ULNAR_GAIN = 1.5             # ... and 1.5 times per ulnar-deviation joint value (radial is left as measured)

# "frontal"  : person faces the camera -> arms/legs spreading sideways (abduction)
# "sagittal" : person seen from the side -> arms/legs swinging forward (flexion)
# Only used for flat 2D (PoseNet) data. With 3D data both are measured at the same time.
MOTION_PLANE = "frontal"

TURN_SPEED_RAD_PER_SEC = 0.15      # slow spin about the VERTICAL (world Z) axis, per second of video
MIN_SCORE = 0.2                    # ignore keypoints below this confidence
SMOOTH_SEC = 0.15                  # smoothing width in SECONDS (0 = off, bigger = smoother)
HAMPEL_HALF_WINDOW = 3             # spike filter looks at +-3 samples
HAMPEL_N_SIGMA = 3.0               # a sample this many robust sigmas off the median is dropped

# --- Pelvis ------------------------------------------------------------------
USE_PELVIS_LIST = True             # hip-line tilt drives pelvis_list
USE_PELVIS_SHIFT = True            # left/right movement of the hips moves the model sideways
MAX_SHIFT_M = 1.0                  # clamp for the sideways movement (keeps the model in view)
# Checked against the model: a POSITIVE pelvis_list raises the person's LEFT hip.
# v014 gave it the opposite sign. -1 is correct; set +1 to get the v014 behaviour back.
PELVIS_LIST_SIGN = -1.0

# --- Muscles -----------------------------------------------------------------
SHOW_MUSCLES = True                # False hides the muscles so the bones are easier to see
COLOR_MUSCLES = True
COLOR_MODE = "length"              # "length": red = shortening, blue = lengthening (vs the standing pose)
                                   # "engagement": grey -> orange, the estimated engagement index
COLOR_FULL_SCALE = 0.20            # a 20% change of a muscle's working range = full colour
ENGAGE_COLOR_FULL = 0.50           # engagement index that gives full orange
SPEED_FULL_SCALE = 0.50            # shortening of 50% of the working range per second = "fast"
SYNERGY_COUNT = 4                  # how many muscle synergies to extract (0 = off)
DRAW_OVERLAY = True                # time + legend + HUD panels in the video
HUD_TOP_MUSCLES = 5                # muscles listed in the live "most engaged" panel

print("=== STARTING SIMULATION PIPELINE (v015) ===")

# =============================================================================
# 1. MODEL SETUP
# =============================================================================
if not os.path.exists(XML_PATH):
    print(f"[Model Setup] '{XML_PATH}' not found locally. Cloning repository from GitHub...")
    try:
        subprocess.run(["git", "clone", "https://github.com/LNSGroup/MS-Human-700.git", MODEL_DIR], check=True)
        print("[Model Setup] Clone completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to clone repository automatically: {e}")
else:
    print(f"[Model Setup] Found existing model file at: {XML_PATH}")

# =============================================================================
# 2. LOAD RECORDED JSON FROM THE BROWSER
# =============================================================================
print(f"[Data Loading] Looking for '{JSON_FILENAME}'...")
file_meta = {}
if os.path.exists(JSON_FILENAME):
    with open(JSON_FILENAME, "r") as f:
        motion_data = json.load(f)
    if isinstance(motion_data, dict):          # v015 files: {"model": ..., "frames": [...]}
        file_meta = {k: v for k, v in motion_data.items() if k != "frames"}
        motion_data = motion_data.get("frames", [])
    print(f"[Data Loading] Success! Loaded {len(motion_data)} frames from {JSON_FILENAME}.")
else:
    print(f"[Data Loading] WARNING: '{JSON_FILENAME}' was NOT found in this directory!")
    motion_data = [{"keypoints": [], "time": i * 0.1} for i in range(60)]

motion_data = sorted(motion_data, key=lambda fr: fr.get("time", 0.0))
frame_times = np.array([fr.get("time", i * 0.1) for i, fr in enumerate(motion_data)], dtype=float)
if len(frame_times) < 2:
    raise RuntimeError("Need at least 2 recorded frames.")
rec_dt = np.diff(frame_times)
print(f"[Data Loading] Recording: {frame_times[-1] - frame_times[0]:.2f} s, "
      f"average {1.0 / max(rec_dt.mean(), 1e-6):.1f} fps, "
      f"frame gaps {rec_dt.min() * 1000:.0f}-{rec_dt.max() * 1000:.0f} ms.")


def parse_keypoints(frame):
    """One frame -> {part: {x, y, score, w}}. x, y = image pixels. w (optional) = world metres (BlazePose).
    Works for PoseNet frames (no world) and BlazePose frames alike; low-confidence points are left out."""
    kp = {}
    for k in frame.get("keypoints", []):
        if "part" not in k or "position" not in k or k.get("score", 1.0) < MIN_SCORE:
            continue
        p = {"x": k["position"]["x"], "y": k["position"]["y"], "score": k.get("score", 1.0)}
        w = k.get("world")
        if isinstance(w, dict) and all(a in w for a in "xyz"):
            p["w"] = np.array([w["x"], w["y"], w["z"]], dtype=float)
        kp[k["part"]] = p
    return kp


all_kp = [parse_keypoints(fr) for fr in motion_data]
n_points = max((len(fr.get("keypoints", [])) for fr in motion_data), default=0)
frames_with_world = sum(1 for kp in all_kp if any("w" in p for p in kp.values()))
source_name = file_meta.get("model") or ("blazepose" if frames_with_world else "posenet")
print(f"[Data Loading] Pose source: {source_name}, {n_points} landmarks per frame, "
      f"{frames_with_world}/{len(motion_data)} frames carry 3D world coordinates"
      + (f", written by page {file_meta['version']}" if "version" in file_meta else "") + ".")

# =============================================================================
# 3. LOAD MUJOCO MODEL
# =============================================================================
print(f"[MuJoCo] Loading model configuration from {XML_PATH}...")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
print(f"[MuJoCo] Model loaded successfully! Total joints in model: {model.njnt}")


def joint_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def body_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def qadr(name):
    return model.jnt_qposadr[joint_id(name)]


# This model has NO free root joint. It is built like an OpenSim model:
# pelvis_tx/ty/tz (slides) + pelvis_tilt/list/rotation (hinges). In the world frame
#   pelvis_tx = forward (+X), pelvis_ty = UP (+Z), pelvis_tz = sideways (-Y),
#   pelvis_rotation spins about the vertical (+Z) axis (+ = counter-clockwise seen from above).
TURN_JOINT = joint_id("pelvis_rotation")
LIFT_JOINT = joint_id("pelvis_ty")
SIDE_JOINT = joint_id("pelvis_tz")
if min(TURN_JOINT, LIFT_JOINT, SIDE_JOINT) == -1:
    raise RuntimeError("Could not find pelvis_rotation / pelvis_ty / pelvis_tz in this model.")

FOOT_BODIES = [b for b in (body_id(n) for n in ("calcn_r", "calcn_l", "toes_r", "toes_l")) if b != -1]

# Joints measured from flat 2D data (PoseNet) ...
DRIVEN_JOINTS_2D = [
    "elbow_flexion_r", "elbow_flexion_l",
    "knee_angle_r", "knee_angle_l",
    "shoulder_elv_r", "shoulder_elv_l", "elv_angle_r", "elv_angle_l",
    "hip_flexion_r", "hip_flexion_l", "hip_adduction_r", "hip_adduction_l",
    "L5_S1_LB", "T12_L1_LB", "T1_head_neck_LB", "T1_head_neck_AR",
    "pelvis_list",
]
# ... and the extra ones that need depth (BlazePose)
DRIVEN_JOINTS_3D = [
    "flexion_r", "flexion_l", "deviation_r", "deviation_l",
    "ankle_angle_r", "ankle_angle_l",
    "L5_S1_AR", "T12_L1_AR",
]
for label, names in (("2D", DRIVEN_JOINTS_2D), ("3D extra", DRIVEN_JOINTS_3D)):
    missing = [n for n in names if joint_id(n) == -1]
    print(f"[Joint Mapping] {len(names) - len(missing)}/{len(names)} {label} joints found in the model."
          + (f" Missing: {missing}" if missing else ""))

# --- Coupled joints ------------------------------------------------------------
# The model ties 42 joints to other joints with polynomial equations: the knee's gliding and
# rotation, the kneecap, the collarbone/shoulder-blade rhythm when the arm is raised, the wrist.
# Setting qpos directly ignores these, so we evaluate the same polynomials ourselves.
COUPLINGS = []
for i in range(model.neq):
    if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[i] >= 0:
        j_dep, j_ind = model.eq_obj1id[i], model.eq_obj2id[i]
        COUPLINGS.append((int(model.jnt_qposadr[j_dep]), int(model.jnt_qposadr[j_ind]),
                          model.eq_data[i, :5].copy()))
print(f"[Coupling] {len(COUPLINGS)} joint-coupling equations will be applied each frame.")


def apply_couplings(q):
    """y - y0 = a0 + a1*(x - x0) + a2*(x - x0)^2 + ...  (same rule MuJoCo uses for joint equalities)."""
    for a_dep, a_ind, c in COUPLINGS:
        x = q[a_ind] - model.qpos0[a_ind]
        q[a_dep] = model.qpos0[a_dep] + np.polyval(c[::-1], x)


# Remember how high the feet are in the neutral standing pose, so squats keep feet on the floor
mujoco.mj_forward(model, data)
REST_FOOT_Z = min(data.xpos[b][2] for b in FOOT_BODIES)

# Pixel -> metre scale: a straight leg (hip to ankle) in the video vs the same length in the model
MODEL_LEG_M = float(np.mean([
    np.linalg.norm(data.xpos[body_id("femur_r")] - data.xpos[body_id("talus_r")]),
    np.linalg.norm(data.xpos[body_id("femur_l")] - data.xpos[body_id("talus_l")]),
]))


# =============================================================================
# 4. ANGLE MATH
# =============================================================================
def interior_angle(a, b, c):
    """2D: angle in degrees at point b between a-b-c (180 = straight line)."""
    ba = np.array([a["x"] - b["x"], a["y"] - b["y"]])
    bc = np.array([c["x"] - b["x"], c["y"] - b["y"]])
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def midpoint(a, b):
    return {"x": (a["x"] + b["x"]) / 2.0, "y": (a["y"] + b["y"]) / 2.0}


def wrap_half_pi(a):
    """A line has no direction, so wrap a roll angle into (-90, +90) degrees. Prevents 180-degree flips."""
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def conf(*pts):
    return min(p["score"] for p in pts)


# --- 3D helpers (world metres) -----------------------------------------------
WORLD_DOWN = np.array([0.0, 1.0, 0.0])    # gravity direction in the world numbers, found by calibrate_world()
FWD_SIGN = 1.0                            # +1 / -1, found by calibrate_world()
DEPTH_READY = False


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def angle_between(a, b):
    """Angle in degrees between two vectors."""
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d < 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / d, -1.0, 1.0))))


def interior_angle_3d(a, b, c):
    return angle_between(a - b, c - b)


def flat_dir(v, up):
    """Horizontal part of v as a unit vector (None if v is vertical)."""
    return unit(v - np.dot(v, up) * up)


def yaw_of(v, up):
    """Turn about the vertical axis from a (left minus right) vector.
    0 = facing the camera, + = turned toward the person's left (counter-clockwise seen from above)."""
    h = flat_dir(v, up)
    if h is None:
        return None
    return float(np.arctan2((1.0 if WORLD_Z_AWAY else -1.0) * h[2], h[0]))


def torso_frame(kp):
    """Body axes from shoulders + hips: (F forward, L left, U up), or None if a point is missing."""
    pts = [kp.get(n) for n in ("leftShoulder", "rightShoulder", "leftHip", "rightHip")]
    if any(p is None or "w" not in p for p in pts):
        return None
    lsh, rsh, lhp, rhp = [p["w"] for p in pts]
    L = unit((lsh - rsh) + (lhp - rhp))
    if L is None:
        return None
    top = (lsh + rsh) / 2.0 - (lhp + rhp) / 2.0
    U = unit(top - np.dot(top, L) * L)
    if U is None:
        return None
    return FWD_SIGN * np.cross(L, U), L, U


def calibrate_world(kps):
    """Find gravity and the forward direction from the data itself, so no axis convention is assumed."""
    global WORLD_DOWN, FWD_SIGN
    dy = []
    for kp in kps:
        if all(k in kp and "w" in kp[k] for k in ("leftHip", "rightHip", "leftAnkle", "rightAnkle")):
            hip_mid = (kp["leftHip"]["w"] + kp["rightHip"]["w"]) / 2.0
            ank_mid = (kp["leftAnkle"]["w"] + kp["rightAnkle"]["w"]) / 2.0
            dy.append(ank_mid[1] - hip_mid[1])
    if len(dy) < 5:
        print("[Depth] Not enough frames with hips + ankles in 3D -> using the flat 2D solver only.")
        return False
    WORLD_DOWN = np.array([0.0, 1.0 if np.median(dy) > 0 else -1.0, 0.0])

    FWD_SIGN = 1.0
    ahead = []
    for kp in kps:
        fr = torso_frame(kp)
        if fr is not None and "nose" in kp and "w" in kp["nose"]:
            mid_sh = (kp["leftShoulder"]["w"] + kp["rightShoulder"]["w"]) / 2.0
            ahead.append(np.dot(kp["nose"]["w"] - mid_sh, fr[0]))
    if ahead:
        FWD_SIGN = -1.0 if np.median(ahead) < 0 else 1.0
    print(f"[Depth] 3D solver ready: {len(dy)} usable frames, gravity along world y "
          f"({'+' if WORLD_DOWN[1] > 0 else '-'}), forward sign {FWD_SIGN:+.0f} "
          f"(from {len(ahead)} nose checks).")
    return True


def compute_targets_2d(kp):
    """Flat 2D solver (PoseNet, and the fallback for frames without depth)."""
    t = {}

    def put(name, value, *pts):
        t[name] = (float(value), float(conf(*pts)))

    for side, s in (("left", "l"), ("right", "r")):
        sh, el, wr = kp.get(side + "Shoulder"), kp.get(side + "Elbow"), kp.get(side + "Wrist")
        hp, kn, an = kp.get(side + "Hip"), kp.get(side + "Knee"), kp.get(side + "Ankle")

        # Elbow and knee: flexion = 180 - interior angle (0 = straight)
        if sh and el and wr:
            ang = interior_angle(sh, el, wr)
            if ang is not None:
                put(f"elbow_flexion_{s}", np.radians(180.0 - ang), sh, el, wr)
        if hp and kn and an:
            ang = interior_angle(hp, kn, an)
            if ang is not None:
                put(f"knee_angle_{s}", np.radians(180.0 - ang), hp, kn, an)

        # Shoulder: how far the upper arm is lifted away from the torso (0 = hanging down)
        if hp and sh and el:
            ang = interior_angle(hp, sh, el)
            if ang is not None:
                put(f"shoulder_elv_{s}", np.radians(ang), hp, sh, el)
                put(f"elv_angle_{s}", 0.0 if MOTION_PLANE == "frontal" else np.pi / 2, hp, sh, el)

        # Hip: how far the thigh is lifted away from the torso line (0 = standing)
        if sh and hp and kn:
            ang = interior_angle(sh, hp, kn)
            if ang is not None:
                lift = np.radians(180.0 - ang)
                if MOTION_PLANE == "frontal":
                    put(f"hip_adduction_{s}", -lift, sh, hp, kn)     # negative = leg moves outward
                else:
                    put(f"hip_flexion_{s}", lift, sh, hp, kn)        # positive = knee moves forward

    # Pelvis tilt sideways (list) from the hip line, and sideways position from the hip midpoint
    pelvis_list = 0.0
    if "leftHip" in kp and "rightHip" in kp:
        lh, rh = kp["leftHip"], kp["rightHip"]
        tilt = float(np.clip(wrap_half_pi(np.arctan2(lh["y"] - rh["y"], lh["x"] - rh["x"])), -0.35, 0.35))
        pelvis_list = PELVIS_LIST_SIGN * tilt         # tilt > 0 = left hip lower in the image
        if USE_PELVIS_LIST:
            put("pelvis_list", pelvis_list, lh, rh)
        put("_hipmid_x", midpoint(lh, rh)["x"], lh, rh)

    # Trunk lean (side bend) from mid-hips -> mid-shoulders, signed.
    # The spine joints sit on top of the pelvis, so the pelvis list is taken out of the bend.
    lean = 0.0
    if all(k in kp for k in ("leftShoulder", "rightShoulder", "leftHip", "rightHip")):
        top = midpoint(kp["leftShoulder"], kp["rightShoulder"])
        bottom = midpoint(kp["leftHip"], kp["rightHip"])
        lean = np.arctan2(top["x"] - bottom["x"], -(top["y"] - bottom["y"]))  # + = leans to image right
        bend = (-lean - (pelvis_list if USE_PELVIS_LIST else 0.0)) / 2.0
        pts = (kp["leftShoulder"], kp["rightShoulder"], kp["leftHip"], kp["rightHip"])
        put("L5_S1_LB", bend, *pts)
        put("T12_L1_LB", bend, *pts)

    # Head roll relative to the trunk, from the ears (eyes as a fallback)
    for l_name, r_name in (("leftEar", "rightEar"), ("leftEye", "rightEye")):
        if l_name in kp and r_name in kp:
            roll = wrap_half_pi(np.arctan2(kp[l_name]["y"] - kp[r_name]["y"], kp[l_name]["x"] - kp[r_name]["x"]))
            put("T1_head_neck_LB", -(roll - lean), kp[l_name], kp[r_name])
            break

    # Head turn from how far the nose sits off-centre between the eyes (+ = face turns to image right)
    if "nose" in kp and "leftEye" in kp and "rightEye" in kp:
        eye_mid = (kp["leftEye"]["x"] + kp["rightEye"]["x"]) / 2.0
        eye_dist = abs(kp["leftEye"]["x"] - kp["rightEye"]["x"]) or 1.0
        ratio = np.clip((kp["nose"]["x"] - eye_mid) / (eye_dist * 0.9), -1.0, 1.0)
        put("T1_head_neck_AR", np.arcsin(ratio), kp["nose"], kp["leftEye"], kp["rightEye"])
    return t


def compute_targets_3d(kp):
    """3D solver: every angle is measured inside the body frame (F forward, L left, U up), so the person
    can turn sideways. Returns the same {joint: (radians, confidence)} layout as the 2D solver."""
    t = {}
    frame = torso_frame(kp)
    if frame is None:
        return t
    F, L, U = frame
    up = -WORLD_DOWN

    def W(name):
        p = kp.get(name)
        return p["w"] if p is not None and "w" in p else None

    def put(name, value, parts, weight=1.0):
        t[name] = (float(value), float(min(kp[p]["score"] for p in parts) * weight))

    ls, rs, lh, rh = W("leftShoulder"), W("rightShoulder"), W("leftHip"), W("rightHip")
    hip_parts = ("leftHip", "rightHip")
    trunk_parts = ("leftShoulder", "rightShoulder", "leftHip", "rightHip")

    # --- pelvis: side tilt, turn about the vertical axis ---------------------------------------
    d_hip = lh - rh
    drop = float(np.dot(d_hip, WORLD_DOWN))                       # > 0 = left hip lower
    flat = float(np.sqrt(max(np.dot(d_hip, d_hip) - drop * drop, 1e-12)))
    pelvis_list = PELVIS_LIST_SIGN * float(np.clip(np.arctan2(drop, flat), -0.35, 0.35))
    if USE_PELVIS_LIST:
        put("pelvis_list", pelvis_list, hip_parts)
    yaw_hip = yaw_of(d_hip, up)
    yaw_sho = yaw_of(ls - rs, up)
    if yaw_hip is not None:
        put("_yaw_hip", yaw_hip, hip_parts)

    # --- trunk: side bend (relative to the pelvis, so turning does not fake a lean) and twist ---
    lat = flat_dir(d_hip, up)
    lean = 0.0
    if lat is not None:
        top = (ls + rs) / 2.0 - (lh + rh) / 2.0
        lean = float(np.arctan2(np.dot(top, lat), np.dot(top, up)))      # + = leans to the person's left
        bend = (-lean - (pelvis_list if USE_PELVIS_LIST else 0.0)) / 2.0
        put("L5_S1_LB", bend, trunk_parts)
        put("T12_L1_LB", bend, trunk_parts)
    if USE_TRUNK_TWIST and yaw_hip is not None and yaw_sho is not None:
        twist = wrap_pi(yaw_sho - yaw_hip) / 2.0                         # shared by the two spine joints
        put("L5_S1_AR", twist, trunk_parts)
        put("T12_L1_AR", twist, trunk_parts)

    # --- head ----------------------------------------------------------------------------------
    le, re = W("leftEar"), W("rightEar")
    if le is not None and re is not None:
        d_ear = le - re
        d_drop = float(np.dot(d_ear, WORLD_DOWN))
        d_flat = float(np.sqrt(max(np.dot(d_ear, d_ear) - d_drop * d_drop, 1e-12)))
        roll = float(np.arctan2(d_drop, d_flat))                         # > 0 = left ear lower
        put("T1_head_neck_LB", -(roll - lean), ("leftEar", "rightEar"))
        yaw_head = yaw_of(d_ear, up)
        if yaw_head is not None and yaw_sho is not None:
            put("T1_head_neck_AR", wrap_pi(yaw_head - yaw_sho), ("leftEar", "rightEar") + trunk_parts[:2])

    # --- arms and legs ---------------------------------------------------------------------------
    for side, s, sign in (("left", "l", 1.0), ("right", "r", -1.0)):
        out = sign * L                          # points away from the body on this side
        sh, el, wr = W(side + "Shoulder"), W(side + "Elbow"), W(side + "Wrist")
        hp, kn, an = W(side + "Hip"), W(side + "Knee"), W(side + "Ankle")
        arm_parts = (side + "Shoulder", side + "Elbow", side + "Wrist")
        leg_parts = (side + "Hip", side + "Knee", side + "Ankle")

        if sh is not None and el is not None and wr is not None:
            ang = interior_angle_3d(sh, el, wr)
            if ang is not None:
                put(f"elbow_flexion_{s}", np.radians(180.0 - ang), arm_parts)
        if hp is not None and kn is not None and an is not None:
            ang = interior_angle_3d(hp, kn, an)
            if ang is not None:
                put(f"knee_angle_{s}", np.radians(180.0 - ang), leg_parts)

        # Shoulder: elevation from the torso, and the plane of elevation (0 = out to the side,
        # 90 deg = forward). When the arm hangs, the plane is meaningless, so it fades to 0.
        if sh is not None and el is not None:
            arm = unit(el - sh)
            if arm is not None:
                elv = float(np.arccos(np.clip(np.dot(arm, -U), -1.0, 1.0)))
                put(f"shoulder_elv_{s}", elv, arm_parts[:2])
                plane = float(np.arctan2(np.dot(arm, F), np.dot(arm, out)))
                strength = float(np.clip((elv - np.radians(10.0)) / np.radians(20.0), 0.0, 1.0))
                put(f"elv_angle_{s}", plane * strength, arm_parts[:2], weight=max(strength, 0.05))

        # Hip: flexion (knee forward) and adduction (knee toward the midline), same order as the model
        if hp is not None and kn is not None:
            v = unit(kn - hp)
            if v is not None:
                put(f"hip_flexion_{s}", np.arctan2(np.dot(v, F), np.dot(v, -U)), leg_parts[:2])
                put(f"hip_adduction_{s}", -np.arcsin(np.clip(np.dot(v, out), -1.0, 1.0)), leg_parts[:2])

        # Ankle: dorsiflexion from the angle between the shank and the foot (heel -> toe)
        heel, toe = W(side + "Heel"), W(side + "FootIndex")
        if USE_ANKLE and kn is not None and an is not None and heel is not None and toe is not None:
            ang = angle_between(kn - an, toe - heel)
            if ang is not None:
                put(f"ankle_angle_{s}", np.radians(ANKLE_NEUTRAL_DEG - ang),
                    (side + "Knee", side + "Ankle", side + "Heel", side + "FootIndex"), weight=0.5)

        # Wrist: the hand direction against the forearm, split into palm-side bend and ulnar bend.
        # The palm side comes from the index/pinky positions (hand rule differs for left and right).
        idx, pnk = W(side + "Index"), W(side + "Pinky")
        if USE_WRIST and el is not None and wr is not None and idx is not None and pnk is not None:
            fore = unit(wr - el)
            hand = (idx + pnk) / 2.0 - wr
            across = pnk - idx                                           # points to the little-finger side
            if fore is not None and np.linalg.norm(hand) > 1e-6 and np.linalg.norm(across) > 1e-6:
                palm = np.cross(across, hand) if side == "left" else np.cross(hand, across)
                palm_dir = unit(palm - np.dot(palm, fore) * fore)
                ulnar_dir = unit(across - np.dot(across, fore) * fore)
                if palm_dir is not None and ulnar_dir is not None:
                    along = np.dot(hand, fore)
                    wrist_parts = (side + "Elbow", side + "Wrist", side + "Index", side + "Pinky")
                    flex = np.arctan2(np.dot(hand, palm_dir), along)
                    dev = np.arctan2(np.dot(hand, ulnar_dir), along)         # + = toward the little finger
                    put(f"flexion_{s}", flex * WRIST_FLEX_GAIN, wrist_parts, weight=0.5)
                    put(f"deviation_{s}", dev / WRIST_ULNAR_GAIN if dev > 0 else dev, wrist_parts, weight=0.5)
    return t


def compute_targets(kp):
    """Turn keypoints into {joint_name: (radians, confidence)}. The 3D solver overrides the 2D one for every
    joint it can measure in a frame; other joints keep the 2D value. Names starting with '_' are helper signals."""
    t = compute_targets_2d(kp)
    if DEPTH_READY:
        t.update(compute_targets_3d(kp))
    return t


def clamp_to_joint_range(jid, value):
    if model.jnt_limited[jid]:
        lo, hi = model.jnt_range[jid]
        return float(np.clip(value, lo, hi))
    return float(value)


# =============================================================================
# 5. PASS 1: MEASURE ALL FRAMES, CLEAN SPIKES, RESAMPLE + SMOOTH ON A TIME GRID
# =============================================================================
def hampel(y, floor):
    """Drop single-sample spikes (set to NaN). 'floor' = smallest deviation that can count as a spike."""
    out = y.copy()
    n = len(y)
    for i in range(n):
        if np.isnan(y[i]):
            continue
        w = y[max(0, i - HAMPEL_HALF_WINDOW):min(n, i + HAMPEL_HALF_WINDOW + 1)]
        w = w[~np.isnan(w)]
        if len(w) < 4:
            continue
        med = np.median(w)
        sigma = max(1.4826 * np.median(np.abs(w - med)), floor)
        if abs(y[i] - med) > HAMPEL_N_SIGMA * sigma:
            out[i] = np.nan
    return out


def resample_smooth(t, y, c, t_grid, sigma):
    """Confidence-weighted Gaussian smoothing evaluated on a uniform time grid.
    Handles uneven frame times and missing samples in one step; long gaps are bridged linearly."""
    ok = ~np.isnan(y)
    if not ok.any():
        return None
    t, y, c = t[ok], y[ok], np.maximum(c[ok], 1e-3)
    if sigma <= 0:
        return np.interp(t_grid, t, y)
    d = (t_grid[:, None] - t[None, :]) / sigma
    w = np.exp(-0.5 * d * d) * c[None, :]
    sw = w.sum(axis=1)
    est = np.full(len(t_grid), np.nan)
    good = sw > 1e-4
    est[good] = (w[good] * y[None, :]).sum(axis=1) / sw[good]
    if not good.any():
        return np.interp(t_grid, t, y)
    return np.interp(t_grid, t_grid[good], est[good])


def gauss_smooth_uniform(y, sigma_frames):
    if sigma_frames <= 0:
        return y
    r = int(np.ceil(3 * sigma_frames))
    kernel = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(y, r, mode="edge"), kernel, mode="valid")


def unwrap_nan(y):
    """np.unwrap that skips NaN samples (a turn past 180 degrees must not jump)."""
    out = y.copy()
    ok = ~np.isnan(y)
    if ok.sum() > 1:
        out[ok] = np.unwrap(y[ok])
    return out


def jitter(curves):
    """Average frame-to-frame wobble (mean absolute second difference) across joints."""
    vals = [np.mean(np.abs(np.diff(c, 2))) for c in curves if len(c) > 2]
    return float(np.mean(vals)) if vals else 0.0


DEPTH_READY = bool(USE_DEPTH and frames_with_world and calibrate_world(all_kp))
print(f"[Pass 1] Solver: {'3D (BlazePose world coordinates) with 2D fallback' if DEPTH_READY else 'flat 2D (PoseNet style)'}.")

print("\n[Pass 1] Measuring joint angles from every frame...")
PIXEL_SIGNALS = {"_hipmid_x"}
raw_targets = []
raw_leg_px = []
for kp in all_kp:
    raw_targets.append(compute_targets(kp) if kp else {})
    # straight-leg length in pixels (hip midpoint to ankle midpoint) for the pixel -> metre scale
    if all(k in kp for k in ("leftHip", "rightHip", "leftAnkle", "rightAnkle")):
        hm = midpoint(kp["leftHip"], kp["rightHip"])
        am = midpoint(kp["leftAnkle"], kp["rightAnkle"])
        raw_leg_px.append(np.hypot(hm["x"] - am["x"], hm["y"] - am["y"]))

frames_with_pose = sum(1 for t in raw_targets if t)
signal_names = sorted({n for t in raw_targets for n in t if n.startswith("_") or joint_id(n) != -1})

# Output grid: uniform in time, covering the recording
t0, t1 = frame_times[0], frame_times[-1]
grid_dt = PLAYBACK_SPEED / OUT_FPS
t_grid = np.arange(t0, t1 + 1e-9, grid_dt)
N_OUT = len(t_grid)

raw_curves, curves = {}, {}
for name in signal_names:
    y = np.array([t[name][0] if name in t else np.nan for t in raw_targets], dtype=float)
    c = np.array([t[name][1] if name in t else 0.0 for t in raw_targets], dtype=float)
    if name == "_yaw_hip":
        y = unwrap_nan(y)
    floor = 4.0 if name in PIXEL_SIGNALS else np.radians(4.0)
    y_clean = hampel(y, floor)
    curves[name] = resample_smooth(frame_times, y_clean, c, t_grid, SMOOTH_SEC)
    raw_curves[name] = resample_smooth(frame_times, y, np.ones_like(c), t_grid, 0.0)

joint_names = [n for n in signal_names if not n.startswith("_")]
spikes = 0
for n in joint_names:
    y = np.array([t[n][0] if n in t else np.nan for t in raw_targets], dtype=float)
    spikes += int(np.isnan(hampel(y, np.radians(4.0))).sum() - np.isnan(y).sum())
print(f"[Pass 1] {frames_with_pose}/{len(motion_data)} frames had a usable pose, {len(joint_names)} joints tracked, "
      f"{spikes} spike samples removed.")
print(f"[Pass 1] Resampled {len(motion_data)} recorded frames onto {N_OUT} video frames at {OUT_FPS} fps "
      f"(playback speed x{PLAYBACK_SPEED}).")
if joint_names:
    before = jitter([raw_curves[n] for n in joint_names])
    after = jitter([curves[n] for n in joint_names])
    if before > 0:
        print(f"[Pass 1] Smoothing reduced frame-to-frame jitter by {100 * (1 - after / before):.0f}%.")
extra_3d = [n for n in joint_names if n in DRIVEN_JOINTS_3D]
if extra_3d:
    print(f"[Pass 1] Depth-only joints tracked: {', '.join(extra_3d)}")

# Body turn: measured from the 3D hip line, or the slow auto-spin used before
if USE_MEASURED_TURN and curves.get("_yaw_hip") is not None:
    yaw = curves["_yaw_hip"].copy()
    if TURN_ZERO_AT_START:
        yaw -= np.median(yaw[:max(3, int(OUT_FPS * 0.3))])
    turn_angle = np.clip(yaw, -2.0, 2.0)
    print(f"[Turn] Measured body turn from 3D hips: {np.degrees(turn_angle.min()):.0f} to "
          f"{np.degrees(turn_angle.max()):.0f} degrees (0 = facing the camera).")
else:
    turn_angle = TURN_SPEED_RAD_PER_SEC / OUT_FPS * np.arange(N_OUT)
    print(f"[Turn] Slow auto-spin of {TURN_SPEED_RAD_PER_SEC:.2f} rad/s (no 3D turn available or turned off).")

# Sideways pelvis movement from the hip midpoint (pixels -> metres)
shift_m = np.zeros(N_OUT)
if USE_PELVIS_SHIFT and "_hipmid_x" in curves and raw_leg_px:
    px_to_m = MODEL_LEG_M / float(np.percentile(raw_leg_px, 90))
    hx = curves["_hipmid_x"]
    # image right = person's left = -pelvis_tz direction is handled where qpos is built
    shift_m = np.clip((hx - np.median(hx)) * px_to_m, -MAX_SHIFT_M, MAX_SHIFT_M)
    print(f"[Pelvis] Scale {px_to_m * 100:.3f} cm/pixel, sideways range "
          f"{shift_m.min() * 100:.0f} to {shift_m.max() * 100:.0f} cm.")

# =============================================================================
# 6. BUILD THE FULL POSE (qpos) FOR EVERY OUTPUT FRAME
# =============================================================================
print("\n[Pose] Building model poses for every video frame...")
Q = np.zeros((N_OUT, model.nq))
lift = np.zeros(N_OUT)
for i in range(N_OUT):
    q = model.qpos0.copy()
    for name in joint_names:
        q[qadr(name)] = clamp_to_joint_range(joint_id(name), curves[name][i])
    apply_couplings(q)
    q[model.jnt_qposadr[TURN_JOINT]] = turn_angle[i]
    q[model.jnt_qposadr[SIDE_JOINT]] = -shift_m[i]      # image right = world +Y = -pelvis_tz
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    lowest_foot = min(data.xpos[b][2] for b in FOOT_BODIES)
    lift[i] = REST_FOOT_Z - lowest_foot                 # keep the lowest foot on the floor
    Q[i] = q

# Smooth the foot-lock correction too, otherwise it jitters on its own
lift = gauss_smooth_uniform(lift, SMOOTH_SEC * OUT_FPS / PLAYBACK_SPEED)
Q[:, model.jnt_qposadr[LIFT_JOINT]] += lift

# =============================================================================
# 7. MUSCLES: LENGTH, SPEED, FORCE CAPACITY, ENGAGEMENT ESTIMATE, SYNERGIES
# =============================================================================
# Honest note: with motion data only, we know how each muscle's length changes, not how hard it
# is working. The engagement index below is an ESTIMATE built from three things we can know:
#   depth  : how far the muscle is shortened compared with standing
#   speed  : how fast it shortens (concentric) or lengthens (eccentric, counted at half weight)
#   capacity: how much active force this muscle could make at that length and speed (the muscle
#            curves already inside MS-Human-700, evaluated by MuJoCo with full activation)
# engagement = capacity * (0.5 * depth + 0.5 * speed), from 0 (idle) to 1 (hard working).
NU = model.nu
MUSCLE_TENDON = model.actuator_trnid[:NU, 0].astype(int)
MUSCLE_NAMES = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(NU)]
span = np.maximum(model.actuator_lengthrange[:, 1] - model.actuator_lengthrange[:, 0], 1e-6)
FMAX = model.actuator_gainprm[:NU, 2].copy()
FMAX[FMAX <= 0] = 1.0

# "Standing" = the neutral pose WITH the joint couplings applied, exactly like every video frame. (Raw qpos0
# differs from it, e.g. the kneecap sits 5 cm off, which made the quadriceps look stretched at rest in v014.)
q_stand = model.qpos0.copy()
apply_couplings(q_stand)
data.qpos[:] = q_stand
data.qvel[:] = 0.0
data.act[:] = 0.0
mujoco.mj_forward(model, data)
rest_len = data.actuator_length.copy()

# Joint speeds from the smoothed pose, in real recording time (this model has no free joint: nq == nv)
if model.nq == model.nv:
    QV = np.gradient(Q, grid_dt, axis=0)
else:
    QV = np.zeros_like(Q)

delta = np.zeros((N_OUT, NU))                           # fraction of working range, - = shortened
speed = np.zeros((N_OUT, NU))                           # working ranges per second, - = shortening
passive = np.zeros((N_OUT, NU))                         # passive tension / Fmax
full = np.zeros((N_OUT, NU))                            # tension with full activation / Fmax
for i in range(N_OUT):
    data.qpos[:] = Q[i]
    data.qvel[:] = QV[i]
    data.act[:] = 0.0
    mujoco.mj_forward(model, data)
    delta[i] = (data.actuator_length - rest_len) / span
    speed[i] = data.actuator_velocity / span
    passive[i] = -data.actuator_force / FMAX
    data.act[:] = 1.0
    mujoco.mj_forward(model, data)
    full[i] = -data.actuator_force / FMAX
data.qvel[:] = 0.0
data.act[:] = 0.0

speed = np.apply_along_axis(gauss_smooth_uniform, 0, speed, 0.08 * OUT_FPS)
capacity = np.clip(full - passive, 0.0, 1.0)            # active force this muscle could make, fraction of Fmax

depth_term = np.clip(-delta / COLOR_FULL_SCALE, 0.0, 1.0)
short_term = np.clip(-speed / SPEED_FULL_SCALE, 0.0, 1.0)
long_term = np.clip(speed / SPEED_FULL_SCALE, 0.0, 1.0)
engagement = capacity * np.clip(0.5 * depth_term + 0.5 * np.maximum(short_term, 0.5 * long_term), 0.0, 1.0)


def extract_synergies(E, k, iters=400):
    """Non-negative matrix factorisation E (time x muscles) ~ W (time x k) @ H (k x muscles).
    Muscles with a large weight in the same column of H rise and fall together: a synergy."""
    rng = np.random.default_rng(0)
    W = rng.random((E.shape[0], k)) + 0.1
    H = rng.random((k, E.shape[1])) + 0.1
    for _ in range(iters):
        H *= (W.T @ E) / (W.T @ W @ H + 1e-9)
        W *= (E @ H.T) / (W @ H @ H.T + 1e-9)
    peak = H.max(axis=1, keepdims=True) + 1e-12
    return W * peak.T, H / peak


synergy_k = 0
syn_time = None
syn_of = np.full(NU, -1)
syn_share = np.zeros(NU)
syn_vaf = 0.0
active_muscles = np.where(engagement.max(axis=0) > 0.02)[0]
if SYNERGY_COUNT > 0 and len(active_muscles) >= 2 and N_OUT >= 4:
    synergy_k = int(min(SYNERGY_COUNT, len(active_muscles), N_OUT))
    E_act = engagement[:, active_muscles]
    W_syn, H_syn = extract_synergies(E_act, synergy_k)
    order_syn = np.argsort(-W_syn.sum(axis=0))                  # biggest synergy first
    W_syn, H_syn = W_syn[:, order_syn], H_syn[order_syn]
    syn_vaf = 1.0 - float(np.sum((E_act - W_syn @ H_syn) ** 2) / max(np.sum(E_act ** 2), 1e-12))
    contribution = H_syn * W_syn.mean(axis=0)[:, None]          # how much each synergy explains each muscle
    syn_of[active_muscles] = np.argmax(contribution, axis=0)
    syn_share[active_muscles] = contribution.max(axis=0) / np.maximum(contribution.sum(axis=0), 1e-12)
    syn_time = W_syn / max(W_syn.max(), 1e-12)                  # 0..1 activation over time, for the HUD
    syn_top = [[MUSCLE_NAMES[active_muscles[m]] for m in np.argsort(-contribution[s])[:4]] for s in range(synergy_k)]
    print(f"\n[Synergies] {synergy_k} synergies explain {100 * syn_vaf:.0f}% of the estimated engagement pattern "
          f"({len(active_muscles)} muscles with any engagement).")
    for s in range(synergy_k):
        print(f"   S{s + 1}: peaks at t = {t_grid[int(np.argmax(syn_time[:, s]))] - t0:5.2f} s   "
              f"led by {', '.join(syn_top[s])}")

base_rgba = model.tendon_rgba[MUSCLE_TENDON].copy()


def muscle_rgba(d):
    m = np.clip(np.abs(d) / COLOR_FULL_SCALE, 0.0, 1.0)[:, None]
    red = np.array([1.0, 0.15, 0.10])
    blue = np.array([0.10, 0.45, 1.0])
    target = np.where((d < 0)[:, None], red[None, :], blue[None, :])
    neutral = np.array([0.55, 0.50, 0.50])
    rgb = neutral[None, :] + (target - neutral[None, :]) * m
    alpha = 0.20 + 0.80 * m
    return np.concatenate([rgb, alpha], axis=1)


def engagement_rgba(e):
    m = np.clip(e / ENGAGE_COLOR_FULL, 0.0, 1.0)[:, None]
    hot = np.array([1.0, 0.55, 0.05])
    neutral = np.array([0.55, 0.50, 0.50])
    rgb = neutral[None, :] + (hot[None, :] - neutral[None, :]) * m
    alpha = 0.20 + 0.80 * m
    return np.concatenate([rgb, alpha], axis=1)


# =============================================================================
# 8. RENDERER, CAMERA, VIDEO WRITER
# =============================================================================
print(f"\n[Renderer] Initializing offscreen renderer ({WIDTH}x{HEIGHT}) and video writer...")
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

# Fixed camera in front of the person (+X is the direction the model faces, Z is up)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.0, 0.85]
cam.distance = 2.6
cam.azimuth = 180
cam.elevation = -8

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = SHOW_MUSCLES

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (WIDTH, HEIGHT))
if not out.isOpened():
    raise RuntimeError("Could not open the video writer.")

FONT = cv2.FONT_HERSHEY_SIMPLEX
pose_label = "pose: BlazePose 3D" if DEPTH_READY else "pose: PoseNet 2D"


def dark_panel(img, x0, y0, x1, y1, alpha=0.55):
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1.0 - alpha, np.zeros_like(roi), alpha, 0)


def bar(img, x, y, w, h, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
    cv2.rectangle(img, (x, y), (x + int(w * float(np.clip(frac, 0.0, 1.0))), y + h), color, -1)


def draw_hud(img, i):
    cv2.putText(img, f"t = {t_grid[i] - t0:5.2f} s", (20, 36), FONT, 0.9, (255, 255, 255), 2)
    cv2.putText(img, pose_label, (20, 62), FONT, 0.5, (200, 200, 200), 1)
    if COLOR_MUSCLES and SHOW_MUSCLES:
        if COLOR_MODE == "engagement":
            cv2.putText(img, "orange = estimated muscle engagement", (20, 88), FONT, 0.6, (40, 140, 255), 2)
        else:
            cv2.putText(img, "red = muscle shorter than standing", (20, 88), FONT, 0.6, (60, 60, 255), 2)
            cv2.putText(img, "blue = muscle longer than standing", (20, 114), FONT, 0.6, (255, 150, 60), 2)

    # most engaged muscles right now (an estimate, see the note in section 7)
    rows = int(HUD_TOP_MUSCLES)
    if rows > 0:
        x0, y0 = WIDTH - 330, 12
        dark_panel(img, x0, y0, WIDTH - 12, y0 + 34 + 26 * rows)
        cv2.putText(img, "most engaged now (estimate)", (x0 + 10, y0 + 22), FONT, 0.5, (255, 255, 255), 1)
        for r, a in enumerate(np.argsort(-engagement[i])[:rows]):
            y = y0 + 40 + 26 * r
            cv2.putText(img, MUSCLE_NAMES[a][:15], (x0 + 10, y + 12), FONT, 0.45, (230, 230, 230), 1)
            bar(img, x0 + 150, y, 150, 14, engagement[i, a], (40, 140, 255))

    # synergy activation over time
    if synergy_k:
        x0, y1 = 12, HEIGHT - 12
        y0 = y1 - (34 + 24 * synergy_k)
        dark_panel(img, x0, y0, x0 + 300, y1)
        cv2.putText(img, f"muscle synergies (explain {100 * syn_vaf:.0f}%)", (x0 + 10, y0 + 22), FONT, 0.5, (255, 255, 255), 1)
        for s in range(synergy_k):
            y = y0 + 38 + 24 * s
            cv2.putText(img, f"S{s + 1}", (x0 + 10, y + 12), FONT, 0.5, (230, 230, 230), 1)
            bar(img, x0 + 50, y, 230, 14, syn_time[i, s], (120, 220, 120))


print(f"\n=== PASS 2: RENDERING {N_OUT} frames at {OUT_FPS} fps ===")
start_time = time.time()

for i in range(N_OUT):
    frame_start_time = time.time()

    data.qpos[:] = Q[i]
    mujoco.mj_forward(model, data)

    if COLOR_MUSCLES and SHOW_MUSCLES:
        if COLOR_MODE == "engagement":
            model.tendon_rgba[MUSCLE_TENDON] = engagement_rgba(engagement[i])
        else:
            model.tendon_rgba[MUSCLE_TENDON] = muscle_rgba(delta[i])

    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

    if DRAW_OVERLAY:
        draw_hud(img, i)
    out.write(img)

    frame_duration_ms = (time.time() - frame_start_time) * 1000
    if i == 0 or (i + 1) % 30 == 0 or (i + 1) == N_OUT:
        print(f"[Frame {i + 1}/{N_OUT}] t = {t_grid[i] - t0:.2f} s | Frame Duration: {frame_duration_ms:.2f} ms")

out.release()
model.tendon_rgba[MUSCLE_TENDON] = base_rgba
total_elapsed = time.time() - start_time

# =============================================================================
# 9. SAVE NUMBERS FOR ANALYSIS
# =============================================================================
with open(OUTPUT_ANGLES_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s"] + [f"{n}_deg" for n in joint_names] + ["body_turn_deg", "pelvis_shift_cm"])
    for i in range(N_OUT):
        w.writerow([f"{t_grid[i] - t0:.3f}"] + [f"{np.degrees(curves[n][i]):.2f}" for n in joint_names]
                   + [f"{np.degrees(turn_angle[i]):.2f}", f"{shift_m[i] * 100:.2f}"])

peak_short = -delta.min(axis=0)       # biggest shortening (fraction of working range)
peak_long = delta.max(axis=0)         # biggest lengthening
motion_range = delta.max(axis=0) - delta.min(axis=0)
peak_short_speed = np.maximum(-speed.min(axis=0), 0.0)      # working ranges per second
peak_long_speed = np.maximum(speed.max(axis=0), 0.0)
mean_capacity = capacity.mean(axis=0)
peak_engage = engagement.max(axis=0)
mean_engage = engagement.mean(axis=0)
time_of_peak = t_grid[np.argmax(engagement, axis=0)] - t0
order = np.argsort(-mean_engage)
with open(OUTPUT_MUSCLES_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["muscle", "peak_shortening_pct_of_range", "peak_lengthening_pct_of_range", "total_change_pct_of_range",
                "peak_shortening_speed_range_per_s", "peak_lengthening_speed_range_per_s",
                "mean_force_capacity_pct", "mean_engagement_pct", "peak_engagement_pct", "time_of_peak_s",
                "synergy", "synergy_share_pct"])
    for a in order:
        w.writerow([MUSCLE_NAMES[a], f"{100 * peak_short[a]:.1f}", f"{100 * peak_long[a]:.1f}", f"{100 * motion_range[a]:.1f}",
                    f"{peak_short_speed[a]:.2f}", f"{peak_long_speed[a]:.2f}",
                    f"{100 * mean_capacity[a]:.1f}", f"{100 * mean_engage[a]:.1f}", f"{100 * peak_engage[a]:.1f}",
                    f"{time_of_peak[a]:.2f}",
                    (f"S{syn_of[a] + 1}" if syn_of[a] >= 0 else ""), (f"{100 * syn_share[a]:.0f}" if syn_of[a] >= 0 else "")])

if synergy_k:
    with open(OUTPUT_SYNERGY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + [f"S{s + 1}_activation" for s in range(synergy_k)])
        for i in range(N_OUT):
            w.writerow([f"{t_grid[i] - t0:.3f}"] + [f"{syn_time[i, s]:.3f}" for s in range(synergy_k)])

print("\n[Muscles] 10 muscles with the highest estimated engagement (mean over the motion):")
for a in order[:10]:
    print(f"   {MUSCLE_NAMES[a]:<18} engagement mean {100 * mean_engage[a]:4.1f}%  peak {100 * peak_engage[a]:4.1f}%   "
          f"shortens {100 * peak_short[a]:5.1f}%  lengthens {100 * peak_long[a]:5.1f}%  of working range")

print("\n=== SIMULATION COMPLETE ===")
print(f"Recorded frames: {len(motion_data)} -> video frames: {N_OUT} at {OUT_FPS} fps")
print(f"Successfully Mapped Motion Frames: {frames_with_pose}")
print(f"Total Processing Time: {total_elapsed:.2f} seconds")
print(f"Output Video Saved Successfully as: {OUTPUT_VIDEO}")
print(f"Also saved: {OUTPUT_ANGLES_CSV}, {OUTPUT_MUSCLES_CSV}" + (f", {OUTPUT_SYNERGY_CSV}" if synergy_k else ""))
