"""
run_simulation.py  v013  -  PoseNet JSON  ->  MS-Human-700 (MuJoCo)  ->  musculoskeletal_output.mp4

The web page only makes a ROUGH pose capture. This program does the real work.

POSSIBLE CHANGE LATER (better pose data, still 100% local)
  PoseNet gives 17 flat 2D points. MediaPipe BlazePose (pip install mediapipe) runs locally in Python,
  can read a recorded video file directly (no browser/JSON step), and returns 33 landmarks with 3D
  coordinates in metres (heels, toes, hands included). That would allow real depth for shoulder/hip
  rotation, ankle and wrist angles. Only compute_targets() would need new mappings; the smoothing,
  coupling, foot-lock and muscle code below would stay as it is.

MUSCLE COLOURS: red/blue show how a muscle's LENGTH changes (a clue, not force). Real muscle
activation needs inverse dynamics plus an optimisation step, which is a separate next version.
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

WIDTH, HEIGHT = 1280, 720

# --- Timing (v013: the recorded timestamps are now used) ---------------------
OUT_FPS = 30                       # video frame rate. The JSON is resampled onto this grid.
PLAYBACK_SPEED = 1.0               # 1.0 = real time, 0.5 = half speed slow motion

# "frontal"  : person faces the camera -> arms/legs spreading sideways (abduction)
# "sagittal" : person seen from the side -> arms/legs swinging forward (flexion)
MOTION_PLANE = "frontal"

TURN_SPEED_RAD_PER_SEC = 0.15      # spin about the VERTICAL (world Z) axis, per second of video
MIN_SCORE = 0.2                    # ignore PoseNet keypoints below this confidence
SMOOTH_SEC = 0.15                  # smoothing width in SECONDS (0 = off, bigger = smoother)
HAMPEL_HALF_WINDOW = 3             # spike filter looks at +-3 samples
HAMPEL_N_SIGMA = 3.0               # a sample this many robust sigmas off the median is dropped

# --- Pelvis ------------------------------------------------------------------
USE_PELVIS_LIST = True             # hip-line tilt drives pelvis_list
USE_PELVIS_SHIFT = True            # left/right movement of the hips moves the model sideways
MAX_SHIFT_M = 1.0                  # clamp for the sideways movement (keeps the model in view)

# --- Muscles -----------------------------------------------------------------
SHOW_MUSCLES = True                # False hides the muscles so the bones are easier to see
COLOR_MUSCLES = True               # red = shortening, blue = lengthening (vs the standing pose)
COLOR_FULL_SCALE = 0.20            # a 20% change of a muscle's working range = full colour
DRAW_OVERLAY = True                # time + colour legend in the corner of the video

print("=== STARTING SIMULATION PIPELINE (v013) ===")

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
if os.path.exists(JSON_FILENAME):
    with open(JSON_FILENAME, "r") as f:
        motion_data = json.load(f)
    if isinstance(motion_data, dict):          # also accept {"frames": [...]}
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
#   pelvis_rotation spins about the vertical (+Z) axis.
TURN_JOINT = joint_id("pelvis_rotation")
LIFT_JOINT = joint_id("pelvis_ty")
SIDE_JOINT = joint_id("pelvis_tz")
if min(TURN_JOINT, LIFT_JOINT, SIDE_JOINT) == -1:
    raise RuntimeError("Could not find pelvis_rotation / pelvis_ty / pelvis_tz in this model.")

FOOT_BODIES = [b for b in (body_id(n) for n in ("calcn_r", "calcn_l", "toes_r", "toes_l")) if b != -1]

# Every joint PoseNet can measure (17 keypoints -> these joints)
DRIVEN_JOINTS = [
    "elbow_flexion_r", "elbow_flexion_l",
    "knee_angle_r", "knee_angle_l",
    "shoulder_elv_r", "shoulder_elv_l", "elv_angle_r", "elv_angle_l",
    "hip_flexion_r", "hip_flexion_l", "hip_adduction_r", "hip_adduction_l",
    "L5_S1_LB", "T12_L1_LB", "T1_head_neck_LB", "T1_head_neck_AR",
    "pelvis_list",
]
missing = [n for n in DRIVEN_JOINTS if joint_id(n) == -1]
print(f"\n[Joint Mapping] {len(DRIVEN_JOINTS) - len(missing)}/{len(DRIVEN_JOINTS)} PoseNet-driven joints found in the model."
      + (f" Missing: {missing}" if missing else ""))

# --- Coupled joints (v013) ----------------------------------------------------
# The model ties 42 joints to other joints with polynomial equations: the knee's gliding and
# rotation, the kneecap, the collarbone/shoulder-blade rhythm when the arm is raised, the wrist.
# Setting qpos directly ignores these, so the kneecap and shoulder blade used to stay frozen.
# We evaluate the same polynomials ourselves so those parts move with the joints we drive.
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
# 4. ANGLE MATH (all angles in 2D image space)
# =============================================================================
def interior_angle(a, b, c):
    """Angle in degrees at point b between a-b-c (180 = straight line)."""
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


def conf(*pts):
    return min(p["score"] for p in pts)


def compute_targets(kp):
    """Turn PoseNet keypoints into {joint_name: (radians, confidence)} for every joint we can measure.
    Names starting with '_' are helper signals, not joints."""
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
        pelvis_list = float(np.clip(wrap_half_pi(np.arctan2(lh["y"] - rh["y"], lh["x"] - rh["x"])),
                                    -0.35, 0.35))
        if USE_PELVIS_LIST:
            put("pelvis_list", pelvis_list, lh, rh)     # + = left hip lower in the image
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


def jitter(curves):
    """Average frame-to-frame wobble (mean absolute second difference) across joints."""
    vals = [np.mean(np.abs(np.diff(c, 2))) for c in curves if len(c) > 2]
    return float(np.mean(vals)) if vals else 0.0


print("\n[Pass 1] Measuring joint angles from every frame...")
raw_targets = []
raw_leg_px = []
for frame in motion_data:
    kp = {}
    for k in frame.get("keypoints", []):
        if "part" in k and "position" in k and k.get("score", 1.0) >= MIN_SCORE:
            kp[k["part"]] = {"x": k["position"]["x"], "y": k["position"]["y"], "score": k.get("score", 1.0)}
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
    floor = 4.0 if name.startswith("_") else np.radians(4.0)
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
turn_per_frame = TURN_SPEED_RAD_PER_SEC / OUT_FPS
for i in range(N_OUT):
    q = model.qpos0.copy()
    for name in joint_names:
        q[qadr(name)] = clamp_to_joint_range(joint_id(name), curves[name][i])
    apply_couplings(q)
    q[model.jnt_qposadr[TURN_JOINT]] = turn_per_frame * i
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
# 7. MUSCLE LENGTHS (kinematic proxy for muscle use)
# =============================================================================
# Honest note: with motion data only, we know how each muscle's length changes, not how hard it
# is working. Shortening / lengthening is a first clue; real activation needs forces (inverse
# dynamics + optimisation), which is a separate step.
NU = model.nu
MUSCLE_TENDON = model.actuator_trnid[:NU, 0].astype(int)
MUSCLE_NAMES = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(NU)]
span = np.maximum(model.actuator_lengthrange[:, 1] - model.actuator_lengthrange[:, 0], 1e-6)

data.qpos[:] = model.qpos0
mujoco.mj_forward(model, data)
rest_len = data.actuator_length.copy()

delta = np.zeros((N_OUT, NU))                           # fraction of working range, - = shortened
for i in range(N_OUT):
    data.qpos[:] = Q[i]
    mujoco.mj_forward(model, data)
    delta[i] = (data.actuator_length - rest_len) / span

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

print(f"\n=== PASS 2: RENDERING {N_OUT} frames at {OUT_FPS} fps ===")
start_time = time.time()

for i in range(N_OUT):
    frame_start_time = time.time()

    data.qpos[:] = Q[i]
    mujoco.mj_forward(model, data)

    if COLOR_MUSCLES and SHOW_MUSCLES:
        model.tendon_rgba[MUSCLE_TENDON] = muscle_rgba(delta[i])

    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

    if DRAW_OVERLAY:
        cv2.putText(img, f"t = {t_grid[i] - t0:5.2f} s", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if COLOR_MUSCLES and SHOW_MUSCLES:
            cv2.putText(img, "red = muscle shorter than standing", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2)
            cv2.putText(img, "blue = muscle longer than standing", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 60), 2)
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
    w.writerow(["time_s"] + [f"{n}_deg" for n in joint_names] + ["pelvis_shift_cm"])
    for i in range(N_OUT):
        w.writerow([f"{t_grid[i] - t0:.3f}"] + [f"{np.degrees(curves[n][i]):.2f}" for n in joint_names]
                   + [f"{shift_m[i] * 100:.2f}"])

peak_short = -delta.min(axis=0)       # biggest shortening (fraction of working range)
peak_long = delta.max(axis=0)         # biggest lengthening
motion_range = delta.max(axis=0) - delta.min(axis=0)
order = np.argsort(-motion_range)
with open(OUTPUT_MUSCLES_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["muscle", "peak_shortening_pct_of_range", "peak_lengthening_pct_of_range", "total_change_pct_of_range"])
    for a in order:
        w.writerow([MUSCLE_NAMES[a], f"{100 * peak_short[a]:.1f}", f"{100 * peak_long[a]:.1f}", f"{100 * motion_range[a]:.1f}"])

print("\n[Muscles] 10 muscles whose length changed the most during this motion:")
for a in order[:10]:
    print(f"   {MUSCLE_NAMES[a]:<18} shortens {100 * peak_short[a]:5.1f}%  lengthens {100 * peak_long[a]:5.1f}%  of working range")

print("\n=== SIMULATION COMPLETE ===")
print(f"Recorded frames: {len(motion_data)} -> video frames: {N_OUT} at {OUT_FPS} fps")
print(f"Successfully Mapped Motion Frames: {frames_with_pose}")
print(f"Total Processing Time: {total_elapsed:.2f} seconds")
print(f"Output Video Saved Successfully as: {OUTPUT_VIDEO}")
print(f"Also saved: {OUTPUT_ANGLES_CSV}, {OUTPUT_MUSCLES_CSV}")
