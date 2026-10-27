import os
import sys
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

WIDTH, HEIGHT = 1280, 720
TARGET_FPS = 5                     # 5 FPS = 200 ms per frame in the saved video
PREVIEW_DELAY_SEC = 0.0            # set to 0.2 only if you want to watch it process slowly

# "frontal"  : person faces the camera -> arms/legs spreading sideways (abduction)
# "sagittal" : person seen from the side -> arms/legs swinging forward (flexion)
MOTION_PLANE = "frontal"

TURN_SPEED_PER_FRAME = 0.03        # radians per frame, spin about the VERTICAL (world Z) axis
MIN_SCORE = 0.3                    # ignore PoseNet keypoints below this confidence
SMOOTH_SIGMA = 1.5                 # smoothing in frames (0 = off, bigger = smoother)
SHOW_MUSCLES = True                # False hides the red muscles so the bones are easier to see

print("=== STARTING SIMULATION PIPELINE (v012) ===")

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
    print(f"[Data Loading] Success! Loaded {len(motion_data)} frames from {JSON_FILENAME}.")
else:
    print(f"[Data Loading] WARNING: '{JSON_FILENAME}' was NOT found in this directory!")
    motion_data = [{"keypoints": [], "time": i} for i in range(60)]

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


# This model has NO free root joint. It is built like an OpenSim model:
# pelvis_tx/ty/tz (slides) + pelvis_tilt/list/rotation (hinges). In the world frame
# pelvis_ty moves straight UP (+Z) and pelvis_rotation spins about the vertical (+Z) axis.
TURN_JOINT = joint_id("pelvis_rotation")
LIFT_JOINT = joint_id("pelvis_ty")
if TURN_JOINT == -1 or LIFT_JOINT == -1:
    raise RuntimeError("Could not find pelvis_rotation / pelvis_ty in this model.")

FOOT_BODIES = [b for b in (body_id(n) for n in ("calcn_r", "calcn_l", "toes_r", "toes_l")) if b != -1]

# Every joint PoseNet can measure (17 keypoints -> these joints)
DRIVEN_JOINTS = [
    "elbow_flexion_r", "elbow_flexion_l",
    "knee_angle_r", "knee_angle_l",
    "shoulder_elv_r", "shoulder_elv_l", "elv_angle_r", "elv_angle_l",
    "hip_flexion_r", "hip_flexion_l", "hip_adduction_r", "hip_adduction_l",
    "L5_S1_LB", "T12_L1_LB", "T1_head_neck_LB", "T1_head_neck_AR",
]
missing = [n for n in DRIVEN_JOINTS if joint_id(n) == -1]
print(f"\n[Joint Mapping] {len(DRIVEN_JOINTS) - len(missing)}/{len(DRIVEN_JOINTS)} PoseNet-driven joints found in the model."
      + (f" Missing: {missing}" if missing else ""))

# Remember how high the feet are in the neutral standing pose, so squats keep feet on the floor
mujoco.mj_forward(model, data)
REST_FOOT_Z = min(data.xpos[b][2] for b in FOOT_BODIES)


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


def compute_targets(kp):
    """Turn PoseNet keypoints into {joint_name: radians} for every joint we can measure."""
    t = {}
    for side, s in (("left", "l"), ("right", "r")):
        sh, el, wr = kp.get(side + "Shoulder"), kp.get(side + "Elbow"), kp.get(side + "Wrist")
        hp, kn, an = kp.get(side + "Hip"), kp.get(side + "Knee"), kp.get(side + "Ankle")

        # Elbow and knee: flexion = 180 - interior angle (0 = straight)
        if sh and el and wr:
            ang = interior_angle(sh, el, wr)
            if ang is not None:
                t[f"elbow_flexion_{s}"] = np.radians(180.0 - ang)
        if hp and kn and an:
            ang = interior_angle(hp, kn, an)
            if ang is not None:
                t[f"knee_angle_{s}"] = np.radians(180.0 - ang)

        # Shoulder: how far the upper arm is lifted away from the torso (0 = hanging down)
        if hp and sh and el:
            ang = interior_angle(hp, sh, el)
            if ang is not None:
                t[f"shoulder_elv_{s}"] = np.radians(ang)
                t[f"elv_angle_{s}"] = 0.0 if MOTION_PLANE == "frontal" else np.pi / 2

        # Hip: how far the thigh is lifted away from the torso line (0 = standing)
        if sh and hp and kn:
            ang = interior_angle(sh, hp, kn)
            if ang is not None:
                lift = np.radians(180.0 - ang)
                if MOTION_PLANE == "frontal":
                    t[f"hip_adduction_{s}"] = -lift     # negative = leg moves outward
                else:
                    t[f"hip_flexion_{s}"] = lift        # positive = knee moves forward

    # Trunk lean (side bend) from mid-hips -> mid-shoulders, signed
    lean = 0.0
    if all(k in kp for k in ("leftShoulder", "rightShoulder", "leftHip", "rightHip")):
        top = midpoint(kp["leftShoulder"], kp["rightShoulder"])
        bottom = midpoint(kp["leftHip"], kp["rightHip"])
        lean = np.arctan2(top["x"] - bottom["x"], -(top["y"] - bottom["y"]))  # + = leans to image right
        t["L5_S1_LB"] = -lean / 2.0     # model's side-bend axis is opposite in sign
        t["T12_L1_LB"] = -lean / 2.0

    # Head roll relative to the trunk, from the ears (eyes as a fallback)
    for l_name, r_name in (("leftEar", "rightEar"), ("leftEye", "rightEye")):
        if l_name in kp and r_name in kp:
            roll = np.arctan2(kp[l_name]["y"] - kp[r_name]["y"], kp[l_name]["x"] - kp[r_name]["x"])
            t["T1_head_neck_LB"] = -(roll - lean)
            break

    # Head turn from how far the nose sits off-centre between the eyes (+ = face turns to image right)
    if "nose" in kp and "leftEye" in kp and "rightEye" in kp:
        eye_mid = (kp["leftEye"]["x"] + kp["rightEye"]["x"]) / 2.0
        eye_dist = abs(kp["leftEye"]["x"] - kp["rightEye"]["x"]) or 1.0
        ratio = np.clip((kp["nose"]["x"] - eye_mid) / (eye_dist * 0.9), -1.0, 1.0)
        t["T1_head_neck_AR"] = float(np.arcsin(ratio))
    return t


def clamp_to_joint_range(jid, value):
    if model.jnt_limited[jid]:
        lo, hi = model.jnt_range[jid]
        return float(np.clip(value, lo, hi))
    return float(value)


# =============================================================================
# 5. PASS 1: MEASURE ALL FRAMES, THEN FILL GAPS AND SMOOTH (offline = zero lag)
# =============================================================================
def smooth_curve(y):
    """Fill missing frames (NaN) by interpolation, then Gaussian-smooth with no time lag."""
    ok = ~np.isnan(y)
    if not ok.any():
        return y
    x = np.arange(len(y))
    y = np.interp(x, x[ok], y[ok])
    if SMOOTH_SIGMA <= 0:
        return y
    r = int(np.ceil(3 * SMOOTH_SIGMA))
    kernel = np.exp(-0.5 * (np.arange(-r, r + 1) / SMOOTH_SIGMA) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(y, r, mode="edge"), kernel, mode="valid")


def jitter(curves):
    """Average frame-to-frame wobble (mean absolute second difference) across joints."""
    vals = [np.mean(np.abs(np.diff(c, 2))) for c in curves if len(c) > 2]
    return float(np.mean(vals)) if vals else 0.0


print("\n[Pass 1] Measuring joint angles from every frame...")
raw_targets = []
for frame in motion_data:
    kp = {}
    for k in frame.get("keypoints", []):
        if "part" in k and "position" in k and k.get("score", 1.0) >= MIN_SCORE:
            kp[k["part"]] = k["position"]
    raw_targets.append(compute_targets(kp) if kp else {})

frames_with_pose = sum(1 for t in raw_targets if t)
joint_names = sorted({n for t in raw_targets for n in t if joint_id(n) != -1})

raw_curves, curves = {}, {}
for name in joint_names:
    raw_curves[name] = np.array([t.get(name, np.nan) for t in raw_targets], dtype=float)
    curves[name] = smooth_curve(raw_curves[name])

gaps = sum(int(np.isnan(c).sum()) for c in raw_curves.values())
print(f"[Pass 1] {frames_with_pose}/{len(motion_data)} frames had a usable pose, {len(joint_names)} joints tracked, "
      f"{gaps} missing joint-samples filled by interpolation.")
if joint_names:
    before = jitter([np.interp(np.arange(len(c)), np.where(~np.isnan(c))[0], c[~np.isnan(c)]) for c in raw_curves.values() if (~np.isnan(c)).any()])
    after = jitter(list(curves.values()))
    if before > 0:
        print(f"[Pass 1] Smoothing reduced frame-to-frame jitter by {100 * (1 - after / before):.0f}%.")

# =============================================================================
# 6. RENDERER, CAMERA, VIDEO WRITER
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

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (WIDTH, HEIGHT))
if not out.isOpened():
    raise RuntimeError("Could not open the video writer.")

print(f"\n=== PASS 2: RENDERING (Target Delay: {int(1000 / TARGET_FPS)} ms / frame) ===")

start_time = time.time()
turn_angle = 0.0           # spin about the vertical Z axis

# =============================================================================
# 7. PASS 2: RENDER EVERY FRAME
# =============================================================================
for i in range(len(motion_data)):
    frame_start_time = time.time()

    data.qpos[:] = model.qpos0
    for name in joint_names:
        jid = joint_id(name)
        data.qpos[model.jnt_qposadr[jid]] = clamp_to_joint_range(jid, curves[name][i])

    # Slow turn about the vertical (world Z) axis
    turn_angle += TURN_SPEED_PER_FRAME
    data.qpos[model.jnt_qposadr[TURN_JOINT]] = turn_angle

    # Keep the lowest foot on the floor (squats lower the pelvis instead of floating)
    mujoco.mj_forward(model, data)
    lowest_foot = min(data.xpos[b][2] for b in FOOT_BODIES)
    data.qpos[model.jnt_qposadr[LIFT_JOINT]] += REST_FOOT_Z - lowest_foot
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    out.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    if PREVIEW_DELAY_SEC > 0:
        time.sleep(PREVIEW_DELAY_SEC)

    frame_duration_ms = (time.time() - frame_start_time) * 1000
    if i == 0 or (i + 1) % 10 == 0 or (i + 1) == len(motion_data):
        status = f"Mapped {len(raw_targets[i])} joints" if raw_targets[i] else "No keypoints (gap filled)"
        print(f"[Frame {i+1}/{len(motion_data)}] Status: {status} | Frame Duration: {frame_duration_ms:.2f} ms")

out.release()
total_elapsed = time.time() - start_time

print("\n=== SIMULATION COMPLETE ===")
print(f"Total Frames Processed: {len(motion_data)}")
print(f"Successfully Mapped Motion Frames: {frames_with_pose}")
print(f"Total Processing Time: {total_elapsed:.2f} seconds")
print(f"Output Video Saved Successfully as: {OUTPUT_VIDEO} ({TARGET_FPS} FPS, turning about the vertical Z axis)")
