import os
import subprocess
import json
import time
import cv2
import numpy as np
import mujoco

print("=== STARTING SIMULATION PIPELINE ===")

# Ensure the MS-Human-700 repository/folder is present locally
model_dir = "MS-Human-700"
xml_filename = "MS-Human-700.xml"
xml_path = os.path.join(model_dir, xml_filename)

if not os.path.exists(xml_path):
    print(f"[Model Setup] '{xml_path}' not found locally. Cloning repository from GitHub...")
    try:
        subprocess.run([
            "git", "clone", 
            "https://github.com/LNSGroup/MS-Human-700.git", 
            model_dir
        ], check=True)
        print("[Model Setup] Clone completed successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to clone repository automatically: {e}\n"
            "Please run 'git clone https://github.com/LNSGroup/MS-Human-700.git' manually in your terminal."
        )
else:
    print(f"[Model Setup] Found existing model file at: {xml_path}")

# 1. Load the recorded JSON file from your browser
json_filename = "pose_motion_data.json"
print(f"[Data Loading] Looking for '{json_filename}'...")

if os.path.exists(json_filename):
    with open(json_filename, "r") as f:
        motion_data = json.load(f)
    print(f"[Data Loading] Success! Loaded {len(motion_data)} frames from {json_filename}.")
else:
    print(f"[Data Loading] WARNING: '{json_filename}' was NOT found in this directory!")
    print("[Data Loading] Please ensure you saved and moved your JSON file from the browser app into this folder.")
    print("[Data Loading] Creating a placeholder test pattern so the script doesn't crash...")
    motion_data = [{"keypoints": [], "time": i} for i in range(60)]

# 2. Load the MS-Human-700 MuJoCo model
print(f"[MuJoCo] Loading model configuration from {xml_path}...")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
print(f"[MuJoCo] Model loaded successfully! Total joints in model: {model.njnt}")

# 3. Setup offscreen renderer and video writer
width, height = 1280, 720
print(f"[Renderer] Initializing offscreen renderer ({width}x{height}) and video writer...")
renderer = mujoco.Renderer(model, height=height, width=width)
out = cv2.VideoWriter(
    "musculoskeletal_output.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    30, 
    (width, height)
)

print("\n=== STARTING FRAME PROCESSING & CONVERSION ===")

# Helper function to calculate angle between 3 keypoints
def calculate_angle(p1, p2, p3):
    a = np.array([p1['x'], p1['y']])
    b = np.array([p2['x'], p2['y']])
    c = np.array([p3['x'], p3['y']])
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return np.radians(angle)

# Track timing for delay diagnostics
start_time = time.time()
mapped_frames_count = 0

# 4. Iterate through each recorded frame and apply pose data
for i, frame in enumerate(motion_data):
    frame_start_time = time.time()
    
    # Extract keypoints from frame
    keypoints_list = frame.get("keypoints", [])
    keypoints = {kp['part']: kp['position'] for kp in keypoints_list if 'part' in kp and 'position' in kp}
    
    conversion_status = "No Keypoints"
    
    if len(keypoints) > 0:
        try:
            # Check if left arm keypoints are available to map
            if 'leftShoulder' in keypoints and 'leftElbow' in keypoints and 'leftWrist' in keypoints:
                elbow_angle = calculate_angle(
                    keypoints['leftShoulder'], 
                    keypoints['leftElbow'], 
                    keypoints['leftWrist']
                )
                
                # Try finding joint ID (print warning if joint name differs)
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_elbow')
                if joint_id != -1:
                    qpos_adr = model.jnt_qposadr[joint_id]
                    data.qpos[qpos_adr] = elbow_angle
                    conversion_status = "Converted & Mapped Left Elbow"
                    mapped_frames_count += 1
                else:
                    conversion_status = "Keypoints found, but joint name 'left_elbow' not in XML"
        except Exception as e:
            conversion_status = f"Error during conversion: {e}"

    # Advance the MuJoCo physics solver forward
    mujoco.mj_forward(model, data)
    
    # Render off-screen scene to pixels
    renderer.update_scene(data)
    pixels = renderer.render()
    
    # Convert RGB to BGR and write frame
    frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
    
    # Calculate processing delay for this specific frame
    frame_duration_ms = (time.time() - frame_start_time) * 1000
    
    # Print status update every 10 frames or on the first frame
    if i == 0 or (i + 1) % 10 == 0 or (i + 1) == len(motion_data):
        print(f"[Frame {i+1}/{len(motion_data)}] Status: {conversion_status} | Frame Render Delay: {frame_duration_ms:.2f} ms")

# 5. Clean up resources
out.release()
total_elapsed = time.time() - start_time

print("\n=== SIMULATION COMPLETE ===")
print(f"Total Frames Processed: {len(motion_data)}")
print(f"Successfully Mapped Motion Frames: {mapped_frames_count}")
print(f"Total Processing Time: {total_elapsed:.2f} seconds")
print("Output Video Saved Successfully as: musculoskeletal_output.mp4")
