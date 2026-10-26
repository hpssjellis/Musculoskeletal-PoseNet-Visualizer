import os
import subprocess
import json
import cv2
import numpy as np
import mujoco

# Ensure the MS-Human-700 repository/folder is present locally
model_dir = "MS-Human-700"
xml_filename = "MS-Human-700.xml"
xml_path = os.path.join(model_dir, xml_filename)

if not os.path.exists(xml_path):
    print("Cloning MS-Human-700 model and asset dependencies from GitHub...")
    try:
        subprocess.run([
            "git", "clone", 
            "https://github.com/LNSGroup/MS-Human-700.git", 
            model_dir
        ], check=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to clone repository automatically: {e}\n"
            "Please run 'git clone https://github.com/LNSGroup/MS-Human-700.git' manually in your terminal."
        )

# 1. Load the recorded JSON file from your browser
print("Loading pose_motion_data.json...")
try:
    with open("pose_motion_data.json", "r") as f:
        motion_data = json.load(f)
    print(f"Loaded {len(motion_data)} frames of motion data.")
except FileNotFoundError:
    print("Warning: 'pose_motion_data.json' not found. Creating mock frame data to allow simulation run.")
    motion_data = [{"keypoints": []} for _ in range(60)]  # 2 seconds of mock data at 30fps

# 2. Load the MS-Human-700 MuJoCo model
print(f"Loading MuJoCo model from {xml_path}...")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 3. Setup offscreen renderer and video writer
width, height = 1280, 720
renderer = mujoco.Renderer(model, height=height, width=width)
out = cv2.VideoWriter(
    "musculoskeletal_output.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    30, 
    (width, height)
)

print("Starting simulation, parsing motion data, and rendering video frames...")

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

# 4. Iterate through each recorded frame and apply pose data
for i, frame in enumerate(motion_data):
    # Convert list of keypoints into a dictionary lookup {part_name: {x, y}}
    keypoints_list = frame.get("keypoints", [])
    keypoints = {kp['part']: kp['position'] for kp in keypoints_list if 'part' in kp and 'position' in kp}
    
    try:
        # Example: Map Left Elbow angle if keypoints are present
        if 'leftShoulder' in keypoints and 'leftElbow' in keypoints and 'leftWrist' in keypoints:
            elbow_angle = calculate_angle(
                keypoints['leftShoulder'], 
                keypoints['leftElbow'], 
                keypoints['leftWrist']
            )
            # Find joint ID in MuJoCo model and update its qpos
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_elbow') # Adjust joint name if needed
            if joint_id != -1:
                qpos_adr = model.jnt_qposadr[joint_id]
                data.qpos[qpos_adr] = elbow_angle
    except Exception:
        pass # Fallback gracefully if specific joint name or indexing differs

    # Advance the MuJoCo physics solver forward to evaluate muscle actuators and dynamics
    mujoco.mj_forward(model, data)
    
    # Read calculated muscle tensions/forces across actuators if needed
    muscle_forces = data.actuator_force  
    
    # Render off-screen scene to pixels
    renderer.update_scene(data)
    pixels = renderer.render()
    
    # Convert RGB (MuJoCo format) to BGR (OpenCV video format) and write frame
    frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

# 5. Clean up resources
out.release()
print("Simulation complete! Video saved as musculoskeletal_output.mp4")
