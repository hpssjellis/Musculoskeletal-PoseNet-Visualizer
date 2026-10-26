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
    motion_data = [{"keypoints": []} for _ in range(30)]  # 1 second mock data at 30fps

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

print("Starting simulation and rendering video frames...")

# 4. Iterate through each recorded frame
for i, frame in enumerate(motion_data):
    # Optional: Map keypoints to model joint states (qpos) using Inverse Kinematics here
    # Example placeholder: data.qpos[...] = calculated_joint_angle
    
    # Advance the MuJoCo physics solver forward to evaluate all muscle actuators
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
