import json
import cv2
import numpy as np
import mujoco

# 1. Load the recorded JSON file from your browser
print("Loading pose_motion_data.json...")
with open("pose_motion_data.json", "r") as f:
    motion_data = json.load(f)

print(f"Loaded {len(motion_data)} frames of motion data.")

# 2. Load MS-Human-700 MuJoCo model
# Ensure MS-Human-700.xml is in your directory or referenced correctly
model = mujoco.MjModel.from_xml_path("MS-Human-700.xml")
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
    keypoints = frame["keypoints"]
    
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
