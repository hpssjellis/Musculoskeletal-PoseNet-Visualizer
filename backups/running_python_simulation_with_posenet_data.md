# Guide: Running the Python Simulation with PoseNet Data

Follow these instructions to run your offline Python biomechanical processing script using the `pose_motion_data.json` file recorded from your browser.

---

## Step 1: Organize Your Project Files

Ensure all required files are placed in your working project directory:

- `pose_motion_data.json` — Downloaded motion capture file from your web app.
- `MS-Human-700.xml` — The MuJoCo XML model file.
- `run_simulation.py` — Your main Python script.

---

## Step 2: Activate Your Virtual Environment

Open your terminal or command prompt, navigate to your project directory, and activate your environment:

### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)
```cmd
venv\Scripts\activate.bat
```

### macOS / Linux
```bash
source venv/bin/activate
```

---

## Step 3: Create the Python Execution Script (`run_simulation.py`)

Save the following code as `run_simulation.py` inside your project folder:

```python
import json
import cv2
import numpy as np
import mujoco

# 1. Load the recorded JSON file exported from your browser
print("Loading pose_motion_data.json...")
with open("pose_motion_data.json", "r") as f:
    motion_data = json.load(f)

print(f"Successfully loaded {len(motion_data)} frames of motion data.")

# 2. Load the MS-Human-700 MuJoCo model
model = mujoco.MjModel.from_xml_path("MS-Human-700.xml")
data = mujoco.MjData(model)

# 3. Setup offscreen renderer and OpenCV video writer
width, height = 1280, 720
renderer = mujoco.Renderer(model, height=height, width=width)
out = cv2.VideoWriter(
    "musculoskeletal_output.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    30, 
    (width, height)
)

print("Processing simulation and rendering video frames...")

# 4. Step through recorded frames
for i, frame in enumerate(motion_data):
    keypoints = frame["keypoints"]
    timestamp = frame["time"]
    
    # Forward simulation state to evaluate muscle forces
    mujoco.mj_forward(model, data)
    
    # Read muscle actuator forces
    muscle_forces = data.actuator_force  
    
    # Render frame
    renderer.update_scene(data)
    pixels = renderer.render()
    
    # Convert RGB to BGR for OpenCV export
    frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

# 5. Clean up resources
out.release()
print("Done! Simulation output saved as 'musculoskeletal_output.mp4'.")
```

---

## Step 4: Run the Script

Execute the Python script in your active terminal:

```bash
python run_simulation.py
```

After execution finishes, open `musculoskeletal_output.mp4` to view the rendered biomechanical output.