

# Musculoskeletal PoseNet & 3D Actuator Visualizer

A real-time biomechanical simulation tool combining PoseNet motion capture with a 700-actuator virtual human body model in MuJoCo.

Live Web Demo: [Musculoskeletal-PoseNet-Visualizer](https://hpssjellis.github.io/Musculoskeletal-PoseNet-Visualizer/index.html)

---

## ⚡ Quick Start Overview

1. Open `index.html` in your browser to record your movement and export `pose_motion_data.json`.
2. Set up your Python environment using Windows Command Prompt (cmd.exe).
3. Run `run_simulation.py` to generate the biomechanical output video (`musculoskeletal_output.mp4`).

---

## 🛠️ Step 1: Windows Environment Setup (Command Prompt)

Open **Command Prompt** (`cmd.exe`) and navigate to your project folder:

```cmd
cd C:\path\to\your\project-folder
```

Create and activate a Python virtual environment:

```cmd
:: Create the virtual environment
python -m venv venv

:: Activate the virtual environment
venv\Scripts\activate.bat
```

Install the required Python packages:

```cmd
python -m pip install --upgrade pip
pip install opencv-python numpy mujoco huggingface_hub
```

---

## 💻 Step 2: Create the Python Script (`run_simulation.py`)

Save the following code as `run_simulation.py` inside your project directory:

```python
import os
import json
import cv2
import numpy as np
import mujoco
from huggingface_hub import hf_hub_download

# 1. Download MS-Human-700 model automatically if missing
xml_filename = "MS-Human-700.xml"
if not os.path.exists(xml_filename):
    print("Downloading MS-Human-700 model from Hugging Face...")
    xml_path = hf_hub_download(repo_id="BAAI/MS-Human-700", filename=xml_filename)
else:
    xml_path = xml_filename

# 2. Load pose motion data exported from the browser
print("Loading pose_motion_data.json...")
try:
    with open("pose_motion_data.json", "r") as f:
        motion_data = json.load(f)
    print(f"Loaded {len(motion_data)} frames.")
except FileNotFoundError:
    print("Warning: pose_motion_data.json not found. Running with mock data.")
    motion_data = [{"keypoints": []} for _ in range(30)]

# 3. Load MuJoCo Model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 4. Setup Offscreen Renderer & Video Writer
width, height = 1280, 720
renderer = mujoco.Renderer(model, height=height, width=width)
out = cv2.VideoWriter(
    "musculoskeletal_output.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    30, 
    (width, height)
)

print("Running simulation and rendering video...")

# 5. Render simulation frames
for frame in motion_data:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

out.release()
print("Done! Video saved to musculoskeletal_output.mp4")
```

---

## 🚀 Step 3: Run the Simulation

Run the script in your activated Command Prompt window:

```cmd
python run_simulation.py
```

Once completed, open `musculoskeletal_output.mp4` to view your rendered simulation.
