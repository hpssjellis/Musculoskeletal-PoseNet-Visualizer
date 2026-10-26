# Musculoskeletal-PoseNet-Visualizer
Combining the 700 actuator virtual Human body with posenet.

https://hpssjellis.github.io/Musculoskeletal-PoseNet-Visualizer/index.html


First a fake example of what we want to do in the browser. The a python version of it, probably starting by recording the video then sending it to psoenet then sending it to the analyzer then having it make a video.




Use the webpage to record your data. then use this instruction set


.


.



# Biomechanical PoseNet & 3D Muscle Actuator Visualizer

This project is a web-based, real-time biomechanical simulation application. It integrates TensorFlow.js PoseNet pose estimation with a 3D musculoskeletal anatomical model built in Three.js to analyze joint kinematics and visualize muscular tension heatmaps.

---

## 🛠️ Tech Stack & Dependencies

The project is built as a single-file Web application requiring no build step or node installation. The following CDN libraries are imported:

- **Tailwind CSS v3**: Styling and glassmorphism interface.
- **FontAwesome v6**: UI iconography.
- **TensorFlow.js Core v3.20.0** & **PoseNet v2.2.2**: AI Pose detection engine.
- **Three.js r128** & **OrbitControls**: 3D rendering and viewport camera controls.

---

## 🚀 Key Features

1. **Real-time Camera Kinematics**: Tracks pose keypoints via webcam input to dynamically compute joint angles in real time.
2. **Interactive 3D Musculoskeletal Model**: Features 36 core muscle groups and bone pivot hierarchies.
3. **Muscle Actuator Heatmap**: Dynamic color interpolation changing from cyan (resting) to red/orange (high tension flex) based on angle kinematics.
4. **Preset Motion Cycles**: Built-in animation engines for T-Pose, Squat Cycles, and Bicep Curls for testing without camera access.
5. **Real-Time Telemetry Panel**: Live numerical display of flexion angles across elbows, knees, shoulders, and hips.

---



## 🏃 Running the Application

1. Save the code block above as an `.html` file (e.g., `index.html`).
2. Open the file directly in any WebGL-compatible modern browser (Chrome, Firefox, Safari, Edge).
3. Click **Start Camera** to allow webcam permissions for AI pose tracking, or select any of the preset motion buttons to test the kinematics engine.


.










.



.



.




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




.



.




.














# Biomechanical Pipeline Environment Setup Guide

---

## Guide 1: Setup Commands for Python 3.14.2 & Pip 25.3

Use these commands if you want to proceed with your current Python installation.

### Windows (PowerShell / Command Prompt)

```bash
# 1. Navigate to your project directory
cd C:\path\to\your\project-folder

# 2. Create a isolated virtual environment
python -m venv venv

# 3. Activate the virtual environment
# For PowerShell:
.\venv\Scripts\Activate.ps1
# For Command Prompt:
venv\Scripts\activate.bat

# 4. Upgrade core build tools
python -m pip install --upgrade pip setuptools wheel

# 5. Install required computer vision & biomechanics packages
pip install opencv-python numpy mujoco

# 6. Save installed dependencies to requirements file
pip freeze > requirements.txt
```

### macOS / Linux

```bash
# 1. Navigate to your project directory
cd /path/to/your/project-folder

# 2. Create a isolated virtual environment
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Upgrade core build tools
python -m pip install --upgrade pip setuptools wheel

# 5. Install required computer vision & biomechanics packages
pip install opencv-python numpy mujoco

# 6. Save installed dependencies to requirements file
pip freeze > requirements.txt
```

---

## Guide 2: Switching to Recommended Python Version (3.11 / 3.12)

Because C-extension binary packages (like MediaPipe, OpenCV, and MuJoCo bindings) often lag behind cutting-edge Python releases, **Python 3.11 or 3.12** is currently the most stable and broadly supported version for ML and biomechanics work.

Below are the commands to install Python 3.11/3.12 alongside your existing version and target it directly when building your environment.

### Windows (Using official Python launcher or `uv`)

```bash
# Option A: If Python 3.11/3.12 is installed via python.org installer
# Use the Python Launcher ('py') to target a specific version:

# 1. Create venv using Python 3.11 specifically
py -3.11 -m venv venv_311

# 2. Activate environment
.\venv_311\Scripts\Activate.ps1

# 3. Install dependencies inside the 3.11 environment
python -m pip install --upgrade pip setuptools wheel
pip install opencv-python numpy mujoco mediapipe


# Option B: Using 'uv' (Fastest tool for managing multiple Python versions)

# 1. Install 'uv'
pip install uv

# 2. Create a venv with Python 3.11 automatically downloading the runtime if needed
uv venv --python 3.11 venv

# 3. Activate and install packages
.\venv\Scripts\Activate.ps1
uv pip install opencv-python numpy mujoco mediapipe
```

### macOS (Using Homebrew & `uv`)

```bash
# Option A: Using Homebrew to manage Python versions

# 1. Install Python 3.11 alongside existing versions
brew install python@3.11

# 2. Create virtual environment pointing to 3.11
python3.11 -m venv venv

# 3. Activate environment
source venv/bin/activate

# 4. Install libraries
pip install --upgrade pip setuptools wheel
pip install opencv-python numpy mujoco mediapipe


# Option B: Using 'uv'

# 1. Install uv
brew install uv

# 2. Create environment targeting Python 3.11
uv venv --python 3.11 venv

# 3. Activate environment and install packages
source venv/bin/activate
uv pip install opencv-python numpy mujoco mediapipe
```

---

## Summary of Environment Status

| Task | Recommended Command |
| :--- | :--- |
| **Check Active Python Version** | `python --version` |
| **Check Active Pip Version** | `pip --version` |
| **Deactivate Environment** | `deactivate` |
