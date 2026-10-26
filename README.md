# Musculoskeletal-PoseNet-Visualizer
Combining the 700 actuator virtual Human body with posenet.

https://hpssjellis.github.io/Musculoskeletal-PoseNet-Visualizer/index.html


First a fake example of what we want to do in the browser. The a python version of it, probably starting by recording the video then sending it to psoenet then sending it to the analyzer then having it make a video.




Use the webpage to record your data. then use this instruction set


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
