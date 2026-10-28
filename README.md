

# Musculoskeletal PoseNet & 3D Actuator Visualizer

A real-time biomechanical simulation tool combining PoseNet motion capture with a 700-actuator virtual human body model in MuJoCo.

Live Web Demo: [Musculoskeletal-PoseNet-Visualizer](https://hpssjellis.github.io/Musculoskeletal-PoseNet-Visualizer/index.html)


Main Inspiration from  https://github.com/LNSGroup/MS-Human-700



---

## ⚡ Quick Start Overview

1. Open `index.html` in your browser to record your movement and export `pose_motion_data.json`.
2. Set up your Python environment using Windows Command Prompt (cmd.exe).
3. Run `run_simulation.py` to generate the biomechanical output video (`musculoskeletal_output.mp4`).

---

## 🛠️ Step 1: Windows Environment Setup (Command Prompt)

Open **Command Prompt** (`cmd.exe`) and navigate to your project folder:

Easier just to download the repository unzip it and open the command prompt in the main folder

```cmd
:: Find your project folder
cd C:\path\to\your\project-folder
```

Create and activate a Python virtual environment:

```cmd
:: Create the virtual environment
python -m venv venv
```

```cmd
:: Activate the virtual environment
venv\Scripts\activate.bat
```

Install the required Python packages:

```cmd
python -m pip install --upgrade pip
```

```cmd
pip install opencv-python numpy mujoco huggingface_hub
```



## 🚀 Step 3: Run the Simulation

Run the script in your activated Command Prompt window:

```cmd
python run_simulation.py
```

Once completed, open `musculoskeletal_output.mp4` to view your rendered simulation.

I save it as a new name since the next run will override it.
