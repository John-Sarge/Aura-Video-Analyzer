# Aura Video-Analyzer v3.1

A RealSense and MediaPipe-powered augmented reality application inspired by the Ghostbusters universe. The Aura Video-Analyzer uses real-time depth mapping to detect when a user places a "Brain Scanner" (like a metal colander) on their head, instantly projecting a psycho-kinetically charged 3D entity (such as a Terror Dog or Slimer) over their face with retro thermal visual effects.

## Features

- **True Depth Detection:** Utilizes intel RealSense depth coordinate mapping to physically detect objects on top of the user's head, ignoring false-positive color heuristics.
- **Real-Time Head Tracking:** Leverages MediaPipe Face Mesh to orient and anchor 3D models to the user's head movements (Pitch, Yaw, Roll).
- **Retro Thermal Rendering:** Custom dynamic vertex color mapping that generates a glowing, multi-colored aura (similar to 1980s thermal scanning visuals) via Pyrender.
- **Interference & Static Effects:** Simulates weak signals with CRT scanlines, tracking tears, and heavy static when the connection to the subject is lost or jittery.
- **Hot-swappable Models & Live Calibration:** Instantly swap between 3D entities, and tweak their rotation/scale in real time via keyboard inputs.
- **Diagnostic PiP:** Features a corner Picture-in-Picture display of the raw camera feed for debugging.

## Prerequisites

### Hardware
- **Intel RealSense D435i** (or compatible depth camera)
- A metal colander (or similar physical prop) to trigger the aura scan.

### Software Libraries
This project requires Python 3.8+ and the following dependencies:
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Intel RealSense SDK (`pyrealsense2`)
- MediaPipe (`mediapipe`)
- Trimesh (`trimesh`)
- Pyrender (`pyrender`)
- *Additional requirement: `pyglet` is usually needed by Trimesh/Pyrender.*

Install dependencies via pip:
```bash
pip install opencv-python numpy pyrealsense2 mediapipe trimesh pyrender pyglet
```

## Running the Application

1. Make sure your Intel RealSense camera is plugged in.
2. Place the required 3D model files (`terrordog.obj`, `slimer.obj`) and background image (`AVA_BACKGROUND.png`) in the same directory.
3. Run the script:
```bash
python terrordog.py
```

## Controls

Once the application is running, use the following keys to configure your view:

| Key | Action |
| --- | --- |
| **`ESC`** | Quit the program |
| **`t`** | Load the Terror Dog model (Default) |
| **`s`** | Load the Slimer model |
| **`x` / `X`** | Rotate the active model along the X-axis (Pitch) |
| **`y` / `Y`** | Rotate the active model along the Y-axis (Yaw) |
| **`z` / `Z`** | Rotate the active model along the Z-axis (Roll) |
| **`+` / `=`** | Increase model scale (Make it larger/closer) |
| **`-` / `_`** | Decrease model scale (Make it smaller/further) |

## Adding Additional Models

You can easily add more entities by editing the `models_config` dictionary found in the `__init__` method of `terrordog.py`. 

```python
self.models_config = {
    'm': {
        'file': 'marshmallow_man.obj',
        'base_rot': [math.radians(-90), 0.0, 0.0], # Fix inverted initial orientation
        'base_scale': 1.5                          # Make it spawn 50% larger
    }
}
```
