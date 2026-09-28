# Smart Targeting System

Real-time object detection, tracking, and servo-driven laser targeting, built as the final project for **ENG 301 – Computer Vision** at Fulbright University Vietnam.

The system uses a custom-trained YOLOv8 model to detect three Vietnamese milk bottle brands (**Fami**, **TH True Milk**, **Vinamilk**) from a live camera. It then:

- smooths the target's position with a **Kalman filter**,
- estimates the target's distance with a **stereo camera** pair,
- and points a **pan/tilt laser** at each detected bottle, visiting them in an order computed with **Kruskal's MST + DFS** to cut down servo travel.

| Kalman tracking | Stereo depth estimation |
|---|---|
| [video/kalman_tracking.mp4](video/kalman_tracking.mp4) | [video/estimate_depth.mp4](video/estimate_depth.mp4) |

---

## Repository layout

```
smart-targeting-system/
├── code/
│   ├── best.pt             # Custom-trained YOLOv8 weights (classes: fami, th, vina)
│   ├── kalman.py           # Detection + Kalman-filtered tracking of one target
│   ├── depth.py            # Stereo depth estimation (SGBM + WLS) + detection
│   └── servo_control.py    # Detection + MST visit order + Arduino servo control
├── dataset/
│   └── images/             # 316 self-captured bottle photos used for training
├── video/                  # Demo recordings
└── Final_Report.docx       # Full write-up: method, parameter choices, results
```

## How it works

```
Camera frame ──► YOLOv8 detection ──► bounding-box centers (cx, cy)
                                          │
              ┌───────────────────────────┼──────────────────────────┐
              ▼                           ▼                          ▼
     Kalman filter (kalman.py)   Stereo depth (depth.py)   Visit order (servo_control.py)
     constant-velocity model     SGBM disparity + WLS      Kruskal MST → DFS traversal
     smooths (cx, cy)            → distance in metres      → pixel → servo angle → laser
```

### 1. Object detection
A YOLOv8 model fine-tuned on photos of the three bottle brands, taken from different angles and lighting conditions. Hue, saturation, and brightness augmentation during training help it cope with lighting changes. Final validation mAP50 is close to 1.0 and mAP50-95 is about 0.84.

### 2. Kalman tracking — [code/kalman.py](code/kalman.py)
A constant-velocity model with state `[x, y, vx, vy]` and measurement `[x, y]` (via `filterpy`):

| Matrix | Value | Why |
|---|---|---|
| `F` | constant-velocity transition | bottle motion is smooth frame to frame |
| `H` | measures position only | YOLO gives a center, not a velocity |
| `P₀` | `1000·I` | high initial uncertainty so the filter locks on quickly |
| `R` | `5·I` | about ±5 px noise in YOLO centers |
| `Q` | `I` | small process noise; the bottles rarely change direction abruptly |

The overlay shows the raw center in red, the filtered center in blue, and the running average error and RMSE between them.

### 3. Stereo depth — [code/depth.py](code/depth.py)
- Left and right frames are converted to grayscale, Gaussian-blurred, and histogram-equalized.
- Disparity comes from `StereoSGBM` (tuned `numDisparities=112`, `blockSize=15`, …), refined with a **WLS filter** that uses a right-view matcher.
- Distance is `Z = f · B / d`, using the mean of the 10 largest valid disparities in a central ROI (`f = 1360 px`, `B = 0.09 m`).
- **Red-region correction:** close objects show up red in the JET-colored depth map, so the share of red pixels in the ROI (averaged over 5 frames) × 9 is subtracted from the raw distance. The result is clamped to 0.1–3.0 m and smoothed over 5 frames.

### 4. Optimized servo targeting — [code/servo_control.py](code/servo_control.py)
When two or more bottles are detected:
1. Build a complete graph over their centers, weighted by Euclidean distance.
2. Compute a minimum spanning tree with **Kruskal's algorithm** (union-find with path compression).
3. Traverse the MST with **DFS** to get a visit order that avoids long back-and-forth jumps.
4. Map each center linearly to pan/tilt angles (X flipped to match the camera mirror) and drive the servos over **Firmata**, pausing 3 s on each target.

## Hardware

| Component | Notes |
|---|---|
| USB webcam (Logitech) | Index `0`; used by all scripts |
| Second USB webcam | Index `1`; right camera for `depth.py`, mounted about 9 cm from the left one |
| Arduino (Uno or similar) | Flashed with **StandardFirmata** (Arduino IDE → Examples → Firmata) |
| 2× hobby servos | Pan on pin **D3**, tilt on pin **D2** |
| Laser pointer | Mounted on the pan/tilt bracket |

## Setup

Requires Python 3.8+.

```bash
git clone <repo-url>
cd smart-targeting-system
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install ultralytics opencv-contrib-python numpy filterpy pyfirmata
```

> `opencv-contrib-python` (not plain `opencv-python`) is required for the WLS filter in `cv2.ximgproc`.
>
> `pyfirmata` calls `inspect.getargspec`, which was removed in Python 3.11. On 3.11+, use Python 3.10 or switch to the `pyfirmata2` fork.

## Usage

The scripts load `best.pt` by relative path, so run them from the `code/` directory. Press **`q`** to quit any window.

```bash
cd code

python kalman.py          # single camera: detection + Kalman tracking
python depth.py           # two cameras: depth map + distance readout
python servo_control.py   # camera + Arduino: laser visits each bottle in MST order
```

Before running `servo_control.py`:
- Set the serial port in `pyfirmata.Arduino('COM3')` to match your board (`/dev/ttyACM0` or `/dev/ttyUSB0` on Linux, `/dev/cu.usbmodem*` on macOS).
- Calibrate `X_SERVO_MIN/MAX` and `Y_SERVO_MIN/MAX` so the laser's range covers the camera's field of view on your mount.

For `depth.py` on a different rig, update `focal_length` and `baseline` to match your camera calibration.

## Results

| Metric | Result |
|---|---|
| Detection mAP50 / mAP50-95 (validation) | ≈ 1.0 / ≈ 0.84 |
| Mean detection confidence, without → with preprocessing | 0.83 → 0.94 |
| Kalman average error / RMSE (poor → good lighting) | 6.33 → 3.46 px / 10.34 → 7.34 px |
| Stereo distance example | Fami 0.87 m vs. TH 0.95 m, correctly ordered |

See [Final_Report.docx](Final_Report.docx) for the full analysis, training curves, and screenshots.

## Limitations and future work

- TH True Milk bottles are sometimes misdetected in poor lighting.
- Each script handles one stage. They are not yet combined into a single pipeline: servo ordering uses 2D pixel distance rather than the stereo depth, and Kalman tracking follows only one target.
- The red-correction multiplier (`9`) and servo angle limits were tuned by hand for one setup.
- Next steps: combine the stages into one loop, give each target its own Kalman filter (multi-object tracking), and order visits by depth.

## Author

**Nguyen Huynh Duc**, ENG 301 – Computer Vision, Fulbright University Vietnam

## Acknowledgements

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · [OpenCV](https://opencv.org/) · [FilterPy](https://github.com/rlabbe/filterpy) · [pyFirmata](https://github.com/tino/pyFirmata)
