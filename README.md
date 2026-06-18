# Zeeta SDK

A Python SDK for generating synthetic robot manipulation datasets in PyBullet, with domain randomization and LeRobot-compatible export.

Built as a tool for sim-to-real workflows on the Franka Panda and KUKA iiwa.

---

## What it does

Runs grasping episodes in PyBullet, captures multi-modal observations (RGB, depth, joint states, end-effector poses), randomizes the scene per episode, and writes the result in LeRobot dataset format — ready to load with `LeRobotDataset(...)` for behavior cloning or diffusion policy training.

Verified end-to-end on 2000 episodes × 100 frames.

---

## Features

- **Robot:** Franka Panda or KUKA iiwa via URDF, with a full IK solver
- **Grasping:** proximity-based constraint welding (≈65–75% verified success rate)
- **Objects:** YCB benchmark set, randomized per episode
- **Domain randomization:** lighting, material, texture, object pose, camera pose
- **Export:** LeRobot format — joint angles, EE poses (xyz + quaternion, 100×7), gripper states, per-episode camera calibration, `info.json`, `stats.json`
- **Headless:** runs in PyBullet DIRECT mode, no GUI

---

## Architecture

---

## Stack

PyBullet · NumPy · OpenCV · Zarr · HDF5 · pandas · LeRobot dataset format · Python 3.10+

---

## Run it

```bash
git clone https://github.com/Hemang1122/zeeta-sdk.git
cd zeeta-sdk

python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
python zeeta_pipeline.py
```

Edit `EPISODES` and `FRAMES` in `zeeta_pipeline.py` to scale up or down. Output goes to `./generated_dataset/`.

---

## Dataset output per episode

| Field                          | Shape / Type             |
|--------------------------------|--------------------------|
| RGB frames                     | per-camera image stack   |
| Depth frames                   | per-camera depth stack   |
| Joint states                   | T × 8 (with gripper)     |
| End-effector poses             | T × 7 (xyz + quaternion) |
| Actions                        | T × 7 (joint deltas)     |
| Domain randomization params    | per-episode log          |
| `info.json`                    | robot type, shapes       |
| `stats.json`                   | LeRobot normalization    |

Loads with `LeRobotDataset(repo_id_or_path)` without modification.

---

## Domain randomization

Per episode, the scene varies along:

- Lighting — intensity, color temperature, direction
- Object placement — XY translation and rotation
- Camera pose — azimuth, elevation, distance
- Material — color, roughness, metallic
- Background and table textures

All sampled values are logged for reproducibility and diversity auditing.

---

## Author

Hemang Tripathi — CS @ SRH Munich
[LinkedIn](https://www.linkedin.com/in/hemang-tripathi-b2979339a/) · [GitHub](https://github.com/Hemang1122)