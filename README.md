# Zeeta SDK

> **Synthetic dataset generation SDK for robotic manipulation** — physics-based simulation with domain randomization, exporting LeRobot-compatible datasets ready for behavior-cloning and diffusion-policy training.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyBullet](https://img.shields.io/badge/PyBullet-3.2+-orange.svg)](https://pybullet.org/)
[![License](https://img.shields.io/badge/License-Personal-lightgrey.svg)](#license)

---

## What is Zeeta?

Zeeta is a Python SDK for generating large-scale synthetic robot manipulation datasets. It runs PyBullet-based grasping episodes with domain randomization, captures multi-modal observations (RGB, depth, joint states, end-effector poses), and exports the result in [LeRobot](https://github.com/huggingface/lerobot) format for downstream training.

Built solo as a research-grade tool for sim-to-real robot learning workflows.

---

## ✨ Highlights

- 🤖 **Real robot model** — Franka Panda / KUKA iiwa via URDF, full inverse kinematics
- 🎲 **Domain randomization** — lighting, materials, textures, object poses, camera positions
- 🦾 **Realistic grasping** — proximity-based constraint welding for stable pick-and-place
- 📦 **YCB object suite** — production-grade benchmark objects
- 📊 **LeRobot-compatible export** — joint angles, EE poses (xyz + quaternion, 100×7), gripper states, per-episode camera calibration
- 🏃 **Headless simulation** — runs in DIRECT mode, no GUI required, scales to 2000+ episodes
- 📈 **Per-episode statistics** — diversity metrics, grasp success verification

---

## Architecture

\\\
zeeta-sdk/
│
├── zeeta_pipeline.py        → Top-level orchestrator (2000-episode generator)
├── zeeta_physics.py         → PyBullet physics setup & grasp constraints
├── zeeta_render.py          → Multi-camera RGB/depth capture
├── zeeta_ycb.py             → YCB object loading and randomization
├── zeeta_textures.py        → Texture & material randomization
├── zeeta_domains.py         → Domain randomization parameter sampler
├── zeeta_dataset.py         → Per-episode recorder + dataset writer
├── zeeta_lerobot.py         → LeRobot format exporter (parquet + videos)
├── zeeta_verify_diversity.py → Dataset diversity audit
├── zeeta_final_stats.py     → Post-generation statistics
│
└── zeeta/                   → Reusable SDK package
    ├── zeeta_ik.py          → Inverse-kinematics solver
    ├── zeeta_robot.py       → Robot abstraction layer
    ├── zeeta_physics.py     → Core physics utilities
    ├── zeeta_render.py      → Rendering primitives
    ├── zeeta_textures.py    → Texture management
    └── zeeta_ycb.py         → YCB object utilities
\\\

---

## ⚡ Tech Stack

| Layer       | Technology                         |
|-------------|------------------------------------|
| Simulation  | PyBullet (DIRECT mode)             |
| Robot       | Franka Panda / KUKA iiwa (URDF)    |
| Imaging     | OpenCV, NumPy                      |
| Storage     | Zarr, HDF5, Parquet (via pandas)   |
| Export      | LeRobot dataset format             |
| Language    | Python 3.10+                       |

---

## 🛠️ Quick Start

\\\ash
# Clone the repo
git clone https://github.com/Hemang1122/zeeta-sdk.git
cd zeeta-sdk

# Set up environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run a small test pipeline (modify EPISODES/FRAMES in zeeta_pipeline.py)
python zeeta_pipeline.py
\\\

---

## 📊 Dataset Output

Each episode produces:

- **\gb/\** — RGB images per frame (multi-camera)
- **\depth/\** — depth maps per frame
- **\ctions\** — joint deltas (T × 7)
- **\joint_states\** — joint positions (T × 8, with gripper)
- **\ee_poses\** — end-effector poses (T × 7, xyz + quaternion)
- **\domain_randomization_params\** — per-episode randomization log
- **\info.json\** — robot type, state/action shapes, episode metadata
- **\stats.json\** — LeRobot normalization statistics

Loads cleanly with \LeRobotDataset(...)\ from the LeRobot library.

---

## 🔬 Domain Randomization

Zeeta randomizes per-episode:

- Lighting (intensity, color temperature, direction)
- Object placement (XY translation, rotation)
- Camera pose (azimuth, elevation, distance)
- Object material (color, roughness, metallic)
- Background textures
- Table textures

The full parameter set per episode is logged for reproducibility and diversity analysis.

---

## 📐 Verified Behavior

The pipeline has been validated end-to-end:

- 2000 episodes × 100 frames generated cleanly
- Grasp physics produces ~65–75% verified success rate (proximity-based constraint welding)
- Output verified loadable via LeRobot \LeRobotDataset()\ without errors
- Per-episode diversity audit confirms broad coverage

---

## 🧠 Design Notes

- **Headless by default** — \p.connect(p.DIRECT)\, no GUI overhead
- **Composable modules** — physics, rendering, randomization, dataset I/O are separate concerns
- **LeRobot-first** — designed to plug directly into Hugging Face's robot-learning ecosystem

---

## 👨‍💻 Built By

**Hemang Tripathi** — CS Student @ SRH Munich
[LinkedIn](https://www.linkedin.com/in/hemang-tripathi-b2979339a/) · [GitHub](https://github.com/Hemang1122)

---

## License

This project is shared publicly as a portfolio piece. All rights reserved unless explicitly licensed otherwise.
