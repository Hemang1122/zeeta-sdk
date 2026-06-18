import os, json, shutil, zarr
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
from typing import Dict, Any

class EpisodeRecorder:
    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.frames = {"rgb": [], "depth": [], "actions": [], "joint_states": [], "domain_randomization_params": []}
        self.metadata = {}

    def record_frame(self, rgb, depth, action, joint_state, dr_params):
        self.frames["rgb"].append(rgb)
        self.frames["depth"].append(depth)
        self.frames["actions"].append(action)
        self.frames["joint_states"].append(joint_state)
        self.frames["domain_randomization_params"].append(dr_params)

    def set_metadata(self, key, value): self.metadata[key] = value
    def get_episode_data(self): return {"episode_id": self.episode_id, "frames": self.frames, "metadata": self.metadata}

class DatasetWriter:
    def __init__(self, output_dir, dataset_name):
        self.output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.episode_count = 0

    def write_episode(self, episode_data):
        ep_path = os.path.join(self.output_dir, episode_data["episode_id"])
        os.makedirs(ep_path, exist_ok=True)
        
        # RGB Video
        rgb = episode_data["frames"]["rgb"]
        if rgb:
            h, w, _ = rgb[0].shape
            out = cv2.VideoWriter(os.path.join(ep_path, "rgb.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
            for f in rgb: out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()

        # Depth Zarr (with overwrite fix)
        z_path = os.path.join(ep_path, "depth.zarr")
        if os.path.exists(z_path): shutil.rmtree(z_path)
        zarr.save(z_path, np.array(episode_data["frames"]["depth"]))

        # Data Parquet
        pd.DataFrame({k: episode_data["frames"][k] for k in ["actions", "joint_states", "domain_randomization_params"]}).to_parquet(os.path.join(ep_path, "data.parquet"))
        
        with open(os.path.join(ep_path, "metadata.json"), "w") as f: json.dump(episode_data["metadata"], f, indent=4)
        self.episode_count += 1

    def generate_info_json(self):
        info = {"dataset_name": "zeeta_data", "episode_count": self.episode_count, "created_at": datetime.now().isoformat()}
        with open(os.path.join(self.output_dir, "info.json"), "w") as f: json.dump(info, f, indent=4)
