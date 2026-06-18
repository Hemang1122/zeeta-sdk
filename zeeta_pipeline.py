import pybullet as p
import pybullet_data
import numpy as np
import os, random, time
from datetime import datetime
from zeeta_render import ZeetaRenderer
from zeeta_dataset import EpisodeRecorder, DatasetWriter
from zeeta_ycb import ZeetaYCB
from zeeta_textures import ZeetaTextures
from zeeta_physics import ZeetaPhysics

OUTPUT_DIR, DATASET_NAME = "./generated_dataset", "OmniBotics_V2_Fixed"
EPISODES, FRAMES = 2000, 100
RENDER_PROFILE = "aggressive"

def setup_env():
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    try: robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0.05])
    except: robotId = p.loadURDF("kuka_lbr_iiwa/model.urdf", [0, 0, 0.05])
    return ZeetaPhysics(client), robotId

def main():
    physics, robotId = setup_env()
    ycb, tex, renderer = ZeetaYCB(physics), ZeetaTextures(physics), ZeetaRenderer()
    writer = DatasetWriter(OUTPUT_DIR, DATASET_NAME)

    for ep_idx in range(EPISODES):
        # 1. Reset simulation every 50 episodes for memory
        if ep_idx > 0 and ep_idx % 50 == 0:
            p.disconnect(physics.client)
            physics, robotId = setup_env()
            ycb, tex = ZeetaYCB(physics), ZeetaTextures(physics)

        # 2. Setup UNIQUE episode
        ep_seed = int(time.time() + ep_idx)
        random.seed(ep_seed) # Seed the GLOBAL random for physics/objects
        renderer.set_episode_style(ep_seed, RENDER_PROFILE) # Seed the RENDER style
        
        ep_id = f"ep_{ep_idx:04d}_{datetime.now().strftime('%H%M%S')}"
        recorder = EpisodeRecorder(ep_id)
        ycb.clear()
        obj_ids = ycb.spawn_scene(3, [0.5, 0], [0.4, 0.3], 0.1)
        
        # 3. Frame Loop
        for f_idx in range(FRAMES):
            p.stepSimulation()
            _, _, rgb, depth, _ = p.getCameraImage(224, 224)
            rgb = np.reshape(rgb, (224, 224, 4))[:,:,:3]
            
            # Apply style + frame-specific noise
            processed_rgb = renderer.render_rgb(rgb, frame_seed=ep_seed + f_idx, profile_name=RENDER_PROFILE)
            
            recorder.record_frame(processed_rgb, depth, [0.0]*7, [0.0]*7, {"ep": ep_idx})

        writer.write_episode(recorder.get_episode_data())
        if (ep_idx + 1) % 10 == 0: print(f"✅ {ep_idx+1}/{EPISODES} Episodes complete...")

    writer.generate_info_json()
    p.disconnect()

if __name__ == "__main__": main()
