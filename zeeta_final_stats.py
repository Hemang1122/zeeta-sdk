import os
import json
import pandas as pd

base_path = "./generated_dataset/OmniBotics_V1_Reaching_2000_Episodes"
episodes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

total_frames = 0
for ep in episodes[:10]: # Check first 10 for average
    df = pd.read_parquet(os.path.join(base_path, ep, "data.parquet"))
    total_frames += len(df)

avg_frames = total_frames / 10
est_total_frames = avg_frames * len(episodes)

print(f"\n--- ZEETA DATA FACTORY: FINAL REPORT ---")
print(f"Total Episodes Generated: {len(episodes)}")
print(f"Estimated Total Frames:  {int(est_total_frames)}")
print(f"Dataset Location:        {os.path.abspath(base_path)}")
print(f"Status:                  READY FOR TRAINING")
print("----------------------------------------")
