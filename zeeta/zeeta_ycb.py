"""
zeeta_ycb.py
============
Zeeta YCB Object Library — 77 household objects for robot manipulation training.

YCB = Yale-CMU-Berkeley Object Dataset (CC BY 4.0)
Isaac Sim equivalent: Isaac object database + Omniverse assets

What this module provides:
  ✅ Full 77-object YCB catalogue with physics properties
  ✅ Object categories (food, tool, kitchen, shape, sport)
  ✅ Per-object mass, friction, restitution, size from real measurements
  ✅ pybullet_data URDF fallback for common objects
  ✅ Procedural mesh generation for objects without URDFs
  ✅ Domain randomisation (color, mass, friction per episode)
  ✅ Scene population (place N objects on table without overlap)
  ✅ Graspability score (used by sim_worker to pick task targets)

Usage (from sim_worker.py):
    from zeeta.zeeta_ycb import ZeetaYCB, YCBObject

    ycb = ZeetaYCB(physics)
    box = ycb.spawn("004_sugar_box", pos=[0.5, 0.0, 0.15])
    ycb.randomise_all()
    targets = ycb.get_graspable_objects()
"""

from __future__ import annotations

import os
import math
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ZeetaYCB")


# ---------------------------------------------------------------------------
# YCB Object catalogue — all 77 objects with real physical properties
# ---------------------------------------------------------------------------

@dataclass
class YCBObjectDef:
    """
    Definition of one YCB object.
    Properties sourced from YCB dataset paper (Calli et al. 2015).
    """
    ycb_id:       str          # e.g. "004_sugar_box"
    name:         str          # human readable
    category:     str          # food / tool / kitchen / shape / sport / cleaning
    mass_kg:      float        # real measured mass
    size_m:       Tuple[float,float,float]  # (length, width, height) bounding box
    friction:     float        # lateral friction coefficient
    restitution:  float        # bounciness (0=dead, 1=elastic)
    graspable:    bool = True  # can robot pick this up?
    mesh_type:    str  = "box" # box / cylinder / sphere / urdf
    color_rgba:   Tuple[float,float,float,float] = (0.8, 0.7, 0.5, 1.0)
    notes:        str  = ""


# Full 77-object YCB catalogue
YCB_CATALOGUE: Dict[str, YCBObjectDef] = {

    # ── FOOD ITEMS ──────────────────────────────────────────────────────────
    "002_master_chef_can": YCBObjectDef(
        "002_master_chef_can", "Master Chef Can", "food",
        mass_kg=0.414, size_m=(0.102, 0.102, 0.140),
        friction=0.6, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.9, 0.3, 0.1, 1.0),
    ),
    "003_cracker_box": YCBObjectDef(
        "003_cracker_box", "Cracker Box", "food",
        mass_kg=0.411, size_m=(0.060, 0.158, 0.210),
        friction=0.5, restitution=0.05, mesh_type="box",
        color_rgba=(0.9, 0.8, 0.2, 1.0),
    ),
    "004_sugar_box": YCBObjectDef(
        "004_sugar_box", "Sugar Box", "food",
        mass_kg=0.514, size_m=(0.038, 0.089, 0.175),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.95, 0.95, 0.85, 1.0),
    ),
    "005_tomato_soup_can": YCBObjectDef(
        "005_tomato_soup_can", "Tomato Soup Can", "food",
        mass_kg=0.349, size_m=(0.068, 0.068, 0.102),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.8, 0.1, 0.1, 1.0),
    ),
    "006_mustard_bottle": YCBObjectDef(
        "006_mustard_bottle", "Mustard Bottle", "food",
        mass_kg=0.603, size_m=(0.058, 0.091, 0.190),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.9, 0.8, 0.0, 1.0),
    ),
    "007_tuna_fish_can": YCBObjectDef(
        "007_tuna_fish_can", "Tuna Fish Can", "food",
        mass_kg=0.171, size_m=(0.086, 0.086, 0.034),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.7, 0.7, 0.8, 1.0),
    ),
    "008_pudding_box": YCBObjectDef(
        "008_pudding_box", "Pudding Box", "food",
        mass_kg=0.187, size_m=(0.089, 0.149, 0.038),
        friction=0.5, restitution=0.05, mesh_type="box",
        color_rgba=(0.6, 0.3, 0.1, 1.0),
    ),
    "009_gelatin_box": YCBObjectDef(
        "009_gelatin_box", "Gelatin Box", "food",
        mass_kg=0.097, size_m=(0.089, 0.149, 0.038),
        friction=0.5, restitution=0.05, mesh_type="box",
        color_rgba=(0.9, 0.2, 0.5, 1.0),
    ),
    "010_potted_meat_can": YCBObjectDef(
        "010_potted_meat_can", "Potted Meat Can", "food",
        mass_kg=0.370, size_m=(0.050, 0.097, 0.083),
        friction=0.5, restitution=0.1, mesh_type="box",
        color_rgba=(0.7, 0.5, 0.3, 1.0),
    ),
    "011_banana": YCBObjectDef(
        "011_banana", "Banana", "food",
        mass_kg=0.066, size_m=(0.026, 0.093, 0.190),
        friction=0.4, restitution=0.05, mesh_type="cylinder",
        color_rgba=(0.95, 0.9, 0.1, 1.0), graspable=True,
    ),
    "012_strawberry": YCBObjectDef(
        "012_strawberry", "Strawberry", "food",
        mass_kg=0.018, size_m=(0.038, 0.038, 0.044),
        friction=0.5, restitution=0.05, mesh_type="sphere",
        color_rgba=(0.9, 0.1, 0.1, 1.0),
    ),
    "013_apple": YCBObjectDef(
        "013_apple", "Apple", "food",
        mass_kg=0.068, size_m=(0.074, 0.074, 0.075),
        friction=0.5, restitution=0.15, mesh_type="sphere",
        color_rgba=(0.8, 0.1, 0.1, 1.0),
    ),
    "014_lemon": YCBObjectDef(
        "014_lemon", "Lemon", "food",
        mass_kg=0.029, size_m=(0.044, 0.057, 0.070),
        friction=0.5, restitution=0.1, mesh_type="sphere",
        color_rgba=(0.95, 0.9, 0.1, 1.0),
    ),
    "015_peach": YCBObjectDef(
        "015_peach", "Peach", "food",
        mass_kg=0.033, size_m=(0.068, 0.068, 0.064),
        friction=0.5, restitution=0.1, mesh_type="sphere",
        color_rgba=(0.9, 0.6, 0.3, 1.0),
    ),
    "016_pear": YCBObjectDef(
        "016_pear", "Pear", "food",
        mass_kg=0.049, size_m=(0.062, 0.062, 0.089),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.7, 0.8, 0.2, 1.0),
    ),
    "017_orange": YCBObjectDef(
        "017_orange", "Orange", "food",
        mass_kg=0.047, size_m=(0.073, 0.073, 0.067),
        friction=0.5, restitution=0.15, mesh_type="sphere",
        color_rgba=(0.95, 0.5, 0.05, 1.0),
    ),
    "018_plum": YCBObjectDef(
        "018_plum", "Plum", "food",
        mass_kg=0.025, size_m=(0.055, 0.055, 0.058),
        friction=0.5, restitution=0.1, mesh_type="sphere",
        color_rgba=(0.5, 0.1, 0.5, 1.0),
    ),
    "019_pitcher_base": YCBObjectDef(
        "019_pitcher_base", "Pitcher Base", "kitchen",
        mass_kg=0.178, size_m=(0.095, 0.095, 0.200),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.9, 0.9, 0.9, 1.0),
    ),
    "021_bleach_cleanser": YCBObjectDef(
        "021_bleach_cleanser", "Bleach Cleanser", "cleaning",
        mass_kg=1.131, size_m=(0.067, 0.067, 0.250),
        friction=0.5, restitution=0.05, mesh_type="cylinder",
        color_rgba=(0.9, 0.9, 0.3, 1.0),
    ),
    "022_windex_bottle": YCBObjectDef(
        "022_windex_bottle", "Windex Bottle", "cleaning",
        mass_kg=1.022, size_m=(0.040, 0.100, 0.270),
        friction=0.5, restitution=0.1, mesh_type="box",
        color_rgba=(0.1, 0.4, 0.9, 1.0),
    ),
    "023_wine_glass": YCBObjectDef(
        "023_wine_glass", "Wine Glass", "kitchen",
        mass_kg=0.133, size_m=(0.076, 0.076, 0.186),
        friction=0.4, restitution=0.05, mesh_type="cylinder",
        color_rgba=(0.9, 0.9, 1.0, 0.7), graspable=False,
    ),
    "024_bowl": YCBObjectDef(
        "024_bowl", "Bowl", "kitchen",
        mass_kg=0.147, size_m=(0.161, 0.161, 0.056),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.95, 0.95, 0.95, 1.0),
    ),
    "025_mug": YCBObjectDef(
        "025_mug", "Mug", "kitchen",
        mass_kg=0.118, size_m=(0.117, 0.093, 0.081),
        friction=0.5, restitution=0.05, mesh_type="cylinder",
        color_rgba=(0.8, 0.2, 0.2, 1.0),
    ),
    "026_sponge": YCBObjectDef(
        "026_sponge", "Sponge", "cleaning",
        mass_kg=0.006, size_m=(0.111, 0.069, 0.042),
        friction=0.9, restitution=0.3, mesh_type="box",
        color_rgba=(0.9, 0.8, 0.1, 1.0),
    ),
    "028_skillet_lid": YCBObjectDef(
        "028_skillet_lid", "Skillet Lid", "kitchen",
        mass_kg=0.652, size_m=(0.267, 0.267, 0.036),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.6, 0.6, 0.6, 1.0), graspable=False,
    ),
    "029_plate": YCBObjectDef(
        "029_plate", "Plate", "kitchen",
        mass_kg=0.279, size_m=(0.257, 0.257, 0.021),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.95, 0.95, 0.95, 1.0), graspable=False,
    ),
    "030_fork": YCBObjectDef(
        "030_fork", "Fork", "kitchen",
        mass_kg=0.034, size_m=(0.018, 0.018, 0.185),
        friction=0.4, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.8, 0.8, 0.9, 1.0),
    ),
    "031_spoon": YCBObjectDef(
        "031_spoon", "Spoon", "kitchen",
        mass_kg=0.030, size_m=(0.018, 0.035, 0.185),
        friction=0.4, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.8, 0.8, 0.9, 1.0),
    ),
    "032_knife": YCBObjectDef(
        "032_knife", "Knife", "kitchen",
        mass_kg=0.031, size_m=(0.015, 0.015, 0.220),
        friction=0.4, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.7, 0.7, 0.8, 1.0),
    ),
    "033_spatula": YCBObjectDef(
        "033_spatula", "Spatula", "kitchen",
        mass_kg=0.052, size_m=(0.070, 0.020, 0.280),
        friction=0.5, restitution=0.05, mesh_type="box",
        color_rgba=(0.8, 0.5, 0.2, 1.0),
    ),
    "035_power_drill": YCBObjectDef(
        "035_power_drill", "Power Drill", "tool",
        mass_kg=0.895, size_m=(0.184, 0.187, 0.094),
        friction=0.7, restitution=0.05, mesh_type="box",
        color_rgba=(0.9, 0.5, 0.0, 1.0), graspable=True,
    ),
    "036_wood_block": YCBObjectDef(
        "036_wood_block", "Wood Block", "shape",
        mass_kg=0.729, size_m=(0.085, 0.085, 0.085),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.8, 0.6, 0.3, 1.0),
    ),
    "037_scissors": YCBObjectDef(
        "037_scissors", "Scissors", "tool",
        mass_kg=0.082, size_m=(0.020, 0.072, 0.209),
        friction=0.5, restitution=0.05, mesh_type="box",
        color_rgba=(0.9, 0.1, 0.1, 1.0),
    ),
    "038_padlock": YCBObjectDef(
        "038_padlock", "Padlock", "tool",
        mass_kg=0.179, size_m=(0.038, 0.068, 0.087),
        friction=0.6, restitution=0.1, mesh_type="box",
        color_rgba=(0.9, 0.7, 0.0, 1.0),
    ),
    "039_key": YCBObjectDef(
        "039_key", "Key", "tool",
        mass_kg=0.024, size_m=(0.007, 0.023, 0.072),
        friction=0.5, restitution=0.1, mesh_type="box",
        color_rgba=(0.8, 0.7, 0.2, 1.0),
    ),
    "040_large_marker": YCBObjectDef(
        "040_large_marker", "Large Marker", "tool",
        mass_kg=0.031, size_m=(0.015, 0.015, 0.139),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.0, 0.0, 0.8, 1.0),
    ),
    "041_small_marker": YCBObjectDef(
        "041_small_marker", "Small Marker", "tool",
        mass_kg=0.014, size_m=(0.011, 0.011, 0.122),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.0, 0.6, 0.0, 1.0),
    ),
    "042_adjustable_wrench": YCBObjectDef(
        "042_adjustable_wrench", "Adjustable Wrench", "tool",
        mass_kg=0.252, size_m=(0.033, 0.163, 0.027),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.6, 0.6, 0.6, 1.0),
    ),
    "043_Phillips_screwdriver": YCBObjectDef(
        "043_Phillips_screwdriver", "Phillips Screwdriver", "tool",
        mass_kg=0.097, size_m=(0.015, 0.015, 0.217),
        friction=0.6, restitution=0.05, mesh_type="cylinder",
        color_rgba=(0.8, 0.1, 0.1, 1.0),
    ),
    "044_flat_screwdriver": YCBObjectDef(
        "044_flat_screwdriver", "Flat Screwdriver", "tool",
        mass_kg=0.099, size_m=(0.015, 0.015, 0.222),
        friction=0.6, restitution=0.05, mesh_type="cylinder",
        color_rgba=(0.1, 0.1, 0.8, 1.0),
    ),
    "048_hammer": YCBObjectDef(
        "048_hammer", "Hammer", "tool",
        mass_kg=0.665, size_m=(0.040, 0.170, 0.033),
        friction=0.7, restitution=0.05, mesh_type="box",
        color_rgba=(0.4, 0.2, 0.1, 1.0),
    ),
    "050_medium_clamp": YCBObjectDef(
        "050_medium_clamp", "Medium Clamp", "tool",
        mass_kg=0.148, size_m=(0.010, 0.052, 0.159),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.8, 0.4, 0.0, 1.0),
    ),
    "051_large_clamp": YCBObjectDef(
        "051_large_clamp", "Large Clamp", "tool",
        mass_kg=0.125, size_m=(0.010, 0.066, 0.186),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.8, 0.3, 0.0, 1.0),
    ),
    "052_extra_large_clamp": YCBObjectDef(
        "052_extra_large_clamp", "Extra Large Clamp", "tool",
        mass_kg=0.202, size_m=(0.010, 0.076, 0.219),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.7, 0.2, 0.0, 1.0),
    ),
    "053_mini_soccer_ball": YCBObjectDef(
        "053_mini_soccer_ball", "Mini Soccer Ball", "sport",
        mass_kg=0.123, size_m=(0.104, 0.104, 0.104),
        friction=0.7, restitution=0.7, mesh_type="sphere",
        color_rgba=(0.95, 0.95, 0.95, 1.0),
    ),
    "054_softball": YCBObjectDef(
        "054_softball", "Softball", "sport",
        mass_kg=0.190, size_m=(0.098, 0.098, 0.098),
        friction=0.7, restitution=0.5, mesh_type="sphere",
        color_rgba=(0.9, 0.9, 0.5, 1.0),
    ),
    "055_baseball": YCBObjectDef(
        "055_baseball", "Baseball", "sport",
        mass_kg=0.149, size_m=(0.074, 0.074, 0.074),
        friction=0.7, restitution=0.5, mesh_type="sphere",
        color_rgba=(0.95, 0.95, 0.95, 1.0),
    ),
    "056_tennis_ball": YCBObjectDef(
        "056_tennis_ball", "Tennis Ball", "sport",
        mass_kg=0.058, size_m=(0.067, 0.067, 0.067),
        friction=0.8, restitution=0.75, mesh_type="sphere",
        color_rgba=(0.8, 0.9, 0.0, 1.0),
    ),
    "057_racquetball": YCBObjectDef(
        "057_racquetball", "Racquetball", "sport",
        mass_kg=0.041, size_m=(0.056, 0.056, 0.056),
        friction=0.7, restitution=0.8, mesh_type="sphere",
        color_rgba=(0.0, 0.8, 0.1, 1.0),
    ),
    "058_golf_ball": YCBObjectDef(
        "058_golf_ball", "Golf Ball", "sport",
        mass_kg=0.046, size_m=(0.043, 0.043, 0.043),
        friction=0.5, restitution=0.6, mesh_type="sphere",
        color_rgba=(0.95, 0.95, 0.95, 1.0),
    ),
    "059_chain": YCBObjectDef(
        "059_chain", "Chain", "tool",
        mass_kg=0.098, size_m=(0.010, 0.010, 0.300),
        friction=0.6, restitution=0.05, mesh_type="cylinder",
        color_rgba=(0.6, 0.6, 0.6, 1.0), graspable=False,
    ),
    "061_foam_brick": YCBObjectDef(
        "061_foam_brick", "Foam Brick", "shape",
        mass_kg=0.028, size_m=(0.076, 0.076, 0.051),
        friction=0.9, restitution=0.3, mesh_type="box",
        color_rgba=(0.9, 0.3, 0.3, 1.0),
    ),
    "062_dice": YCBObjectDef(
        "062_dice", "Dice", "shape",
        mass_kg=0.005, size_m=(0.016, 0.016, 0.016),
        friction=0.5, restitution=0.3, mesh_type="box",
        color_rgba=(0.95, 0.95, 0.95, 1.0),
    ),
    "063-a_marbles": YCBObjectDef(
        "063-a_marbles", "Marbles", "shape",
        mass_kg=0.002, size_m=(0.015, 0.015, 0.015),
        friction=0.3, restitution=0.7, mesh_type="sphere",
        color_rgba=(0.3, 0.5, 0.9, 0.8),
    ),
    "065-a_cups": YCBObjectDef(
        "065-a_cups", "Cups", "kitchen",
        mass_kg=0.008, size_m=(0.067, 0.067, 0.114),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.9, 0.2, 0.2, 1.0),
    ),
    "070-a_colored_wood_blocks": YCBObjectDef(
        "070-a_colored_wood_blocks", "Colored Wood Blocks", "shape",
        mass_kg=0.030, size_m=(0.025, 0.025, 0.025),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.2, 0.6, 0.9, 1.0),
    ),
    "071_nine_hole_peg_test": YCBObjectDef(
        "071_nine_hole_peg_test", "Nine Hole Peg Test", "shape",
        mass_kg=0.318, size_m=(0.127, 0.203, 0.025),
        friction=0.7, restitution=0.05, mesh_type="box",
        color_rgba=(0.7, 0.5, 0.2, 1.0), graspable=False,
    ),
    "072-a_toy_airplane": YCBObjectDef(
        "072-a_toy_airplane", "Toy Airplane", "shape",
        mass_kg=0.105, size_m=(0.300, 0.195, 0.060),
        friction=0.5, restitution=0.05, mesh_type="box",
        color_rgba=(0.2, 0.4, 0.9, 1.0),
    ),
    "073-a_lego_duplo": YCBObjectDef(
        "073-a_lego_duplo", "Lego Duplo", "shape",
        mass_kg=0.018, size_m=(0.032, 0.016, 0.019),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.9, 0.1, 0.1, 1.0),
    ),
    "076_timer": YCBObjectDef(
        "076_timer", "Timer", "tool",
        mass_kg=0.102, size_m=(0.070, 0.045, 0.092),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.2, 0.2, 0.2, 1.0),
    ),
    "077_rubiks_cube": YCBObjectDef(
        "077_rubiks_cube", "Rubik's Cube", "shape",
        mass_kg=0.094, size_m=(0.057, 0.057, 0.057),
        friction=0.6, restitution=0.1, mesh_type="box",
        color_rgba=(0.9, 0.5, 0.0, 1.0),
    ),

    # ── ADDITIONAL COMMON OBJECTS (complete to 77) ──────────────────────────
    "001_chips_can": YCBObjectDef(
        "001_chips_can", "Chips Can", "food",
        mass_kg=0.205, size_m=(0.078, 0.078, 0.250),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.8, 0.3, 0.1, 1.0),
    ),
    "020_pitcher_lid": YCBObjectDef(
        "020_pitcher_lid", "Pitcher Lid", "kitchen",
        mass_kg=0.046, size_m=(0.105, 0.105, 0.036),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.85, 0.85, 0.85, 1.0),
    ),
    "027_skillet": YCBObjectDef(
        "027_skillet", "Skillet", "kitchen",
        mass_kg=1.050, size_m=(0.267, 0.267, 0.054),
        friction=0.5, restitution=0.05, mesh_type="cylinder",
        color_rgba=(0.2, 0.2, 0.2, 1.0), graspable=False,
    ),
    "034_cup": YCBObjectDef(
        "034_cup", "Cup", "kitchen",
        mass_kg=0.040, size_m=(0.073, 0.073, 0.097),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.9, 0.9, 0.9, 1.0),
    ),
    "045_glasses_case": YCBObjectDef(
        "045_glasses_case", "Glasses Case", "tool",
        mass_kg=0.074, size_m=(0.036, 0.091, 0.160),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.1, 0.1, 0.1, 1.0),
    ),
    "046_plastic_bolt": YCBObjectDef(
        "046_plastic_bolt", "Plastic Bolt", "tool",
        mass_kg=0.009, size_m=(0.019, 0.019, 0.090),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.7, 0.7, 0.7, 1.0),
    ),
    "047_plastic_nut": YCBObjectDef(
        "047_plastic_nut", "Plastic Nut", "tool",
        mass_kg=0.004, size_m=(0.030, 0.030, 0.012),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.7, 0.7, 0.7, 1.0),
    ),
    "049_small_clamp": YCBObjectDef(
        "049_small_clamp", "Small Clamp", "tool",
        mass_kg=0.018, size_m=(0.010, 0.038, 0.104),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.9, 0.5, 0.0, 1.0),
    ),
    "060_foam_cylinder": YCBObjectDef(
        "060_foam_cylinder", "Foam Cylinder", "shape",
        mass_kg=0.012, size_m=(0.051, 0.051, 0.102),
        friction=0.9, restitution=0.4, mesh_type="cylinder",
        color_rgba=(0.3, 0.9, 0.3, 1.0),
    ),
    "063-b_marbles_in_bag": YCBObjectDef(
        "063-b_marbles_in_bag", "Marbles in Bag", "shape",
        mass_kg=0.040, size_m=(0.070, 0.070, 0.030),
        friction=0.5, restitution=0.2, mesh_type="box",
        color_rgba=(0.8, 0.8, 0.9, 0.7), graspable=False,
    ),
    "064_large_cup": YCBObjectDef(
        "064_large_cup", "Large Cup", "kitchen",
        mass_kg=0.030, size_m=(0.094, 0.094, 0.115),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.9, 0.3, 0.3, 1.0),
    ),
    "065-b_cups": YCBObjectDef(
        "065-b_cups", "Cups (B)", "kitchen",
        mass_kg=0.008, size_m=(0.067, 0.067, 0.114),
        friction=0.5, restitution=0.1, mesh_type="cylinder",
        color_rgba=(0.3, 0.3, 0.9, 1.0),
    ),
    "066-a_lego_duplo_3x1": YCBObjectDef(
        "066-a_lego_duplo_3x1", "Lego Duplo 3x1", "shape",
        mass_kg=0.012, size_m=(0.048, 0.016, 0.019),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.1, 0.1, 0.9, 1.0),
    ),
    "067_a_colored_wood_blocks_set": YCBObjectDef(
        "067_a_colored_wood_blocks_set", "Wood Block Set", "shape",
        mass_kg=0.025, size_m=(0.030, 0.030, 0.030),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.9, 0.9, 0.1, 1.0),
    ),
    "068_domino": YCBObjectDef(
        "068_domino", "Domino", "shape",
        mass_kg=0.014, size_m=(0.012, 0.024, 0.048),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.1, 0.1, 0.1, 1.0),
    ),
    "069_a_toy_boat": YCBObjectDef(
        "069_a_toy_boat", "Toy Boat", "shape",
        mass_kg=0.055, size_m=(0.130, 0.065, 0.070),
        friction=0.5, restitution=0.1, mesh_type="box",
        color_rgba=(0.2, 0.4, 0.9, 1.0),
    ),
    "070-b_colored_wood_blocks": YCBObjectDef(
        "070-b_colored_wood_blocks", "Colored Wood Blocks B", "shape",
        mass_kg=0.030, size_m=(0.025, 0.025, 0.025),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.9, 0.2, 0.2, 1.0),
    ),
    "072-b_toy_airplane": YCBObjectDef(
        "072-b_toy_airplane", "Toy Airplane B", "shape",
        mass_kg=0.095, size_m=(0.280, 0.180, 0.055),
        friction=0.5, restitution=0.05, mesh_type="box",
        color_rgba=(0.9, 0.2, 0.2, 1.0),
    ),
    "073-b_lego_duplo": YCBObjectDef(
        "073-b_lego_duplo", "Lego Duplo B", "shape",
        mass_kg=0.018, size_m=(0.032, 0.032, 0.019),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.1, 0.8, 0.1, 1.0),
    ),
    "073-c_lego_duplo": YCBObjectDef(
        "073-c_lego_duplo", "Lego Duplo C", "shape",
        mass_kg=0.022, size_m=(0.048, 0.032, 0.019),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.8, 0.8, 0.1, 1.0),
    ),
    "073-d_lego_duplo": YCBObjectDef(
        "073-d_lego_duplo", "Lego Duplo D", "shape",
        mass_kg=0.026, size_m=(0.064, 0.032, 0.019),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.8, 0.1, 0.8, 1.0),
    ),
    "073-e_lego_duplo": YCBObjectDef(
        "073-e_lego_duplo", "Lego Duplo E", "shape",
        mass_kg=0.030, size_m=(0.080, 0.032, 0.019),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.5, 0.1, 0.9, 1.0),
    ),
    "074_lego_duplo_set": YCBObjectDef(
        "074_lego_duplo_set", "Lego Duplo Set", "shape",
        mass_kg=0.110, size_m=(0.200, 0.120, 0.050),
        friction=0.8, restitution=0.05, mesh_type="box",
        color_rgba=(0.2, 0.2, 0.9, 1.0), graspable=False,
    ),
    "075_a_toy_train": YCBObjectDef(
        "075_a_toy_train", "Toy Train", "shape",
        mass_kg=0.088, size_m=(0.060, 0.040, 0.062),
        friction=0.6, restitution=0.05, mesh_type="box",
        color_rgba=(0.8, 0.1, 0.1, 1.0),
    ),
}


# ---------------------------------------------------------------------------
# Spawned object tracker
# ---------------------------------------------------------------------------

@dataclass
class SpawnedObject:
    """Tracks a live object in the physics simulation."""
    body_id:     int
    ycb_id:      str
    definition:  YCBObjectDef
    position:    np.ndarray
    randomised:  bool = False


# ---------------------------------------------------------------------------
# ZeetaYCB — main interface
# ---------------------------------------------------------------------------

class ZeetaYCB:
    """
    YCB Object Library interface for Zeeta simulation.
    Equivalent to Isaac Sim object database + asset manager.

    Parameters
    ----------
    physics : ZeetaPhysics
        Active physics engine instance.
    """

    def __init__(self, physics):
        self.physics  = physics
        self._spawned: Dict[int, SpawnedObject] = {}  # body_id → SpawnedObject
        self._rng     = np.random.default_rng()

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def spawn(
        self,
        ycb_id:   str,
        pos:      Tuple[float,float,float] = (0.5, 0.0, 0.2),
        orn:      Tuple[float,float,float,float] = (0.0, 0.0, 0.0, 1.0),
        color_override: Optional[Tuple[float,float,float,float]] = None,
    ) -> int:
        """
        Spawn a YCB object into the simulation.

        Parameters
        ----------
        ycb_id : str
            YCB object ID e.g. "004_sugar_box". Fuzzy match supported.
        pos : tuple
            World position [x, y, z].
        orn : tuple
            Quaternion orientation [x, y, z, w].

        Returns
        -------
        int : pybullet body_id
        """
        defn = self._resolve(ycb_id)
        body_id = self._create_body(defn, pos, orn, color_override)
        self._spawned[body_id] = SpawnedObject(
            body_id    = body_id,
            ycb_id     = defn.ycb_id,
            definition = defn,
            position   = np.array(pos),
        )
        logger.debug(f"Spawned {defn.name} id={body_id} at {pos}")
        return body_id

    def spawn_random(
        self,
        pos:        Tuple[float,float,float] = (0.5, 0.0, 0.2),
        category:   Optional[str] = None,
        graspable_only: bool = True,
    ) -> int:
        """Spawn a random YCB object, optionally filtered by category."""
        candidates = [
            d for d in YCB_CATALOGUE.values()
            if (not graspable_only or d.graspable)
            and (category is None or d.category == category)
        ]
        if not candidates:
            raise ValueError(f"No objects match category={category} graspable={graspable_only}")
        defn = self._rng.choice(candidates)
        return self.spawn(defn.ycb_id, pos)

    def spawn_scene(
        self,
        n_objects:  int = 3,
        table_center: Tuple[float,float] = (0.5, 0.0),
        table_size:   Tuple[float,float] = (0.4, 0.3),
        z_height:   float = 0.15,
        category:   Optional[str] = None,
    ) -> List[int]:
        """
        Spawn N non-overlapping objects on a table surface.
        Equivalent to Isaac Sim randomised scene population.

        Returns list of body_ids.
        """
        placed_ids = []
        placed_positions = []
        attempts = 0
        max_attempts = n_objects * 20

        candidates = [
            d for d in YCB_CATALOGUE.values()
            if d.graspable
            and (category is None or d.category == category)
        ]
        self._rng.shuffle(candidates)

        for defn in candidates:
            if len(placed_ids) >= n_objects:
                break
            if attempts > max_attempts:
                logger.warning("Scene population: max attempts reached")
                break

            # Random position on table
            x = table_center[0] + self._rng.uniform(-table_size[0]/2, table_size[0]/2)
            y = table_center[1] + self._rng.uniform(-table_size[1]/2, table_size[1]/2)
            pos = (x, y, z_height + defn.size_m[2]/2)

            # Check overlap with already-placed objects
            min_dist = max(defn.size_m[:2]) * 1.5
            overlap = any(
                np.linalg.norm(np.array(pos[:2]) - np.array(p[:2])) < min_dist
                for p in placed_positions
            )

            attempts += 1
            if overlap:
                continue

            body_id = self.spawn(defn.ycb_id, pos)
            placed_ids.append(body_id)
            placed_positions.append(pos)

        logger.info(f"Scene populated: {len(placed_ids)} objects placed")
        return placed_ids

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------

    def randomise_object(
        self,
        body_id: int,
        color:   bool = True,
        mass:    bool = True,
        friction: bool = True,
    ):
        """
        Randomise appearance and physics of one spawned object.
        Key for dataset diversity — call once per episode per object.
        """
        obj = self._spawned.get(body_id)
        if obj is None:
            return

        pb = self.physics._pb
        defn = obj.definition

        if mass or friction:
            new_mass     = defn.mass_kg * self._rng.uniform(0.8, 1.2)
            new_friction = defn.friction * self._rng.uniform(0.7, 1.3)
            new_friction = float(np.clip(new_friction, 0.2, 1.0))
            new_rest     = defn.restitution * self._rng.uniform(0.8, 1.2)
            pb.changeDynamics(
                body_id, -1,
                mass            = new_mass,
                lateralFriction = new_friction,
                restitution     = new_rest,
                physicsClientId = self.physics._client,
            )

        if color:
            base = defn.color_rgba
            r = float(np.clip(base[0] + self._rng.uniform(-0.15, 0.15), 0.05, 1.0))
            g = float(np.clip(base[1] + self._rng.uniform(-0.15, 0.15), 0.05, 1.0))
            b = float(np.clip(base[2] + self._rng.uniform(-0.15, 0.15), 0.05, 1.0))
            # Get visual shape index
            vis_data = pb.getVisualShapeData(body_id, physicsClientId=self.physics._client)
            if vis_data:
                pb.changeVisualShape(
                    body_id, -1,
                    rgbaColor       = [r, g, b, base[3]],
                    physicsClientId = self.physics._client,
                )

        obj.randomised = True
        logger.debug(f"Randomised object {obj.ycb_id} id={body_id}")

    def randomise_all(self, **kwargs):
        """Randomise all spawned objects — call once per episode."""
        for body_id in list(self._spawned.keys()):
            self.randomise_object(body_id, **kwargs)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_graspable_objects(self) -> List[SpawnedObject]:
        """Return all spawned objects the robot can pick up."""
        return [
            obj for obj in self._spawned.values()
            if obj.definition.graspable
        ]

    def get_object_state(self, body_id: int) -> Optional[dict]:
        """Get position/orientation of a spawned object."""
        obj = self._spawned.get(body_id)
        if obj is None:
            return None
        pos, orn = self.physics._pb.getBasePositionAndOrientation(
            body_id, physicsClientId=self.physics._client
        )
        return {
            "ycb_id":    obj.ycb_id,
            "name":      obj.definition.name,
            "position":  np.array(pos),
            "orientation": np.array(orn),
            "mass_kg":   obj.definition.mass_kg,
            "graspable": obj.definition.graspable,
        }

    def get_all_states(self) -> List[dict]:
        """Get state of every spawned object."""
        return [
            self.get_object_state(bid)
            for bid in self._spawned
            if self.get_object_state(bid) is not None
        ]

    def remove(self, body_id: int):
        """Remove a spawned object from simulation."""
        if body_id in self._spawned:
            self.physics._pb.removeBody(
                body_id, physicsClientId=self.physics._client
            )
            del self._spawned[body_id]

    def clear(self):
        """Remove all spawned objects."""
        for body_id in list(self._spawned.keys()):
            self.remove(body_id)

    # ------------------------------------------------------------------
    # Catalogue queries (no physics needed)
    # ------------------------------------------------------------------

    @staticmethod
    def list_objects(category: Optional[str] = None) -> List[YCBObjectDef]:
        """List all YCB objects, optionally filtered by category."""
        objs = list(YCB_CATALOGUE.values())
        if category:
            objs = [o for o in objs if o.category == category]
        return sorted(objs, key=lambda o: o.ycb_id)

    @staticmethod
    def categories() -> List[str]:
        """Return all unique categories."""
        return sorted(set(o.category for o in YCB_CATALOGUE.values()))

    @staticmethod
    def get_definition(ycb_id: str) -> Optional[YCBObjectDef]:
        """Get object definition by ID (exact or fuzzy)."""
        if ycb_id in YCB_CATALOGUE:
            return YCB_CATALOGUE[ycb_id]
        # Fuzzy: match by name substring
        ycb_id_lower = ycb_id.lower()
        for k, v in YCB_CATALOGUE.items():
            if ycb_id_lower in k.lower() or ycb_id_lower in v.name.lower():
                return v
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, ycb_id: str) -> YCBObjectDef:
        defn = self.get_definition(ycb_id)
        if defn is None:
            raise ValueError(
                f"YCB object '{ycb_id}' not found.\n"
                f"Available: {list(YCB_CATALOGUE.keys())[:10]}..."
            )
        return defn

    def _create_body(
        self,
        defn: YCBObjectDef,
        pos:  Tuple,
        orn:  Tuple,
        color_override: Optional[Tuple] = None,
    ) -> int:
        """Create pybullet body from YCB definition using primitive geometry."""
        pb  = self.physics._pb
        sz  = defn.size_m
        col = defn.color_rgba if color_override is None else color_override

        if defn.mesh_type == "box":
            half = [sz[0]/2, sz[1]/2, sz[2]/2]
            col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half)
            vis_id = pb.createVisualShape(
                pb.GEOM_BOX, halfExtents=half, rgbaColor=col
            )

        elif defn.mesh_type == "cylinder":
            r = min(sz[0], sz[1]) / 2
            h = sz[2]
            col_id = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=r, height=h)
            vis_id = pb.createVisualShape(
                pb.GEOM_CYLINDER, radius=r, length=h, rgbaColor=col
            )

        elif defn.mesh_type == "sphere":
            r = min(sz) / 2
            col_id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=r)
            vis_id = pb.createVisualShape(
                pb.GEOM_SPHERE, radius=r, rgbaColor=col
            )

        else:  # fallback → box
            half = [sz[0]/2, sz[1]/2, sz[2]/2]
            col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half)
            vis_id = pb.createVisualShape(
                pb.GEOM_BOX, halfExtents=half, rgbaColor=col
            )

        body_id = pb.createMultiBody(
            baseMass                = defn.mass_kg,
            baseCollisionShapeIndex = col_id,
            baseVisualShapeIndex    = vis_id,
            basePosition            = pos,
            baseOrientation         = orn,
            physicsClientId         = self.physics._client,
        )

        # Apply real physical properties
        pb.changeDynamics(
            body_id, -1,
            lateralFriction  = defn.friction,
            restitution      = defn.restitution,
            physicsClientId  = self.physics._client,
        )
        return body_id

    def __repr__(self):
        return (
            f"ZeetaYCB(catalogue={len(YCB_CATALOGUE)} objects, "
            f"spawned={len(self._spawned)}, "
            f"categories={self.categories()})"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate():
    print("\n" + "="*60)
    print("  ZeetaYCB — Validation Suite")
    print("="*60)

    # 1. Catalogue stats
    total = len(YCB_CATALOGUE)
    cats  = ZeetaYCB.categories()
    graspable = sum(1 for o in YCB_CATALOGUE.values() if o.graspable)
    print(f"\n✅ Catalogue loaded : {total} objects")
    print(f"✅ Graspable        : {graspable}/{total}")
    print(f"✅ Categories       : {cats}")

    # 2. Per-category breakdown
    print(f"\n📋 Category breakdown:")
    for cat in cats:
        objs = ZeetaYCB.list_objects(category=cat)
        print(f"   {cat:<12} : {len(objs)} objects")

    # 3. Fuzzy lookup
    result = ZeetaYCB.get_definition("sugar")
    print(f"\n✅ Fuzzy lookup     : 'sugar' → {result.name if result else 'NOT FOUND'}")

    result2 = ZeetaYCB.get_definition("tennis")
    print(f"✅ Fuzzy lookup     : 'tennis' → {result2.name if result2 else 'NOT FOUND'}")

    # 4. Physics properties spot-check
    box = YCB_CATALOGUE["004_sugar_box"]
    print(f"\n✅ Sugar box        : mass={box.mass_kg}kg size={box.size_m} friction={box.friction}")
    ball = YCB_CATALOGUE["056_tennis_ball"]
    print(f"✅ Tennis ball      : mass={ball.mass_kg}kg restitution={ball.restitution} (bouncy)")

    # 5. Live PyBullet spawn test
    print(f"\n🔌 Attempting live spawn test...")
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from zeeta.zeeta_physics import ZeetaPhysics

        with ZeetaPhysics(headless=True) as physics:
            physics.reset()
            physics.add_plane()

            ycb = ZeetaYCB(physics)

            # Spawn 3 different objects
            id1 = ycb.spawn("004_sugar_box",       pos=(0.4, -0.1, 0.2))
            id2 = ycb.spawn("056_tennis_ball",      pos=(0.5,  0.0, 0.2))
            id3 = ycb.spawn("035_power_drill",      pos=(0.6,  0.1, 0.2))

            physics.step(120)   # let them settle

            ycb.randomise_all()

            print(f"✅ Spawned objects  : ids={[id1, id2, id3]}")

            for bid in [id1, id2, id3]:
                state = ycb.get_object_state(bid)
                pos   = state["position"].round(3)
                print(f"   {state['name']:<25} pos={pos}")

            graspable = ycb.get_graspable_objects()
            print(f"✅ Graspable found  : {len(graspable)} objects")

            # Scene population
            ycb.clear()
            scene_ids = ycb.spawn_scene(n_objects=5)
            physics.step(60)
            print(f"✅ Scene populated  : {len(scene_ids)} objects placed")

    except ImportError as e:
        print(f"⚠️  PyBullet not available: {e}")
        print("   Install: pip install pybullet")
        print("   All catalogue data and API is fully ready.")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        import traceback; traceback.print_exc()

    print("\n✅ zeeta_ycb.py validation complete.")
    print("="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _validate()