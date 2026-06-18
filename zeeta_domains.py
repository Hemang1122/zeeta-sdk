"""
zeeta_domains.py - Domain Preset Library
Each domain bundles: robot + objects + camera + task + scene
An engineer picks a domain instead of configuring 50 parameters.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class CameraPreset:
    distance: float
    yaw: float
    pitch: float
    target: Tuple[float, float, float]

@dataclass
class DomainPreset:
    key: str
    name: str
    description: str
    robot: str                      # franka | ur5 | kuka
    robot_base_z: float
    object_categories: List[str]    # which YCB categories fit this domain
    n_objects: Tuple[int, int]      # min, max objects per scene
    cameras: List[CameraPreset]
    task: str                       # pick_and_place | sorting | stacking | reaching
    table_center: Tuple[float, float]
    table_size: Tuple[float, float]
    spawn_height: float
    render_profile: str
    notes: str = ""

# ── THE DOMAIN LIBRARY ──────────────────────────────────────────────────────
DOMAINS: Dict[str, DomainPreset] = {

    "warehouse": DomainPreset(
        key="warehouse",
        name="Warehouse Picking",
        description="Robot picks boxes/containers off a surface and sorts them",
        robot="ur5",
        robot_base_z=0.63,
        object_categories=["food", "shape"],   # boxes, cans, rigid items
        n_objects=(2, 4),
        cameras=[
            CameraPreset(1.3, 0,   -55, (0.5, 0.0, 0.70)),   # top-down (conveyor cam)
            CameraPreset(1.4, 30,  -45, (0.5, 0.0, 0.70)),
        ],
        task="pick_and_place",
        table_center=(0.5, 0.0),
        table_size=(0.6, 0.4),
        spawn_height=0.645,
        render_profile="standard",
        notes="Top-down camera mimics conveyor-belt vision systems",
    ),

    "kitchen": DomainPreset(
        key="kitchen",
        name="Kitchen Manipulation",
        description="Robot handles cups, bowls, utensils, food items",
        robot="franka",
        robot_base_z=0.63,
        object_categories=["kitchen", "food"],
        n_objects=(2, 5),
        cameras=[
            CameraPreset(1.1, 45,  -30, (0.5, 0.0, 0.72)),
            CameraPreset(1.2, 90,  -25, (0.5, 0.0, 0.72)),
            CameraPreset(1.0, 135, -35, (0.45,0.0, 0.72)),
        ],
        task="pick_and_place",
        table_center=(0.5, 0.0),
        table_size=(0.5, 0.5),
        spawn_height=0.645,
        render_profile="standard",
        notes="Multiple angles for cluttered countertop scenes",
    ),

    "retail": DomainPreset(
        key="retail",
        name="Retail Shelf Stocking",
        description="Robot picks products and places them on shelves",
        robot="ur5",
        robot_base_z=0.63,
        object_categories=["food", "kitchen"],
        n_objects=(3, 6),
        cameras=[
            CameraPreset(1.2, 0,   -40, (0.5, 0.0, 0.72)),
            CameraPreset(1.3, 60,  -35, (0.5, 0.1, 0.72)),
        ],
        task="pick_and_place",
        table_center=(0.5, 0.0),
        table_size=(0.7, 0.3),
        spawn_height=0.645,
        render_profile="aggressive",   # varied store lighting
        notes="Wide shallow table mimics shelf; aggressive lighting variation",
    ),

    "lab": DomainPreset(
        key="lab",
        name="Lab Automation",
        description="Robot handles tools, small precise objects",
        robot="franka",
        robot_base_z=0.63,
        object_categories=["tool", "shape"],
        n_objects=(2, 4),
        cameras=[
            CameraPreset(1.0, 45,  -40, (0.5, 0.0, 0.72)),
            CameraPreset(0.9, 90,  -45, (0.5, 0.0, 0.72)),
        ],
        task="pick_and_place",
        table_center=(0.5, 0.0),
        table_size=(0.4, 0.4),
        spawn_height=0.645,
        render_profile="standard",
        notes="Close cameras for precision tool manipulation",
    ),

    "tabletop": DomainPreset(
        key="tabletop",
        name="General Tabletop (default)",
        description="Mixed objects, general manipulation research",
        robot="kuka",
        robot_base_z=0.63,
        object_categories=["food", "kitchen", "tool", "shape"],
        n_objects=(2, 4),
        cameras=[
            CameraPreset(1.2, 45,  -35, (0.5, 0.0, 0.75)),
            CameraPreset(1.0, 90,  -25, (0.5, 0.0, 0.75)),
            CameraPreset(1.4, 20,  -40, (0.5, 0.1, 0.70)),
            CameraPreset(1.1, 135, -30, (0.45,0.0, 0.75)),
        ],
        task="pick_and_place",
        table_center=(0.5, 0.0),
        table_size=(0.5, 0.5),
        spawn_height=0.645,
        render_profile="standard",
        notes="The default 4-angle generalist setup",
    ),
}

def get_domain(key: str) -> DomainPreset:
    """Fetch a domain preset, fuzzy matched."""
    key = key.lower().strip()
    if key in DOMAINS:
        return DOMAINS[key]
    # fuzzy
    for k, d in DOMAINS.items():
        if key in k or key in d.name.lower():
            return d
    raise KeyError(f"Unknown domain '{key}'. Available: {list(DOMAINS.keys())}")

def list_domains():
    """Print all available domains."""
    print("\n" + "="*60)
    print("  ZEETA DOMAIN LIBRARY")
    print("="*60)
    for i, (k, d) in enumerate(DOMAINS.items(), 1):
        print(f"\n  [{i}] {d.name}  (key: {k})")
        print(f"      {d.description}")
        print(f"      Robot: {d.robot.upper()}  |  "
              f"Objects: {', '.join(d.object_categories)}  |  "
              f"Cameras: {len(d.cameras)}")
    print("\n" + "="*60)

if __name__ == "__main__":
    list_domains()
    print("\nExample fetch:")
    d = get_domain("warehouse")
    print(f"  get_domain('warehouse') -> {d.name}")
    print(f"    robot={d.robot}, task={d.task}, cameras={len(d.cameras)}")
