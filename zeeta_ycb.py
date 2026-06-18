import pybullet as p
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class YCBObjectDef:
    ycb_id: str; name: str; category: str; mass_kg: float; size_m: Tuple[float, float, float]; mesh_type: str; color_rgba: Tuple[float, float, float, float]

YCB_CATALOGUE = {
    "002_master_chef_can": YCBObjectDef("002_master_chef_can", "Master Chef Can", "food", 0.414, (0.102, 0.102, 0.140), "cylinder", (0.9, 0.3, 0.1, 1.0)),
    "003_cracker_box": YCBObjectDef("003_cracker_box", "Cracker Box", "food", 0.411, (0.060, 0.158, 0.210), "box", (0.9, 0.8, 0.2, 1.0)),
    "004_sugar_box": YCBObjectDef("004_sugar_box", "Sugar Box", "food", 0.514, (0.038, 0.089, 0.175), "box", (0.95, 0.95, 0.85, 1.0)),
    "005_tomato_soup_can": YCBObjectDef("005_tomato_soup_can", "Tomato Soup Can", "food", 0.349, (0.068, 0.068, 0.102), "cylinder", (0.8, 0.1, 0.1, 1.0)),
    "006_mustard_bottle": YCBObjectDef("006_mustard_bottle", "Mustard Bottle", "food", 0.603, (0.058, 0.091, 0.190), "cylinder", (0.9, 0.8, 0.0, 1.0)),
    "013_apple": YCBObjectDef("013_apple", "Apple", "food", 0.068, (0.074, 0.074, 0.075), "sphere", (0.8, 0.1, 0.1, 1.0)),
    "035_power_drill": YCBObjectDef("035_power_drill", "Power Drill", "tool", 0.895, (0.184, 0.187, 0.094), "box", (0.9, 0.5, 0.0, 1.0)),
    "053_mini_soccer_ball": YCBObjectDef("053_mini_soccer_ball", "Mini Soccer Ball", "sport", 0.123, (0.104, 0.104, 0.104), "sphere", (0.95, 0.95, 0.95, 1.0)),
}

class ZeetaYCB:
    def __init__(self, physics):
        self.physics = physics
        self._spawned = [] # Stores (body_id, YCBObjectDef)

    def spawn_scene(self, n_objects: int, table_center: List[float], table_size: List[float], z_height: float) -> List[int]:
        pb = self.physics.pb
        client = self.physics.client
        ids = []
        
        for _ in range(n_objects):
            defn = random.choice(list(YCB_CATALOGUE.values()))
            
            # Random position on the table
            pos_x = table_center[0] + random.uniform(-table_size[0]/2, table_size[0]/2)
            pos_y = table_center[1] + random.uniform(-table_size[1]/2, table_size[1]/2)
            pos = [pos_x, pos_y, z_height]
            orn = pb.getQuaternionFromEuler([0, 0, random.uniform(0, 2 * np.pi)])
            
            if defn.mesh_type == "box":
                col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[s/2 for s in defn.size_m], physicsClientId=client)
                vis_id = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[s/2 for s in defn.size_m], rgbaColor=defn.color_rgba, physicsClientId=client)
            elif defn.mesh_type == "cylinder":
                col_id = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=defn.size_m[0]/2, height=defn.size_m[2], physicsClientId=client)
                vis_id = pb.createVisualShape(pb.GEOM_CYLINDER, radius=defn.size_m[0]/2, length=defn.size_m[2], rgbaColor=defn.color_rgba, physicsClientId=client)
            elif defn.mesh_type == "sphere":
                col_id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=defn.size_m[0]/2, physicsClientId=client)
                vis_id = pb.createVisualShape(pb.GEOM_SPHERE, radius=defn.size_m[0]/2, rgbaColor=defn.color_rgba, physicsClientId=client)
            else:
                # Fallback for unknown mesh types or URDFs not handled here
                col_id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.05, physicsClientId=client)
                vis_id = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.05, rgbaColor=[1,0,0,1], physicsClientId=client)
            
            body_id = pb.createMultiBody(baseMass=defn.mass_kg, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=pos, baseOrientation=orn, physicsClientId=client)
            ids.append(body_id)
            self._spawned.append((body_id, defn)) # Store body_id and its definition
        return ids

    def clear(self):
        pb = self.physics.pb
        client = self.physics.client
        for body_id, _ in self._spawned: 
            pb.removeBody(body_id, physicsClientId=client)
        self._spawned = []

    def randomise_object(self, obj_id: int):
        # Example: Randomize friction and restitution
        pb = self.physics.pb
        client = self.physics.client
        lateral_friction = random.uniform(0.5, 1.5)
        restitution = random.uniform(0.0, 0.5)
        pb.changeDynamics(obj_id, -1, lateralFriction=lateral_friction, restitution=restitution, physicsClientId=client)
