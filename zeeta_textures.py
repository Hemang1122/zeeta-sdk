import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import random
import logging
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ZeetaTextures")

# --- PBR Data Structures ---
@dataclass
class PBRMaterial:
    name: str
    base_color: Tuple[float, float, float]
    roughness: float
    metalness: float
    normal_strength: float = 1.0
    tile_scale: float = 1.0
    ycb_categories: List[str] = field(default_factory=list)

@dataclass
class PBRMapSet:
    albedo: np.ndarray
    normal: np.ndarray
    roughness: np.ndarray
    metalness: np.ndarray
    ao: np.ndarray

# --- Material Catalogue ---
MATERIAL_CATALOGUE = {
    "wood_grain": PBRMaterial("wood_grain", (0.4, 0.2, 0.05), 0.6, 0.0, 1.0, 1.0, ["food", "tool"]),
    "brushed_metal": PBRMaterial("brushed_metal", (0.7, 0.7, 0.8), 0.2, 0.9, 0.8, 1.0, ["tool", "kitchen"]),
    "rough_concrete": PBRMaterial("rough_concrete", (0.5, 0.5, 0.5), 0.8, 0.0, 1.2, 1.0, ["cleaning"]),
    "smooth_plastic": PBRMaterial("smooth_plastic", (0.1, 0.5, 0.9), 0.3, 0.0, 0.1, 1.0, ["food", "sport"]),
    "rubber": PBRMaterial("rubber", (0.1, 0.1, 0.1), 0.9, 0.0, 1.5, 1.0, ["sport"]),
    "fabric": PBRMaterial("fabric", (0.6, 0.2, 0.2), 0.7, 0.0, 0.9, 1.0, ["food"]),
    "ceramic": PBRMaterial("ceramic", (0.9, 0.9, 0.9), 0.1, 0.0, 0.05, 1.0, ["kitchen"]),
    "cardboard": PBRMaterial("cardboard", (0.7, 0.5, 0.3), 0.8, 0.0, 0.7, 1.0, ["food"]),
}

class ProceduralPBR:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _noise(self, H: int, W: int, scale: float = 1.0, octaves: int = 6) -> np.ndarray:
        return self.rng.random((H, W)).astype(np.float32)

    def _aniso_noise(self, H: int, W: int, scale: float = 1.0, stretch: float = 8.0) -> np.ndarray:
        stretched_W = min(int(W * stretch), 4096)
        n = self._noise(H, stretched_W, scale=scale, octaves=6)
        img = Image.fromarray((n * 255).astype(np.uint8))
        return np.array(img.resize((W, H), Image.BILINEAR)) / 255.0

    def _normal_from_height(self, height: np.ndarray, strength: float = 1.0) -> np.ndarray:
        normal = np.zeros((height.shape[0], height.shape[1], 3), dtype=np.uint8)
        normal[:, :, 2] = 255
        return normal

    def generate(
        self,
        material:   PBRMaterial,
        resolution: int = 64,
        seed:       Optional[int] = None,
    ) -> PBRMapSet:
        H, W = resolution, resolution
        albedo = np.full((H, W, 3), (np.array(material.base_color) * 255), dtype=np.uint8)
        normal = np.full((H, W, 3), 128, dtype=np.uint8)
        roughness = np.full((H, W, 1), int(material.roughness * 255), dtype=np.uint8)
        metalness = np.full((H, W, 1), int(material.metalness * 255), dtype=np.uint8)
        ao = np.full((H, W, 1), 255, dtype=np.uint8)

        if "wood" in material.name:
            grain_noise = self._aniso_noise(H, W, scale=0.05 * material.tile_scale, stretch=5.0)
            albedo = (albedo * (grain_noise * 0.3 + 0.7)[:,:,np.newaxis]).astype(np.uint8)
        elif "metal" in material.name:
            surface_noise = self._noise(H, W, scale=0.02 * material.tile_scale, octaves=8)
            albedo = (albedo * (surface_noise * 0.1 + 0.9)[:,:,np.newaxis]).astype(np.uint8)
        elif "concrete" in material.name:
            lump_noise = self._noise(H, W, scale=0.2 * material.tile_scale, octaves=3)
            albedo = (albedo * (lump_noise * 0.3 + 0.7)[:,:,np.newaxis]).astype(np.uint8)

        return PBRMapSet(albedo, normal, roughness, metalness, ao)

class ZeetaTextures:
    def __init__(self, physics):
        self.physics = physics
        self.cache_dir = Path("./texture_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._procedural = ProceduralPBR()
        self._loaded_textures = {}

    def _load_or_generate_map(self, material: PBRMaterial, map_type: str, resolution: int) -> np.ndarray:
        cache_key = f"{material.name}_{map_type}_{resolution}"
        if cache_key in self._loaded_textures:
            return self._loaded_textures[cache_key]
        maps = self._procedural.generate(material, resolution)
        if map_type == "Color": return maps.albedo
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)

    def apply(self, body_id: int, material_name: str, resolution: int = 64, link_id: int = -1):
        pb = self.physics.pb
        client = self.physics.client
        material = MATERIAL_CATALOGUE.get(material_name, PBRMaterial("default", (0.5, 0.5, 0.5), 0.5, 0.0))
        albedo_map = self._load_or_generate_map(material, "Color", resolution)
        albedo_path = self.cache_dir / f"{material_name}_albedo_temp.png"
        Image.fromarray(albedo_map).save(albedo_path)
        albedo_tex_id = pb.loadTexture(str(albedo_path), physicsClientId=client)
        pb.changeVisualShape(body_id, link_id, rgbaColor=[*material.base_color, 1.0], textureUniqueId=albedo_tex_id, physicsClientId=client)

    def randomise(self, body_id: int, link_id: int = -1):
        pb = self.physics.pb
        client = self.physics.client
        base_color = np.array(pb.getVisualShapeData(body_id, link_id, physicsClientId=client)[0][7][:3])
        hsv = cv2.cvtColor(np.uint8([[base_color * 255]]), cv2.COLOR_RGB2HSV)[0][0]
        hsv[0] = (hsv[0] + random.uniform(-15, 15)) % 180
        new_rgb = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0] / 255.0
        pb.changeVisualShape(body_id, link_id, rgbaColor=[*new_rgb, 1.0], physicsClientId=client)

    @staticmethod
    def list_materials(category: Optional[str] = None) -> Dict[str, PBRMaterial]:
        return MATERIAL_CATALOGUE
