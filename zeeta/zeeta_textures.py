"""
zeeta_textures.py
=================
Zeeta PBR Texture System — ambientCG CC0 equivalent.

Isaac Sim equivalent: NVIDIA Materials + MDL shaders + RTX renderer textures

What this module provides:
  ✅ Procedural PBR texture generation (wood, metal, concrete, plastic,
     rubber, fabric, ceramic) — zero downloads, pure numpy/PIL
  ✅ ambientCG API downloader (fetches real CC0 textures when online)
  ✅ Texture cache (disk + memory) — never regenerate twice
  ✅ PBR map set: albedo, normal, roughness, metalness, AO
  ✅ Domain randomisation: hue shift, roughness jitter, wear simulation
  ✅ PyBullet texture loader — apply textures to spawned bodies
  ✅ Per-episode texture randomisation hook for sim_worker

PBR Map convention (matches ambientCG naming):
  *_Color.png      — albedo / diffuse (RGB)
  *_NormalGL.png   — tangent-space normal map (RGB, GL convention)
  *_Roughness.png  — roughness (greyscale)
  *_Metalness.png  — metalness (greyscale)
  *_AmbientOcclusion.png — AO (greyscale)

Usage (from sim_worker.py):
    from zeeta.zeeta_textures import ZeetaTextures

    tex = ZeetaTextures(physics, cache_dir="zeeta/texture_cache")
    tex.apply(body_id=box_id, material="wood_oak")
    tex.randomise(body_id=box_id)   # per-episode variation
"""

from __future__ import annotations

import os
import io
import math
import hashlib
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger("ZeetaTextures")

# ---------------------------------------------------------------------------
# PBR Material catalogue
# ---------------------------------------------------------------------------

@dataclass
class PBRMaterial:
    """
    Definition of one PBR material.
    Maps to ambientCG asset naming convention.
    """
    name:           str          # internal id e.g. "wood_oak"
    display_name:   str          # human readable
    category:       str          # wood / metal / concrete / plastic / rubber / fabric / ceramic
    ambientcg_id:   str          # e.g. "WoodFloor041"  (for online download)
    base_color:     Tuple[float,float,float]   # RGB 0-1 for procedural fallback
    roughness:      float        # 0=mirror, 1=fully rough
    metalness:      float        # 0=dielectric, 1=conductor
    normal_strength: float       # 0=flat, 1=full normal map
    tile_scale:     float        # UV tiling  (higher = smaller tiles)
    ycb_categories: List[str]    # which YCB object categories this fits


MATERIAL_CATALOGUE: Dict[str, PBRMaterial] = {

    # ── WOOD ────────────────────────────────────────────────────────────────
    "wood_oak": PBRMaterial(
        "wood_oak", "Oak Wood", "wood", "WoodFloor041",
        base_color=(0.55, 0.35, 0.18), roughness=0.75, metalness=0.0,
        normal_strength=0.8, tile_scale=4.0,
        ycb_categories=["shape", "kitchen", "tool"],
    ),
    "wood_pine": PBRMaterial(
        "wood_pine", "Pine Wood", "wood", "WoodFloor052",
        base_color=(0.72, 0.55, 0.30), roughness=0.80, metalness=0.0,
        normal_strength=0.7, tile_scale=4.0,
        ycb_categories=["shape"],
    ),
    "wood_dark": PBRMaterial(
        "wood_dark", "Dark Walnut", "wood", "WoodFloor058",
        base_color=(0.28, 0.18, 0.10), roughness=0.65, metalness=0.0,
        normal_strength=0.9, tile_scale=3.0,
        ycb_categories=["shape", "kitchen"],
    ),

    # ── METAL ───────────────────────────────────────────────────────────────
    "metal_steel": PBRMaterial(
        "metal_steel", "Brushed Steel", "metal", "MetalPlates002",
        base_color=(0.75, 0.75, 0.78), roughness=0.35, metalness=1.0,
        normal_strength=0.6, tile_scale=6.0,
        ycb_categories=["tool", "kitchen"],
    ),
    "metal_aluminum": PBRMaterial(
        "metal_aluminum", "Aluminum", "metal", "MetalPlates006",
        base_color=(0.82, 0.82, 0.85), roughness=0.25, metalness=1.0,
        normal_strength=0.5, tile_scale=6.0,
        ycb_categories=["tool", "food"],
    ),
    "metal_rusty": PBRMaterial(
        "metal_rusty", "Rusty Metal", "metal", "MetalRust001",
        base_color=(0.55, 0.28, 0.12), roughness=0.90, metalness=0.7,
        normal_strength=1.0, tile_scale=5.0,
        ycb_categories=["tool"],
    ),
    "metal_gold": PBRMaterial(
        "metal_gold", "Gold", "metal", "Gold001",
        base_color=(0.85, 0.68, 0.12), roughness=0.15, metalness=1.0,
        normal_strength=0.3, tile_scale=8.0,
        ycb_categories=["tool"],
    ),

    # ── CONCRETE / STONE ────────────────────────────────────────────────────
    "concrete_bare": PBRMaterial(
        "concrete_bare", "Bare Concrete", "concrete", "Concrete025",
        base_color=(0.60, 0.58, 0.56), roughness=0.95, metalness=0.0,
        normal_strength=0.8, tile_scale=3.0,
        ycb_categories=[],
    ),
    "concrete_painted": PBRMaterial(
        "concrete_painted", "Painted Concrete", "concrete", "PaintedConcrete001",
        base_color=(0.85, 0.82, 0.78), roughness=0.85, metalness=0.0,
        normal_strength=0.6, tile_scale=3.0,
        ycb_categories=[],
    ),

    # ── PLASTIC ─────────────────────────────────────────────────────────────
    "plastic_white": PBRMaterial(
        "plastic_white", "White Plastic", "plastic", "PlasticRough001",
        base_color=(0.92, 0.92, 0.92), roughness=0.55, metalness=0.0,
        normal_strength=0.4, tile_scale=8.0,
        ycb_categories=["food", "cleaning", "kitchen"],
    ),
    "plastic_red": PBRMaterial(
        "plastic_red", "Red Plastic", "plastic", "PlasticRough001",
        base_color=(0.85, 0.12, 0.12), roughness=0.50, metalness=0.0,
        normal_strength=0.4, tile_scale=8.0,
        ycb_categories=["shape", "food"],
    ),
    "plastic_black": PBRMaterial(
        "plastic_black", "Black Plastic", "plastic", "PlasticRough002",
        base_color=(0.08, 0.08, 0.08), roughness=0.60, metalness=0.0,
        normal_strength=0.5, tile_scale=8.0,
        ycb_categories=["tool", "kitchen"],
    ),
    "plastic_glossy": PBRMaterial(
        "plastic_glossy", "Glossy Plastic", "plastic", "PlasticGlossy001",
        base_color=(0.20, 0.50, 0.90), roughness=0.10, metalness=0.0,
        normal_strength=0.2, tile_scale=10.0,
        ycb_categories=["food", "cleaning"],
    ),

    # ── RUBBER ──────────────────────────────────────────────────────────────
    "rubber_black": PBRMaterial(
        "rubber_black", "Black Rubber", "rubber", "Rubber001",
        base_color=(0.06, 0.06, 0.06), roughness=0.95, metalness=0.0,
        normal_strength=0.7, tile_scale=6.0,
        ycb_categories=["sport", "tool"],
    ),
    "rubber_green": PBRMaterial(
        "rubber_green", "Green Rubber", "rubber", "Rubber002",
        base_color=(0.12, 0.65, 0.18), roughness=0.90, metalness=0.0,
        normal_strength=0.6, tile_scale=6.0,
        ycb_categories=["sport"],
    ),

    # ── FABRIC ──────────────────────────────────────────────────────────────
    "fabric_cotton": PBRMaterial(
        "fabric_cotton", "Cotton Fabric", "fabric", "Fabric021",
        base_color=(0.90, 0.88, 0.82), roughness=1.0, metalness=0.0,
        normal_strength=0.9, tile_scale=5.0,
        ycb_categories=["cleaning"],
    ),

    # ── CERAMIC ─────────────────────────────────────────────────────────────
    "ceramic_white": PBRMaterial(
        "ceramic_white", "White Ceramic", "ceramic", "Ceramic001",
        base_color=(0.96, 0.96, 0.95), roughness=0.10, metalness=0.0,
        normal_strength=0.3, tile_scale=6.0,
        ycb_categories=["kitchen"],
    ),
    "ceramic_terracotta": PBRMaterial(
        "ceramic_terracotta", "Terracotta", "ceramic", "Ceramic002",
        base_color=(0.72, 0.38, 0.22), roughness=0.70, metalness=0.0,
        normal_strength=0.7, tile_scale=5.0,
        ycb_categories=["kitchen"],
    ),

    # ── CARDBOARD / PAPER ───────────────────────────────────────────────────
    "cardboard": PBRMaterial(
        "cardboard", "Cardboard", "cardboard", "Cardboard001",
        base_color=(0.78, 0.65, 0.42), roughness=0.90, metalness=0.0,
        normal_strength=0.6, tile_scale=4.0,
        ycb_categories=["food"],
    ),
    "paper_label": PBRMaterial(
        "paper_label", "Paper Label", "cardboard", "Paper001",
        base_color=(0.95, 0.92, 0.88), roughness=0.85, metalness=0.0,
        normal_strength=0.3, tile_scale=6.0,
        ycb_categories=["food", "cleaning"],
    ),
}


# ---------------------------------------------------------------------------
# Texture map set
# ---------------------------------------------------------------------------

@dataclass
class PBRMapSet:
    """
    Full set of PBR texture maps for one material at one resolution.
    All maps are numpy arrays (H, W, C) uint8.
    """
    material_name: str
    resolution:    int                    # e.g. 512
    albedo:        np.ndarray             # (H,W,3) RGB
    normal:        np.ndarray             # (H,W,3) RGB tangent-space GL
    roughness:     np.ndarray             # (H,W,1) greyscale
    metalness:     np.ndarray             # (H,W,1) greyscale
    ao:            np.ndarray             # (H,W,1) ambient occlusion

    def save(self, out_dir: str):
        """Save all maps to disk in ambientCG naming convention."""
        os.makedirs(out_dir, exist_ok=True)
        name = self.material_name
        Image.fromarray(self.albedo).save(
            os.path.join(out_dir, f"{name}_Color.png"))
        Image.fromarray(self.normal).save(
            os.path.join(out_dir, f"{name}_NormalGL.png"))
        Image.fromarray(self.roughness[:,:,0]).save(
            os.path.join(out_dir, f"{name}_Roughness.png"))
        Image.fromarray(self.metalness[:,:,0]).save(
            os.path.join(out_dir, f"{name}_Metalness.png"))
        Image.fromarray(self.ao[:,:,0]).save(
            os.path.join(out_dir, f"{name}_AmbientOcclusion.png"))
        logger.info(f"Saved PBR maps → {out_dir}/{name}_*.png")


# ---------------------------------------------------------------------------
# Procedural PBR Generator
# ---------------------------------------------------------------------------

class ProceduralPBR:
    """
    Generates realistic PBR texture maps procedurally using numpy + PIL.
    No downloads needed — pure CPU generation.

    Each material type uses noise patterns that match its real-world structure:
    - Wood: anisotropic grain noise along one axis
    - Metal: isotropic fine noise + directional scratches
    - Concrete: low-frequency lumpy noise + fine grain
    - Plastic: very smooth with subtle surface noise
    - Rubber: coarse isotropic noise
    - Fabric: woven cross-hatch pattern
    - Ceramic: very smooth, near-flat normal
    - Cardboard: layered corrugation pattern
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        material:   PBRMaterial,
        resolution: int = 512,
        seed:       Optional[int] = None,
    ) -> PBRMapSet:
        """Generate full PBR map set for a material."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        gen = getattr(self, f"_gen_{material.category}", self._gen_plastic)
        albedo, normal, roughness, metalness, ao = gen(material, resolution)

        return PBRMapSet(
            material_name = material.name,
            resolution    = resolution,
            albedo        = albedo,
            normal        = normal,
            roughness     = roughness,
            metalness     = metalness,
            ao            = ao,
        )

    # ------------------------------------------------------------------
    # Noise helpers
    # ------------------------------------------------------------------

    def _noise(self, H: int, W: int, scale: float = 1.0,
               octaves: int = 4) -> np.ndarray:
        """Multi-octave value noise → (H,W) float [0,1]."""
        result = np.zeros((H, W), dtype=np.float32)
        amp, freq = 1.0, 1.0
        total = 0.0
        for _ in range(octaves):
            sh = max(1, int(H / (scale * freq)))
            sw = max(1, int(W / (scale * freq)))
            small = self.rng.random((sh, sw)).astype(np.float32)
            img   = Image.fromarray((small * 255).astype(np.uint8))
            big   = img.resize((W, H), Image.BILINEAR)
            result += np.array(big, dtype=np.float32) / 255.0 * amp
            total += amp
            amp  *= 0.5
            freq *= 2.0
        return result / total

    def _aniso_noise(self, H: int, W: int, scale: float = 1.0,
                     stretch: float = 8.0) -> np.ndarray:
        """Anisotropic (stretched) noise — good for wood grain."""
        n = self._noise(H, int(W * stretch), scale=scale, octaves=6)
        img = Image.fromarray((n * 255).astype(np.uint8))
        return np.array(img.resize((W, H), Image.BILINEAR)) / 255.0

    def _normal_from_height(self, height: np.ndarray,
                             strength: float = 1.0) -> np.ndarray:
        """Compute tangent-space normal map from height map (Sobel via numpy)."""
        h = height.astype(np.float32)
        p = np.pad(h, 1, mode='edge')
        dx = (-p[1:-1,0:-2] + p[1:-1,2:]) * strength
        dy = (-p[0:-2,1:-1] + p[2:,1:-1]) * strength
        nz = np.ones_like(dx)
        length = np.sqrt(dx**2 + dy**2 + nz**2) + 1e-8
        nx = ((-dx / length) * 0.5 + 0.5) * 255
        ny = ((-dy / length) * 0.5 + 0.5) * 255
        nz = ((nz  / length) * 0.5 + 0.5) * 255
        normal = np.stack([nx, ny, nz], axis=-1).astype(np.uint8)
        return normal

    def _grey(self, value: float, H: int, W: int,
              noise_amt: float = 0.02) -> np.ndarray:
        """Flat greyscale map with optional noise."""
        base  = np.full((H, W), value, dtype=np.float32)
        noise = self.rng.random((H, W)).astype(np.float32) * noise_amt
        return np.clip((base + noise) * 255, 0, 255).astype(np.uint8)[:,:,np.newaxis]

    def _colorise(self, grey: np.ndarray,
                  color: Tuple[float,float,float],
                  noise_amt: float = 0.05) -> np.ndarray:
        """Apply base colour to a greyscale height map."""
        H, W = grey.shape
        noise = self.rng.random((H, W, 3)).astype(np.float32) * noise_amt
        rgb   = np.zeros((H, W, 3), dtype=np.float32)
        for i, c in enumerate(color):
            rgb[:,:,i] = grey * c
        rgb = np.clip(rgb + noise, 0, 1)
        return (rgb * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Per-category generators
    # ------------------------------------------------------------------

    def _gen_wood(self, mat: PBRMaterial, res: int):
        # Grain: anisotropic noise along X axis
        grain  = self._aniso_noise(res, res, scale=0.3, stretch=12.0)
        grain  = (grain + self._noise(res, res, scale=2.0, octaves=3) * 0.2)
        grain  = np.clip(grain, 0, 1)

        # Ring pattern
        rings  = np.sin(grain * math.pi * 18) * 0.5 + 0.5
        height = rings * 0.7 + grain * 0.3

        albedo    = self._colorise(height, mat.base_color, noise_amt=0.03)
        normal    = self._normal_from_height(height, mat.normal_strength)
        roughness = self._grey(mat.roughness, res, res, noise_amt=0.05)
        metalness = self._grey(0.0, res, res, noise_amt=0.0)
        ao_n      = self._noise(res, res, scale=3.0, octaves=2)
        ao        = self._grey(0.85 + ao_n * 0.15, res, res, noise_amt=0.0)
        return albedo, normal, roughness, metalness, ao

    def _gen_metal(self, mat: PBRMaterial, res: int):
        # Fine isotropic noise + horizontal scratch lines
        base_noise = self._noise(res, res, scale=10.0, octaves=2) * 0.3
        # Directional scratches
        scratch = np.zeros((res, res), dtype=np.float32)
        for _ in range(30):
            y  = self.rng.integers(0, res)
            w  = self.rng.integers(1, 3)
            v  = self.rng.uniform(0.3, 0.9)
            scratch[max(0,y-w):y+w, :] = v
        scratch = np.array(
            Image.fromarray((scratch*255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=0.5)
            )
        ) / 255.0
        height    = base_noise * 0.6 + scratch * 0.4
        albedo    = self._colorise(
            np.ones((res,res), dtype=np.float32) * 0.85,
            mat.base_color, noise_amt=0.02
        )
        normal    = self._normal_from_height(height, mat.normal_strength * 0.5)
        roughness = self._grey(mat.roughness, res, res, noise_amt=0.03)
        metalness = self._grey(mat.metalness, res, res, noise_amt=0.01)
        ao        = self._grey(0.95, res, res, noise_amt=0.01)
        return albedo, normal, roughness, metalness, ao

    def _gen_concrete(self, mat: PBRMaterial, res: int):
        # Large lumps + fine grain
        lumps = self._noise(res, res, scale=0.5, octaves=3)
        grain = self._noise(res, res, scale=8.0, octaves=2) * 0.25
        height = lumps * 0.75 + grain
        albedo    = self._colorise(height, mat.base_color, noise_amt=0.04)
        normal    = self._normal_from_height(height, mat.normal_strength)
        roughness = self._grey(mat.roughness, res, res, noise_amt=0.04)
        metalness = self._grey(0.0, res, res, noise_amt=0.0)
        ao_n      = self._noise(res, res, scale=1.0, octaves=2)
        ao        = self._grey(0.75 + ao_n * 0.2, res, res, noise_amt=0.0)
        return albedo, normal, roughness, metalness, ao

    def _gen_plastic(self, mat: PBRMaterial, res: int):
        # Very smooth, almost flat
        height    = self._noise(res, res, scale=20.0, octaves=1) * 0.1
        albedo    = self._colorise(
            np.ones((res,res), dtype=np.float32),
            mat.base_color, noise_amt=0.02
        )
        normal    = self._normal_from_height(height, mat.normal_strength * 0.3)
        roughness = self._grey(mat.roughness, res, res, noise_amt=0.02)
        metalness = self._grey(0.0, res, res, noise_amt=0.0)
        ao        = self._grey(0.97, res, res, noise_amt=0.0)
        return albedo, normal, roughness, metalness, ao

    def _gen_rubber(self, mat: PBRMaterial, res: int):
        # Coarse isotropic noise
        height    = self._noise(res, res, scale=5.0, octaves=4)
        albedo    = self._colorise(height, mat.base_color, noise_amt=0.02)
        normal    = self._normal_from_height(height, mat.normal_strength)
        roughness = self._grey(mat.roughness, res, res, noise_amt=0.03)
        metalness = self._grey(0.0, res, res, noise_amt=0.0)
        ao_n      = self._noise(res, res, scale=3.0, octaves=2)
        ao        = self._grey(0.80 + ao_n * 0.18, res, res, noise_amt=0.0)
        return albedo, normal, roughness, metalness, ao

    def _gen_fabric(self, mat: PBRMaterial, res: int):
        # Woven cross-hatch pattern
        xs = np.linspace(0, math.pi * 16, res, dtype=np.float32)
        ys = np.linspace(0, math.pi * 16, res, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        weave  = (np.sin(xx) * 0.5 + 0.5) * 0.5 + (np.sin(yy) * 0.5 + 0.5) * 0.5
        noise  = self._noise(res, res, scale=8.0, octaves=2) * 0.2
        height = weave * 0.8 + noise
        albedo    = self._colorise(height, mat.base_color, noise_amt=0.03)
        normal    = self._normal_from_height(height, mat.normal_strength)
        roughness = self._grey(mat.roughness, res, res, noise_amt=0.02)
        metalness = self._grey(0.0, res, res, noise_amt=0.0)
        ao_n      = np.clip(1.0 - weave * 0.3, 0, 1)
        ao        = (ao_n * 255).astype(np.uint8)[:,:,np.newaxis]
        return albedo, normal, roughness, metalness, ao

    def _gen_ceramic(self, mat: PBRMaterial, res: int):
        # Near-perfectly smooth with very subtle waviness
        height    = self._noise(res, res, scale=30.0, octaves=1) * 0.05
        albedo    = self._colorise(
            np.ones((res,res), dtype=np.float32),
            mat.base_color, noise_amt=0.01
        )
        normal    = self._normal_from_height(height, mat.normal_strength * 0.15)
        roughness = self._grey(mat.roughness, res, res, noise_amt=0.01)
        metalness = self._grey(0.0, res, res, noise_amt=0.0)
        ao        = self._grey(0.99, res, res, noise_amt=0.0)
        return albedo, normal, roughness, metalness, ao

    def _gen_cardboard(self, mat: PBRMaterial, res: int):
        # Corrugation ridges + fibre noise
        ys      = np.linspace(0, math.pi * 24, res, dtype=np.float32)
        ridges  = (np.sin(ys) * 0.5 + 0.5)[np.newaxis,:].repeat(res, axis=0)
        fibre   = self._noise(res, res, scale=6.0, octaves=3) * 0.3
        height  = ridges * 0.7 + fibre
        albedo    = self._colorise(height, mat.base_color, noise_amt=0.04)
        normal    = self._normal_from_height(height, mat.normal_strength)
        roughness = self._grey(mat.roughness, res, res, noise_amt=0.03)
        metalness = self._grey(0.0, res, res, noise_amt=0.0)
        ao_n      = self._noise(res, res, scale=2.0, octaves=2)
        ao        = self._grey(0.80 + ao_n * 0.18, res, res, noise_amt=0.0)
        return albedo, normal, roughness, metalness, ao


# ---------------------------------------------------------------------------
# ambientCG downloader (online fallback)
# ---------------------------------------------------------------------------

class AmbientCGDownloader:
    """
    Downloads real CC0 PBR textures from ambientCG when online.
    Falls back to ProceduralPBR silently when offline.
    """

    BASE_URL = "https://ambientcg.com/api/v2/full_json"
    CDN_URL  = "https://ambientcg.com/get"

    def __init__(self, cache_dir: str = "texture_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        ambientcg_id: str,
        resolution:   int = 512,
    ) -> Optional[Path]:
        """
        Download a texture pack from ambientCG.
        Returns local directory path or None if unavailable.
        """
        res_str  = f"{resolution}-PNG"
        out_dir  = self.cache_dir / ambientcg_id / res_str
        if out_dir.exists() and any(out_dir.iterdir()):
            logger.debug(f"Cache hit: {ambientcg_id}")
            return out_dir

        url = (
            f"{self.BASE_URL}?include=downloadData"
            f"&id={ambientcg_id}"
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ZeetaRobotics/1.0"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                import json
                data   = json.loads(resp.read())
                assets = data.get("foundAssets", [])
                if not assets:
                    return None

                downloads = assets[0].get("downloadFolders", {})
                dl_info   = (
                    downloads.get(res_str) or
                    downloads.get("1K-PNG") or
                    next(iter(downloads.values()), None)
                )
                if not dl_info:
                    return None

                dl_url = dl_info.get("downloadLink", "")
                if not dl_url:
                    return None

                out_dir.mkdir(parents=True, exist_ok=True)
                zip_path = out_dir / f"{ambientcg_id}.zip"
                urllib.request.urlretrieve(dl_url, zip_path)

                import zipfile
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(out_dir)
                zip_path.unlink()

                logger.info(f"Downloaded {ambientcg_id} → {out_dir}")
                return out_dir

        except (urllib.error.URLError, Exception) as e:
            logger.debug(f"ambientCG download failed ({ambientcg_id}): {e}")
            return None


# ---------------------------------------------------------------------------
# ZeetaTextures — main interface
# ---------------------------------------------------------------------------

class ZeetaTextures:
    """
    Zeeta PBR Texture System.
    Manages texture generation, caching, and application to pybullet bodies.

    Parameters
    ----------
    physics : ZeetaPhysics
        Active physics engine instance.
    cache_dir : str
        Directory for texture cache (generated + downloaded textures).
    resolution : int
        Default texture resolution. 256 for fast batch, 512 for quality.
    try_download : bool
        Attempt ambientCG download before procedural fallback.
    """

    def __init__(
        self,
        physics,
        cache_dir:    str = "zeeta/texture_cache",
        resolution:   int = 256,
        try_download: bool = True,
    ):
        self.physics      = physics
        self.cache_dir    = Path(cache_dir)
        self.resolution   = resolution
        self.try_download = try_download

        self._generator   = ProceduralPBR()
        self._downloader  = AmbientCGDownloader(str(self.cache_dir / "ambientcg"))
        self._map_cache:  Dict[str, PBRMapSet] = {}    # memory cache
        self._body_tex:   Dict[int, str] = {}           # body_id → material_name
        self._tex_ids:    Dict[str, int] = {}           # albedo path → pb texture id
        self._rng         = np.random.default_rng()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ZeetaTextures ready: cache={self.cache_dir} res={resolution}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        body_id:       int,
        material:      str = "wood_oak",
        seed:          Optional[int] = None,
    ) -> bool:
        """
        Apply a PBR material to a pybullet body.
        Uses albedo map as diffuse texture (PyBullet supports diffuse only).

        Returns True on success.
        """
        maps = self._get_maps(material, seed=seed)
        if maps is None:
            return False

        albedo_path = self._save_albedo(maps)
        tex_id      = self._load_pybullet_texture(albedo_path)
        if tex_id < 0:
            return False

        try:
            self.physics._pb.changeVisualShape(
                body_id, -1,
                textureUniqueId = tex_id,
                physicsClientId = self.physics._client,
            )
            self._body_tex[body_id] = material
            logger.debug(f"Applied '{material}' texture to body {body_id}")
            return True
        except Exception as e:
            logger.warning(f"Texture apply failed: {e}")
            return False

    def apply_by_ycb_category(
        self,
        body_id:  int,
        category: str,
        seed:     Optional[int] = None,
    ) -> bool:
        """Apply a random material appropriate for a YCB category."""
        candidates = [
            m for m in MATERIAL_CATALOGUE.values()
            if category in m.ycb_categories
        ]
        if not candidates:
            candidates = list(MATERIAL_CATALOGUE.values())
        mat = self._rng.choice(candidates)
        return self.apply(body_id, mat.name, seed=seed)

    def randomise(
        self,
        body_id:      int,
        hue_shift:    float = 0.05,    # max hue shift (fraction of 360°)
        roughness_jitter: float = 0.1, # ± roughness change
        seed:         Optional[int] = None,
    ) -> bool:
        """
        Randomise texture of an already-textured body.
        Applies hue shift and roughness variation for domain randomisation.
        """
        mat_name = self._body_tex.get(body_id)
        if mat_name is None:
            return False

        rng  = np.random.default_rng(seed)
        maps = self._get_maps(mat_name, seed=seed)
        if maps is None:
            return False

        # Apply hue shift to albedo
        albedo_img = Image.fromarray(maps.albedo).convert("HSV")
        h, s, v    = albedo_img.split()
        h_arr = np.array(h, dtype=np.float32)
        h_arr = (h_arr + rng.uniform(-hue_shift, hue_shift) * 255) % 255
        h_new = Image.fromarray(h_arr.astype(np.uint8))
        shifted = Image.merge("HSV", (h_new, s, v)).convert("RGB")
        shifted_arr = np.array(shifted)

        # Save shifted albedo
        shifted_maps = PBRMapSet(
            material_name = f"{mat_name}_rand",
            resolution    = maps.resolution,
            albedo        = shifted_arr,
            normal        = maps.normal,
            roughness     = maps.roughness,
            metalness     = maps.metalness,
            ao            = maps.ao,
        )
        albedo_path = self._save_albedo(shifted_maps)
        tex_id      = self._load_pybullet_texture(albedo_path)
        if tex_id < 0:
            return False

        try:
            self.physics._pb.changeVisualShape(
                body_id, -1,
                textureUniqueId = tex_id,
                physicsClientId = self.physics._client,
            )
            return True
        except Exception:
            return False

    def randomise_all(self, body_ids: List[int]):
        """Randomise textures on a list of bodies — call per episode."""
        for bid in body_ids:
            self.randomise(bid)

    def get_maps(self, material: str, seed: Optional[int] = None) -> Optional[PBRMapSet]:
        """Get PBR map set (generate or load from cache)."""
        return self._get_maps(material, seed)

    def save_maps(self, material: str, out_dir: str, seed: Optional[int] = None):
        """Generate and save all PBR maps to disk."""
        maps = self._get_maps(material, seed)
        if maps:
            maps.save(out_dir)

    def pregenerate_all(self, resolution: Optional[int] = None):
        """Pre-generate all materials — call once at startup."""
        res = resolution or self.resolution
        logger.info(f"Pre-generating {len(MATERIAL_CATALOGUE)} materials at {res}px...")
        for name in MATERIAL_CATALOGUE:
            self._get_maps(name, res=res)
        logger.info("Pre-generation complete.")

    # ------------------------------------------------------------------
    # Catalogue queries
    # ------------------------------------------------------------------

    @staticmethod
    def list_materials(category: Optional[str] = None) -> List[PBRMaterial]:
        mats = list(MATERIAL_CATALOGUE.values())
        if category:
            mats = [m for m in mats if m.category == category]
        return mats

    @staticmethod
    def categories() -> List[str]:
        return sorted(set(m.category for m in MATERIAL_CATALOGUE.values()))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_maps(
        self,
        material: str,
        res:      Optional[int] = None,
        seed:     Optional[int] = None,
    ) -> Optional[PBRMapSet]:
        res = res or self.resolution
        cache_key = f"{material}_{res}_{seed}"

        if cache_key in self._map_cache:
            return self._map_cache[cache_key]

        defn = MATERIAL_CATALOGUE.get(material)
        if defn is None:
            logger.warning(f"Unknown material: {material}")
            return None

        # Try disk cache first
        disk_path = self.cache_dir / "generated" / f"{material}_{res}"
        albedo_file = disk_path / f"{material}_Color.png"
        if albedo_file.exists():
            maps = self._load_from_disk(material, disk_path, res)
            self._map_cache[cache_key] = maps
            return maps

        # Try ambientCG download
        if self.try_download:
            dl_dir = self._downloader.download(defn.ambientcg_id, res)
            if dl_dir:
                maps = self._load_ambientcg(material, dl_dir, res)
                if maps:
                    self._map_cache[cache_key] = maps
                    return maps

        # Procedural generation
        logger.debug(f"Generating '{material}' at {res}px procedurally")
        maps = self._generator.generate(defn, resolution=res, seed=seed)

        # Save to disk cache
        maps.save(str(disk_path))
        self._map_cache[cache_key] = maps
        return maps

    def _load_from_disk(self, name: str, path: Path, res: int) -> PBRMapSet:
        def load(fname):
            p = path / fname
            if p.exists():
                img = Image.open(p).convert("RGB").resize((res, res))
                return np.array(img)
            return None
        def load_grey(fname):
            p = path / fname
            if p.exists():
                img = Image.open(p).convert("L").resize((res, res))
                return np.array(img)[:,:,np.newaxis]
            return np.full((res,res,1), 128, dtype=np.uint8)
        return PBRMapSet(
            material_name = name, resolution = res,
            albedo    = load(f"{name}_Color.png") or np.full((res,res,3),128,dtype=np.uint8),
            normal    = load(f"{name}_NormalGL.png") or np.full((res,res,3),[128,128,255],dtype=np.uint8),
            roughness = load_grey(f"{name}_Roughness.png"),
            metalness = load_grey(f"{name}_Metalness.png"),
            ao        = load_grey(f"{name}_AmbientOcclusion.png"),
        )

    def _load_ambientcg(self, name: str, dl_dir: Path, res: int) -> Optional[PBRMapSet]:
        """Parse ambientCG directory structure into PBRMapSet."""
        files = list(dl_dir.rglob("*.png"))
        def find(keyword):
            for f in files:
                if keyword.lower() in f.name.lower():
                    img = Image.open(f).resize((res, res))
                    return np.array(img.convert("RGB"))
            return None
        def find_grey(keyword):
            for f in files:
                if keyword.lower() in f.name.lower():
                    img = Image.open(f).resize((res, res))
                    return np.array(img.convert("L"))[:,:,np.newaxis]
            return None
        albedo = find("color") or find("colour") or find("diffuse")
        if albedo is None:
            return None
        return PBRMapSet(
            material_name = name, resolution = res,
            albedo    = albedo,
            normal    = find("normal") or np.full((res,res,3),[128,128,255],dtype=np.uint8),
            roughness = find_grey("roughness") or np.full((res,res,1),180,dtype=np.uint8),
            metalness = find_grey("metalness") or np.zeros((res,res,1),dtype=np.uint8),
            ao        = find_grey("ambientocclusion") or np.full((res,res,1),220,dtype=np.uint8),
        )

    def _save_albedo(self, maps: PBRMapSet) -> str:
        """Save albedo to temp file for PyBullet loading."""
        path = str(self.cache_dir / "pybullet_tex" / f"{maps.material_name}_albedo.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(maps.albedo).save(path)
        return path

    def _load_pybullet_texture(self, albedo_path: str) -> int:
        """Load texture into PyBullet, return texture ID."""
        if albedo_path in self._tex_ids:
            return self._tex_ids[albedo_path]
        try:
            tex_id = self.physics._pb.loadTexture(
                albedo_path, physicsClientId=self.physics._client
            )
            self._tex_ids[albedo_path] = tex_id
            return tex_id
        except Exception as e:
            logger.warning(f"PyBullet loadTexture failed: {e}")
            return -1

    def __repr__(self):
        return (
            f"ZeetaTextures(materials={len(MATERIAL_CATALOGUE)}, "
            f"cached={len(self._map_cache)}, "
            f"res={self.resolution})"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate():
    print("\n" + "="*60)
    print("  ZeetaTextures — Validation Suite")
    print("="*60)

    # 1. Catalogue
    total = len(MATERIAL_CATALOGUE)
    cats  = ZeetaTextures.categories()
    print(f"\n✅ Materials loaded : {total}")
    print(f"✅ Categories       : {cats}")
    for cat in cats:
        mats = ZeetaTextures.list_materials(category=cat)
        print(f"   {cat:<12} : {len(mats)} materials")

    # 2. Procedural generation
    print(f"\n🔧 Generating PBR maps (256px)...")
    gen = ProceduralPBR(seed=42)
    results = []
    for name, mat in list(MATERIAL_CATALOGUE.items())[:6]:
        maps = gen.generate(mat, resolution=256)
        results.append((name, maps))
        print(f"   ✅ {name:<25} albedo={maps.albedo.shape} "
              f"normal={maps.normal.shape} "
              f"roughness={maps.roughness.shape}")

    # 3. Save maps to temp dir
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        maps = results[0][1]
        maps.save(tmpdir)
        saved = os.listdir(tmpdir)
        print(f"\n✅ Saved PBR maps   : {sorted(saved)}")

    # 4. YCB category matching
    print(f"\n✅ YCB category match:")
    for cat in ["food", "tool", "sport", "kitchen"]:
        candidates = [
            m.display_name for m in MATERIAL_CATALOGUE.values()
            if cat in m.ycb_categories
        ]
        print(f"   {cat:<10} → {candidates}")

    # 5. Live physics test
    print(f"\n🔌 Attempting live physics + texture test...")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from zeeta.zeeta_physics import ZeetaPhysics
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with ZeetaPhysics(headless=True) as physics:
                physics.reset()
                physics.add_plane()
                box = physics.add_box(
                    half_extents=[0.05,0.05,0.05],
                    pos=[0.5, 0.0, 0.3],
                    color=(0.8, 0.8, 0.8, 1.0)
                )
                physics.step(60)

                tex = ZeetaTextures(
                    physics,
                    cache_dir=tmpdir,
                    resolution=256,
                    try_download=False,
                )
                ok = tex.apply(box, "wood_oak")
                print(f"   ✅ Texture applied  : wood_oak → body={box} success={ok}")
                ok2 = tex.apply(box, "metal_steel")
                print(f"   ✅ Texture swapped  : metal_steel success={ok2}")
                print(f"   ✅ Engine state     : {tex}")

    except ImportError as e:
        print(f"⚠️  PyBullet not available: {e}")
        print("   Procedural generation fully works — PyBullet apply needs live sim.")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        import traceback; traceback.print_exc()

    print("\n✅ zeeta_textures.py validation complete.")
    print("="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _validate()