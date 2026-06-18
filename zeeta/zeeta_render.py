
"""
zeeta_render.py - Layer 6: Photorealistic Post-Processing Renderer
"""
from __future__ import annotations
import random, logging, time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)

@dataclass
class RenderProfile:
    name: str
    brightness_range:    Tuple[float, float] = (0.6, 1.4)
    contrast_range:      Tuple[float, float] = (0.7, 1.5)
    gamma_range:         Tuple[float, float] = (0.7, 1.4)
    shadow_strength:     Tuple[float, float] = (0.0, 0.3)
    highlight_strength:  Tuple[float, float] = (0.0, 0.2)
    saturation_range:    Tuple[float, float] = (0.6, 1.6)
    hue_shift_range:     Tuple[float, float] = (-12, 12)
    white_balance_range: Tuple[float, float] = (0.85, 1.15)
    gaussian_noise_std:  Tuple[float, float] = (0.0, 8.0)
    salt_pepper_prob:    Tuple[float, float] = (0.0, 0.003)
    jpeg_quality_range:  Tuple[int,   int]   = (75, 100)
    blur_prob:           float = 0.3
    blur_radius_range:   Tuple[float, float] = (0.3, 1.2)
    dof_prob:            float = 0.2
    motion_blur_prob:    float = 0.15
    motion_blur_kernel:  Tuple[int, int]     = (3, 9)
    depth_noise_std:     Tuple[float, float] = (0.0, 0.005)
    depth_dropout_prob:  Tuple[float, float] = (0.0, 0.02)

PROFILES = {
    "clean": RenderProfile(
        name="clean",
        brightness_range=(0.95, 1.05), contrast_range=(0.95, 1.05),
        gamma_range=(0.95, 1.05), saturation_range=(0.95, 1.05),
        hue_shift_range=(-3, 3), white_balance_range=(0.97, 1.03),
        gaussian_noise_std=(0.0, 2.0), salt_pepper_prob=(0.0, 0.0002),
        jpeg_quality_range=(95, 100), blur_prob=0.05, dof_prob=0.0,
        motion_blur_prob=0.0, depth_noise_std=(0.0, 0.001),
    ),
    "standard": RenderProfile(name="standard"),
    "aggressive": RenderProfile(
        name="aggressive",
        brightness_range=(0.4, 1.8), contrast_range=(0.5, 2.0),
        gamma_range=(0.5, 1.8), shadow_strength=(0.0, 0.5),
        highlight_strength=(0.0, 0.4), saturation_range=(0.3, 2.0),
        hue_shift_range=(-20, 20), white_balance_range=(0.75, 1.25),
        gaussian_noise_std=(0.0, 20.0), salt_pepper_prob=(0.0, 0.008),
        jpeg_quality_range=(55, 100), blur_prob=0.5,
        blur_radius_range=(0.3, 2.5), dof_prob=0.35,
        motion_blur_prob=0.25, motion_blur_kernel=(3, 15),
        depth_noise_std=(0.0, 0.015), depth_dropout_prob=(0.0, 0.05),
    ),
}

@dataclass
class RenderDiagnostics:
    frames_rendered:   int = 0
    depth_rendered:    int = 0
    episodes_rendered: int = 0

class ZeetaRenderer:
    def __init__(self, profile: str = "standard", seed: Optional[int] = None):
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile. Choose: {list(PROFILES)}")
        self.profile  = PROFILES[profile]
        self.rng      = random.Random(seed)
        self.np_rng   = np.random.RandomState(seed)
        self._diag    = RenderDiagnostics()
        logger.info("ZeetaRenderer ready | profile=%s", profile)

    def _sample(self, lo, hi): return self.rng.uniform(lo, hi)

    def _gamma(self, img, g):
        return np.power(np.clip(img/255.0, 0, 1), 1.0/g) * 255.0

    def _brightness(self, img, f): return img * f

    def _contrast(self, img, f):
        m = img.mean(); return (img - m) * f + m

    def _shadow_highlight(self, img, shadow, highlight):
        lum = img.mean(axis=2, keepdims=True) / 255.0
        img = img * (1.0 - np.clip(1.0 - lum*3, 0, 1) * shadow)
        img = img + (255.0 - img) * np.clip((lum-0.7)*3, 0, 1) * highlight
        return img

    def _white_balance(self, img, wb_range):
        for c in range(3): img[:,:,c] *= self._sample(*wb_range)
        return img

    def _saturation(self, img, f):
        gray = img.mean(axis=2, keepdims=True)
        return gray + (img - gray) * f

    def _hue_shift(self, img, deg):
        if abs(deg) < 0.5: return img
        u8  = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + deg/2.0) % 180.0
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    def _gaussian_noise(self, img, std):
        return img + self.np_rng.normal(0, std, img.shape).astype(np.float32)

    def _salt_pepper(self, img, prob):
        out = img.copy()
        out[self.np_rng.random(img.shape[:2]) < prob/2] = 255.0
        out[self.np_rng.random(img.shape[:2]) < prob/2] = 0.0
        return out

    def _gaussian_blur(self, img, r):
        k = max(1, int(r*2)|1)
        return cv2.GaussianBlur(np.clip(img,0,255).astype(np.uint8),(k,k),r).astype(np.float32)

    def _dof_blur(self, img, r):
        h,w = img.shape[:2]; k = max(1,int(r*2)|1)
        u8  = np.clip(img,0,255).astype(np.uint8)
        blr = cv2.GaussianBlur(u8,(k,k),r)
        Y,X = np.ogrid[:h,:w]
        mask = np.clip(np.sqrt((X-w//2)**2+(Y-h//2)**2)/(max(h,w)*0.4),0,1)[...,np.newaxis]
        return (u8*(1-mask)+blr*mask).astype(np.float32)

    def _motion_blur(self, img, k):
        u8  = np.clip(img,0,255).astype(np.uint8)
        ker = np.zeros((k,k),np.float32); ker[k//2,:] = 1.0/k
        return cv2.filter2D(u8,-1,ker).astype(np.float32)

    def _jpeg(self, img, q):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _,enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def render_rgb(self, rgb, episode_seed=None):
        if episode_seed is not None:
            self.rng.seed(episode_seed); self.np_rng.seed(episode_seed)
        p   = self.profile
        img = rgb.astype(np.float32)
        img = self._gamma(img, self._sample(*p.gamma_range))
        img = self._brightness(img, self._sample(*p.brightness_range))
        img = self._shadow_highlight(img, self._sample(*p.shadow_strength), self._sample(*p.highlight_strength))
        img = self._white_balance(img, p.white_balance_range)
        img = self._contrast(img, self._sample(*p.contrast_range))
        img = self._saturation(img, self._sample(*p.saturation_range))
        img = self._hue_shift(img, self._sample(*p.hue_shift_range))
        std = self._sample(*p.gaussian_noise_std)
        if std > 0: img = self._gaussian_noise(img, std)
        sp  = self._sample(*p.salt_pepper_prob)
        if sp > 0:  img = self._salt_pepper(img, sp)
        if self.rng.random() < p.motion_blur_prob:
            k = self.rng.randrange(p.motion_blur_kernel[0], p.motion_blur_kernel[1]+1, 2)
            img = self._motion_blur(img, k)
        elif self.rng.random() < p.dof_prob:
            img = self._dof_blur(img, self._sample(*p.blur_radius_range)*2)
        elif self.rng.random() < p.blur_prob:
            img = self._gaussian_blur(img, self._sample(*p.blur_radius_range))
        u8  = np.clip(img, 0, 255).astype(np.uint8)
        u8  = self._jpeg(u8, self.rng.randint(*p.jpeg_quality_range))
        self._diag.frames_rendered += 1
        return u8

    def render_depth(self, depth, episode_seed=None):
        if episode_seed is not None: self.np_rng.seed(episode_seed+1000)
        p   = self.profile
        out = depth.astype(np.float32).copy()
        std = self._sample(*p.depth_noise_std)
        if std > 0: out += self.np_rng.normal(0, std, out.shape).astype(np.float32)
        dp  = self._sample(*p.depth_dropout_prob)
        if dp > 0: out[self.np_rng.random(out.shape) < dp] = 0.0
        self._diag.depth_rendered += 1
        return np.clip(out, 0.0, None)

    def render_episode(self, rgb_frames, depth_frames, episode_id=0):
        out_rgb, out_depth = [], []
        for i,(rgb,depth) in enumerate(zip(rgb_frames, depth_frames)):
            seed = episode_id*1000 + i
            out_rgb.append(self.render_rgb(rgb, episode_seed=seed))
            out_depth.append(self.render_depth(depth, episode_seed=seed))
        self._diag.episodes_rendered += 1
        return {"rgb_frames": out_rgb, "depth_frames": out_depth}

    def diagnostics(self):
        return {"frames_rendered": self._diag.frames_rendered,
                "depth_rendered":  self._diag.depth_rendered,
                "episodes_rendered": self._diag.episodes_rendered,
                "profile": self.profile.name}

def _validate():
    print("="*60)
    print("ZeetaRenderer - Layer 6 Validation")
    print("="*60)
    H, W = 224, 224
    for profile_name in ["clean", "standard", "aggressive"]:
        renderer = ZeetaRenderer(profile=profile_name, seed=42)
        rgb   = np.random.randint(80, 200, (H,W,3), dtype=np.uint8)
        depth = np.random.uniform(0.3, 1.5, (H,W)).astype(np.float32)
        n  = 50; t0 = time.perf_counter()
        for i in range(n):
            out_rgb   = renderer.render_rgb(rgb, episode_seed=i)
            out_depth = renderer.render_depth(depth, episode_seed=i)
        ms = (time.perf_counter()-t0)/n*1000
        print(f"\n  Profile : {profile_name}")
        print(f"  ms/frame: {ms:.1f}ms")
        print(f"  Shape   : {out_rgb.shape} {out_rgb.dtype}")
        print(f"  Range   : [{out_rgb.min()}, {out_rgb.max()}]")
        assert out_rgb.shape==(H,W,3) and out_rgb.dtype==np.uint8
        assert out_depth.shape==(H,W)
        print(f"  Assertions: PASSED")
    print("\n  Episode test (9 frames):")
    r2 = ZeetaRenderer("standard", seed=0)
    rgbs   = [np.random.randint(50,220,(H,W,3),dtype=np.uint8) for _ in range(9)]
    depths = [np.random.uniform(0.3,1.5,(H,W)).astype(np.float32) for _ in range(9)]
    t0 = time.perf_counter()
    res = r2.render_episode(rgbs, depths, episode_id=7)
    print(f"  9 frames in {(time.perf_counter()-t0)*1000:.1f}ms")
    print(f"  RGB out: {len(res[\"rgb_frames\"])}  Depth out: {len(res[\"depth_frames\"])}")
    print("\n"+"="*60)
    print("Layer 6 - Renderer: PASSED")
    print("="*60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _validate()

