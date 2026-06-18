from __future__ import annotations
import random, logging
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import cv2

@dataclass
class RenderProfile:
    name: str
    brightness_range:    Tuple[float, float] = (0.6, 1.4)
    contrast_range:      Tuple[float, float] = (0.7, 1.5)
    noise_std:           Tuple[float, float] = (0.0, 8.0)

class ZeetaRenderer:
    def __init__(self):
        self._rng = random.Random()
        self.style_brightness = 1.0
        self.style_contrast = 1.0
        self.profiles = {
            "aggressive": RenderProfile("aggressive", (0.5, 1.5), (0.5, 1.5), (5.0, 15.0)),
            "standard": RenderProfile("standard", (0.8, 1.2), (0.8, 1.2), (2.0, 5.0))
        }

    def set_episode_style(self, episode_seed: int, profile_name: str = "standard"):
        """Sets the visual style for the WHOLE episode (No more disco!)"""
        self._rng.seed(episode_seed)
        profile = self.profiles.get(profile_name, self.profiles["standard"])
        self.style_brightness = self._rng.uniform(*profile.brightness_range)
        self.style_contrast = self._rng.uniform(*profile.contrast_range)

    def render_rgb(self, rgb_frame: np.ndarray, frame_seed: int, profile_name: str = "standard") -> np.ndarray:
        profile = self.profiles.get(profile_name, self.profiles["standard"])
        img = rgb_frame.copy()
        
        # Apply the CONSISTENT episode style
        img = cv2.convertScaleAbs(img, alpha=self.style_contrast, beta=128*(1-self.style_contrast) + (self.style_brightness-1)*255)
        
        # Apply the RANDOM frame noise
        self._rng.seed(frame_seed)
        if profile.noise_std[1] > 0:
            std = self._rng.uniform(*profile.noise_std)
            noise = np.zeros(img.shape, np.int16)
            cv2.randn(noise, 0, std)
            img = cv2.add(img, noise, dtype=cv2.CV_8U)
            
        return img
