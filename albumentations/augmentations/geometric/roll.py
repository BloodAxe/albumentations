import random
from typing import Dict, Any

import numpy as np


from ...core.transforms_interface import DualTransform

__all__ = ["Roll"]


class Roll(DualTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image = params["image"]
        roll_x = random.randint(0, image.shape[0])
        roll_y = random.randint(0, image.shape[1])
        return {"roll_x": roll_x, "roll_y": roll_y}

    def apply(self, img: np.ndarray, roll_x: int = 0, roll_y: int = 0, **params) -> np.ndarray:
        img = np.roll(img, shift=roll_y, axis=0)
        img = np.roll(img, shift=roll_x, axis=1)
        return img

    def apply_to_mask(self, img: np.ndarray, roll_x: int = 0, roll_y: int = 0, **params) -> np.ndarray:
        img = np.roll(img, shift=roll_y, axis=0)
        img = np.roll(img, shift=roll_x, axis=1)
        return img

    def targets_as_params(self) -> List[str]:
        return ["image"]
