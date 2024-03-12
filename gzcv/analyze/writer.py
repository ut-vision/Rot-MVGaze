import os
from typing import Any, Dict, List

from PIL import Image


class PngWriter:
    def __init__(self, img_key, save_dir):
        self.img_key = img_key
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(self, samples: List[Dict[str, Any]]) -> None:
        """
        Args:
            samples: A data batch represented in list format.
        """
        for sample in samples:
            img_id = sample["id"]
            img = sample[self.img_key]
            save_path = os.path.join(self.save_dir, f"{img_id}.png")
            Image.fromarray(img).save(save_path)


class TbWriter:
    pass
