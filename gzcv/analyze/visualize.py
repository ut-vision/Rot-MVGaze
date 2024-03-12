from typing import Any, Dict, List

import aggdraw
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib.colors import to_hex
from PIL import Image, ImageDraw, ImageFont

from gzcv.metrics import angular_error
from gzcv.tools.utils import devide_batch_to_samples
from gzcv.utils.math import pitchyaw_to_vector


class Visualizer:
    def __init__(self, processes: List, writer):
        self.processes = processes
        self.writer = writer

    def __call__(
        self, loader, preds: List[Dict[str, Any]] = None, max_items=256
    ) -> None:
        n_samples = 0
        is_sample_sufficient = False

        if preds is not None:
            preds = devide_batch_to_samples(preds)

        for data in loader:
            if is_sample_sufficient:
                break
            samples = []
            for i in range(len(data["id"])):
                sample = {key: value[i] for key, value in data.items()}
                img_id = sample["id"]
                if preds is not None:
                    # print('preds.keys = ', preds.keys())
                    # print('img_id = ', img_id)
                    sample.update(preds[img_id])
                sample = self.apply_process(sample)
                samples.append(sample)
                n_samples += 1
                is_sample_sufficient = n_samples >= max_items
                if is_sample_sufficient:
                    break
            self.writer(samples)

    def apply_process(self, sample):
        for process in self.processes:
            sample = process(sample)
        return sample


class GazePlot:
    def __init__(self, contents: Dict[str, Any]):
        """
        `contents` is a dict that includes the image key and corresponding gaze keys.
        For example,
        contents
            - img_0:
                - pred_gaze: tab:orange
                - gt_gaze: tab:blue
            - img_1:
                - pred_gaze: tab:orange
                - gt_gaze: tab:blue
        """
        self.contents = contents

    def __call__(self, sample: Dict[str, Any]):
        for img_id, gaze_property in self.contents.items():
            img = sample[img_id]
            img_pil = Image.fromarray(img)
            canvas = aggdraw.Draw(img_pil)
            for gaze_key, color in gaze_property.items():
                gaze = pitchyaw_to_vector(sample[gaze_key])
                self.draw_line(canvas, gaze, color, img_pil.height, img_pil.width)
            canvas.flush()
            sample[img_id] = np.array(img_pil)
        return sample

    @staticmethod
    def draw_line(canvas, line, color, height, width, line_width=5, line_length=80):
        ch = height // 2
        cw = width // 2
        pen = aggdraw.Pen(to_hex(color), line_width)
        py = ch - line[0] * line_length
        px = cw - line[1] * line_length
        canvas.line((ch, cw, py, px), pen)


class AngularErrorEmbedding:
    def __init__(
        self, img_key, pred_gaze_key, gt_gaze_key, font_path: str = None, font_size=12
    ):
        if font_path is None:  # HACK
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono"
        font_paths = []
        font_paths.append(font_path)
        font_paths.append("/usr/share/fonts/dejavu/DejaVuSansMono.ttf")

        self.font = None
        for font_path in font_paths:
            try:
                self.font = ImageFont.truetype(font_path, size=font_size)
            except OSError:
                pass

        if self.font is None:
            raise OSError("Specified font path is invalid")

        self.img_key = img_key
        self.pred_gaze_key = pred_gaze_key
        self.gt_gaze_key = gt_gaze_key

    def __call__(self, sample: Dict[str, Any]):
        img = sample[self.img_key]
        img = Image.fromarray(img)
        error = angular_error(sample[self.gt_gaze_key], sample[self.pred_gaze_key])
        canvas = ImageDraw.Draw(img)
        canvas.text(
            (6, 0), f"angular error: {error:.2f}", font=self.font, fill=(255, 255, 255)
        )
        canvas.text((6, 12), "gt", font=self.font, fill=to_hex("tab:blue"))
        canvas.text((6, 24), "pred", font=self.font, fill=to_hex("tab:orange"))
        # canvas.flush()
        sample[self.img_key] = np.array(img)
        return sample


class DenormalizePixel:
    def __init__(self, img_keys: List[str]):
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.denorm = transforms.Normalize(
            mean=-self.mean / self.std, std=1.0 / self.std
        )
        self.img_keys = img_keys

    def __call__(self, sample: Dict[str, Any]):
        for key in self.img_keys:
            img = sample[key]
            denorm_img = self.denorm(img)
            denorm_img = denorm_img.permute(1, 2, 0).cpu().numpy()
            denorm_img = np.uint8(denorm_img * 255)
            sample[key] = denorm_img
        return sample


class StereoConcatenate:
    def __init__(self, img_keys):
        self.img_keys = img_keys

    def __call__(self, sample: Dict[str, Any]):
        imgs = []
        for img_key in self.img_keys:
            imgs.append(sample[img_key])
        tiled = self.horizontal_concat(imgs)
        sample["tiled"] = tiled
        return sample

    @staticmethod
    def horizontal_concat(imgs: List[np.array]) -> Image.Image:
        total_width = sum([img.shape[1] for img in imgs])

        tiled = Image.new("RGB", (total_width, imgs[0].shape[0]))
        cur_x = 0
        for img in imgs:
            img = Image.fromarray(img)
            tiled.paste(img, (cur_x, 0))
            cur_x += img.width
        return np.array(tiled)
