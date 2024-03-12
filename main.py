import argparse
import copy
import csv
import importlib
import os
import os.path as osp
import random
import sys
import time
from datetime import datetime
from functools import partial
from glob import glob

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from dataset.mpiigaze import MPIIGazeDataset
from dataset.xgaze import GazeDataset
from omegaconf import OmegaConf
from rich.progress import track
from src.tools.label_transform import (pitchyaw_to_vector, rotation_matrix_2d,
                                       vector_to_pitchyaw)
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm
from utils.args import str2bool
from utils.gaze_utils import AverageMeter, angular_error, draw_gaze

from losses.gaze_loss import GazeLoss
from losses.stereo_loss import IterationLoss, StereoL1Loss


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer(nn.Module):

    def __init__(
        self, model, train_loader, test_loader, output_dir=None, augment=False
    ):
        super().__init__()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.augment = augment
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

        self.start_epoch = 0
        self.epochs = 15
        self.train_iter = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckpt_dir = osp.join(self.output_dir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.tensorboard_dir = osp.join(self.output_dir, "tensorboard")
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(epoch)
            error = self.test(epoch)

            if (epoch + 1) % (self.epochs // 3) == 0:
                add_file_name = (
                    "epoch_"
                    + str(epoch + 1).zfill(2)
                    + "_error="
                    + str(round(error, 2))
                )
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state": (
                            self.model.module.state_dict()
                            if isinstance(self.model, torch.nn.DataParallel)
                            else self.model.state_dict()
                        ),
                        "optim_state": self.optimizer.state_dict(),
                        "schedule_state": self.scheduler.state_dict(),
                    },
                    add=add_file_name,
                )

    def prepare_dual_input(self, batch):
        img_0 = batch["image_a"].float()  ## ( batch, 3, 224, 224)
        gaze_0 = batch["gaze_a"].float()  ## ( batch, 2)
        head_0 = batch["head_a"].float()  ## ( batch, 2)

        img_1 = self.get_input(batch, "image_b").squeeze(
            1
        )  ## only 1 image is supported now
        gaze_1 = self.get_label(batch, "gaze_b").squeeze(1)
        head_1 = self.get_label(batch, "head_b").squeeze(1)

        rot_0 = rotation_matrix_2d(head_0)  ## from canonical to head_0
        rot_1 = rotation_matrix_2d(head_1)  ## from canonical to head_1

        data = {
            "img_0": img_0,
            "rot_0": rot_0,
            "gt_gaze": gaze_0,
            "img_1": img_1,
            "rot_1": rot_1,
            "gt_gaze_1": gaze_1,
        }
        return data

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        data = self.prepare_hisadome_input(batch)
        data = self.model(data)
        loss = self.loss(data)
        angular_error = self.angular_evaluator.gaze_angular_loss(
            data["pred_gaze"].detach(), data["gt_gaze"].detach()
        )
        return loss

    def one_iteration(self, data, tag="train"):
        l1_criterion = nn.L1Loss()
        data = self.prepare_hisadome_input(data)
        data = self.model(data)
        pred_gaze = data["pred_gaze"]
        gaze_var = data["gt_gaze"]
        input_var = data["img_0"]

        loss_gaze = l1_criterion(pred_gaze, gaze_var)
        error_gaze = np.mean(
            angular_error(pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy())
        )

        if self.train_iter != 0 and self.train_iter % 10 == 0:
            self.writer.add_scalar(
                f"{tag}/loss_gaze", loss_gaze.item(), self.train_iter
            )
            self.writer.add_scalar(
                f"{tag}/error_gaze", error_gaze.item(), self.train_iter
            )
            log_img = torchvision.utils.make_grid(input_var[:8], nrow=4, normalize=True)
            self.writer.add_image(f"{tag}/images", log_img, self.train_iter)

        self.train_iter += 1
        return loss_gaze, error_gaze

    def train_one_epoch(self, epoch):
        print(f"Epoch: {epoch + 1} / {self.epochs}")
        self.model.train()
        for i, data in enumerate(
            track(self.train_loader, description="Training", transient=True)
        ):

            loss, _ = self.one_iteration(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

    def test(self, epoch):
        errors_gaze = AverageMeter()

        self.model.eval()
        for i, data in enumerate(
            track(self.test_loader, description="Testing", transient=True)
        ):
            _, error_gaze = self.one_iteration(data, "test")
            errors_gaze.update(error_gaze.item(), data["image"].size(0))

        print(f"Epoch: {epoch + 1}, Error gaze: {errors_gaze.avg}")
        self.model.train()

        self.writer.add_scalar("test/epoch_error_gaze", errors_gaze.avg, epoch + 1)

        with open(osp.join(self.output_dir, "test_results.txt"), "a") as f:
            f.write("test on epoch {}, error: {}\n".format(epoch + 1, errors_gaze.avg))
        return errors_gaze.avg

    def save_checkpoint(self, state, add=None):
        """
        Save a copy of the model
        """
        if add is not None:
            filename = add + ".pth.tar"
        else:
            filename = "ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        print("save file to: ", ckpt_path)


def build_model_from_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    """
	cfg is like:
		type: networks.GazeResNet.GazeRes18
		params: {}
	"""
    module, cls = cfg["type"].rsplit(".", 1)
    MODEL = getattr(importlib.import_module(module, package=None), cls)
    model = MODEL(**cfg.get("params", dict()))
    return model


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xgaze_augment_path",
        help="the path to the base directory of augmented xgaze data",
        required=True,
    )
    parser.add_argument(
        "--mpii_path",
        help="the path to the base directory of mpiifacegaze",
        required=True,
    )
    parser.add_argument(
        "--model_cfg_path",
        help="the path to the config file of the model",
        required=True,
    )
    parser.add_argument(
        "--augment", type=str2bool, default=False, help="whether to use augmented data"
    )
    parser.add_argument(
        "--output_dir", help="the path to the output directory", required=True
    )
    args = parser.parse_args()

    set_seed()

    output_dir = osp.join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    created_subjects = [
        "subject0003.h5",
        "subject0004.h5",
        "subject0008.h5",
        "subject0033.h5",
        "subject0035.h5",
        "subject0040.h5",
        "subject0041.h5",
        "subject0080.h5",
        "subject0083.h5",
        "subject0106.h5",
    ]
    xgaze_dataset = GazeDataset(
        dataset_path=args.xgaze_augment_path,
        color_type="bgr",
        keys_to_use=created_subjects,
        image_size=224,
        image_key="face_patch",
        transform_Normalize={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        camera_tag="all",  ## ignore this since the augmented data already only has camera 0, 1
    )
    mpii_dataset = GazeDataset(
        dataset_path=args.mpii_path,
        color_type="bgr",
        keys_to_use=[
            "p00.h5",
            "p01.h5",
            "p02.h5",
            "p03.h5",
            "p04.h5",
            "p05.h5",
            "p06.h5",
            "p07.h5",
            "p08.h5",
            "p09.h5",
            "p10.h5",
            "p11.h5",
            "p12.h5",
            "p13.h5",
            "p14.h5",
        ],
        image_size=224,
        image_key="face_patch",
        transform_Normalize={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    )
    print("xgaze_dataset num of samples: ", len(xgaze_dataset))
    print("mpii_dataset num of samples: ", len(mpii_dataset))
    train_loader = DataLoader(
        xgaze_dataset, batch_size=200, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(mpii_dataset, batch_size=200, shuffle=False, num_workers=8)

    model = build_model_from_cfg(args.model_cfg_path)

    summary(model, (3, 224, 224))

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        output_dir=output_dir,
        augment=args.augment,
    )
    trainer.train()
