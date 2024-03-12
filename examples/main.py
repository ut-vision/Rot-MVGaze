import copy
import logging
import os
import pprint
import warnings
from typing import Optional

import helpers
import torch
from omegaconf import OmegaConf
from torch.optim import *
from torch.optim.lr_scheduler import *

from gzcv.tools.trainer import Trainer
from gzcv.utils import (build_from_config, extract_configs, fix_seed, get_args,
                        pretty, record_gitinfo, save_as_json)
from gzcv.utils.auto_parameter import AutoParameter
from gzcv.utils.logging import setup_logger


def train(device: str, cfg, experiment_name: Optional[str] = None):
    visualizer = helpers.misc.build_from_config(cfg.visualizer)
    model = helpers.models.build_from_config(cfg.model).to(device)
    metrics = helpers.metrics.build_from_config(cfg.metrics)
    train_loader = helpers.datasets.build_from_config(cfg.train_loader)
    val_loader = helpers.datasets.build_from_config(cfg.val_loader)
    auto_tuner = AutoParameter(cfg, model, len(train_loader.dataset))
    cfg = auto_tuner()
    optimizer = build_from_config(cfg.optimizer, globals(), params=model.parameters())
    scheduler = build_from_config(cfg.scheduler, globals(), optimizer=optimizer)
    reporter = helpers.misc.build_from_config(cfg.reporter, num_iter=len(train_loader))
    evaluator = helpers.misc.build_from_config(
        cfg.evaluator, device=device, metrics=metrics
    )
    predictor = helpers.misc.build_from_config(cfg.predictor, device=device)
    if experiment_name is None:
        experiment_name = "train"

    trainer = Trainer(
        model=model,
        evaluator=evaluator,
        predictor=predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        visualizer=visualizer,
        metrics=metrics,
        device=device,
        reporter=reporter,
        experiment_name=experiment_name,
        **cfg.trainer,
    )
    trainer.train()
    return model


def test(device: str, cfg, experiment_name: str = None):
    test_loader = build_from_config(cfg.test_loader, helpers.datasets.types)
    model = helpers.models.build_from_config(cfg.model).to(device)
    metrics = helpers.metrics.build_from_config(cfg.metrics)
    evaluator = helpers.misc.build_from_config(
        cfg.evaluator, device=device, metrics=metrics
    )
    visualizer = helpers.misc.build_from_config(cfg.visualizer)
    predictor = helpers.misc.build_from_config(cfg.predictor, device=device)

    preds = predictor(model, test_loader)
    _, preds = evaluator(preds)
    visualizer(test_loader, preds, max_items=256)
    save_as_json(
        preds,
        os.path.join(cfg.log_dir, f"{experiment_name}-preds.json"),
        save_keys=cfg.save_keys,
    )


@pretty
def run(args, cfg, experiment_name: str = "train"):
    args = copy.deepcopy(args)
    cfg = copy.deepcopy(cfg)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.is_train:
        train(device, cfg, experiment_name)

    if args.is_test:
        if args.resume is None:
            args.resume = os.path.join(
                cfg.log_dir, "ckpt", f"{experiment_name}-model-last.pth"
            )
        cfg.model.resume = args.resume
        test(device, cfg, experiment_name)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    fix_seed(0)
    args = get_args()
    is_debug = args.level == "debug"

    cfg = extract_configs(args.model, args.exp)
    cfg.log_dir += f"-{args.k_fold_index}"
    if is_debug:
        cfg.log_dir = "../logs/debug"
    setup_logger(cfg.log_dir, exist_ok=is_debug, level=args.level)
    logger = logging.getLogger()
    git_msg = record_gitinfo(cfg.log_dir)
    logger.info(git_msg)

    logger.info(f"\n{pprint.pformat(vars(args), width=40)}")

    OmegaConf.save(cfg, os.path.join(cfg.log_dir, os.path.basename(args.model)))

    if args.run_k_fold:
        k_fold_cfg = OmegaConf.load("../configs/split/xgaze_4_fold_seed_0.yaml")
        test_block_index = args.k_fold_index
        logger.info(f"Test block index = {test_block_index}")
        cfg = OmegaConf.merge(cfg, k_fold_cfg)
        run(args, cfg, experiment_name=f"split-{test_block_index}")
