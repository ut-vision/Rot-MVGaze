import logging
import os
import time
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger("config parser")


def setup_custom_resolver():
    def yaml_range(start, stop, step=1):
        return list(range(start, stop, step))

    def yaml_list_sub(l1, l2):
        return [x for x in l1 if x not in l2]

    def get_hosttype():
        hostname = os.uname()[1]
        if hostname.endswith("abci.local"):
            hostname = "abci-fs"
        return hostname

    def yaml_datetime():
        if not hasattr(yaml_datetime, "exp_datetime"):
            exp_datetime = time.strftime("%Y_%m%d_%H%M")
            yaml_datetime.exp_datetime = exp_datetime
        return yaml_datetime.exp_datetime

    def if_condition(cond: bool, a: Any, b: Any):
        return a if cond else b

    OmegaConf.register_new_resolver("if", if_condition)
    OmegaConf.register_new_resolver("range", yaml_range)
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.register_new_resolver("list_sub", yaml_list_sub)
    OmegaConf.register_new_resolver("open", lambda path: OmegaConf.load(path))
    OmegaConf.register_new_resolver("hostname", get_hosttype)
    OmegaConf.register_new_resolver("datetime", yaml_datetime)
    OmegaConf.register_new_resolver("join", lambda a, b: os.path.join(a, b))


setup_custom_resolver()  # NOQA


def extract_configs(*cfg_paths):
    cfgs = []
    for path in cfg_paths:
        cfg = OmegaConf.load(path)
        cfgs.append(cfg)
    cfgs = OmegaConf.merge(*cfgs)
    cfgs = unpack_other_config(cfgs)
    cli_cfg = OmegaConf.from_cli()
    merged_cfg = OmegaConf.merge(cfgs, cli_cfg)
    return merged_cfg


def parse_optuna(cfg):
    raise NotImplementedError


def unpack_other_config(cfg):
    if "unpack_other_configs" in cfg:
        paths = cfg.pop("unpack_other_configs")
        for path in paths:
            other_cfg = OmegaConf.load(path)
            cfg.merge_with(other_cfg)
        return unpack_other_config(cfg)
    else:
        return cfg


def check_cli_args(cli_cfg, base_cfg):
    for key, val in cli_cfg.items():
        if key != "" and val is not None and val not in base_cfg:
            logger.warning(f"Found unrecognized key: {key}")


def build_from_config(cfg, namespace, **kwargs):
    if not OmegaConf.is_config(cfg):
        return cfg
    elif OmegaConf.is_list(cfg):
        return [build_from_config(args, namespace) for args in cfg]
    elif "type" not in cfg:
        return cfg
    cls_type = cfg["type"]
    Class = namespace[cls_type]
    for key, args in cfg.items():
        if not key == "type":
            kwargs[key] = build_from_config(args, namespace)
    instance = Class(**kwargs)
    return instance
