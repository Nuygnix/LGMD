import os
import random
import numpy as np
import torch
from transformers import set_seed
import sys
# from utils import logger
import logging
import torch.distributed as dist

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def print_config(args, logger):
    logger.info("========Here is your configuration========")
    args = args.__dict__
    for key, value in args.items():
        logger.info(f"\t{key} = {value}")


def print_model(model, logger):
    # print("\nModel Structure")
    # print(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"The model has {param_count / 1000**2:.1f}M trainable parameters")

def get_rank():
    if dist.is_initialized():  # 确保分布式初始化
        rank = dist.get_rank()
    else:
        rank = 0  # 默认认为是单机非分布式环境
    return rank


def config_logging(name, file_name: str, console_level: int = logging.INFO, file_level: int = logging.DEBUG):
    # 参数解释：
    # logger名字
    # 输出文件名
    # 控制台日志输出最小等级（默认：logging.INFO）
    # 文件日志输出最小等级(默认：logging.DEBUG)
    
    logger = logging.getLogger(name)

    
    if get_rank() == 0:  # 仅在主节点打印
        file_handler = logging.FileHandler(file_name, mode='w', encoding="utf8")
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        file_handler.setLevel(file_level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '[%(asctime)s %(levelname)s] %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        console_handler.setLevel(console_level)

        logger.setLevel(min(console_level, file_level))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger