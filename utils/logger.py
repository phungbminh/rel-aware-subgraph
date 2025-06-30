import logging
import os
import sys
import time
import torch


def setup_logger(output_dir, name="main"):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{name}_log.txt")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def debug_tensor(tensor, name, logger=None):
    msg = f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}"
    if logger:
        logger.debug(msg)
    else:
        print(msg)