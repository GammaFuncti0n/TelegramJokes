'''
Some util functions for code
'''
import torch
import torch.nn as nn
import numpy as np
import random
import os
import logging
from datetime import datetime

def setup_logging():
    os.makedirs('artifacts/logs/', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=f"artifacts/logs/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8"
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

def check_paths(config):
    for path in config['paths'].values():
        try:
            os.makedirs(path, exist_ok=True)
        except:
            logger = logging.getLogger(__name__)
            logger.exception("Can not create path: %s", path)