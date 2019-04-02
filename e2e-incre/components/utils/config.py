import logging
import os, sys
import random

import numpy as np
import yaml
import torch

logger = logging.getLogger('experiment')


def load_config(config_path):
    """Loads and reads a yaml configuration file

    :param config_path: path of the configuration file to load
    :type config_path: str
    :return: the configuration dictionary
    :rtype: dict
    """

    with open(config_path, 'r') as user_config_file:
        return yaml.load(user_config_file.read())


def fix_seed(seed):
    logger.debug("Fixing seed: %d" % seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
