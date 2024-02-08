import warnings
from typing import Dict, Tuple, Union
import numpy as np
import torch as th
from torch.nn import functional as F


def preprocess_obs(
    obs: Union[th.Tensor, Dict[str, th.Tensor]],
    observation_space,
    normalize_images: bool = True,
) -> Union[th.Tensor, Dict[str, th.Tensor]]:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.

    :param obs: Observation
    :param observation_space:
    :param normalize_images: Whether to normalize images or not
        (True by default)
    :return:
    """
    if normalize_images:
        return obs.float() / 255.0
    return obs.float()


