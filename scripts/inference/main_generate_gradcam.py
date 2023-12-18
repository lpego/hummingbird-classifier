# %%
import os, sys

import argparse
import yaml

import numpy as np
import pandas as pd

from pathlib import Path

from matplotlib import pyplot as plt

sys.path.append(".")

from src.utils import (
    Denormalize,
    cfg_to_arguments,
)


def per_video_gradcam(video_result, args, config):
    """

    Parameters
    ----------

    Returns
    -------

    """

    return None


# %%

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Hummingbird inference script")
    args.add_argument(
        "--results_path",
        type=Path,
        help="Path to the model checkpoint to use for inference",
    )
    args = args.parse_args()

with open(args.config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = cfg_to_arguments(config)
