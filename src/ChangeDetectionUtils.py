# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import os, sys
from typing import Dict, List, Tuple, Union, Optional, Any

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

from skimage import exposure
from scipy import stats

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


# %%
def just_im_difference(
    impair: List[Path], pars: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image differences for a triplet of consecutive frames.

    Args:
        impair: List of three Path objects representing consecutive image files
        pars: Dictionary containing processing parameters:
            - imsize: int, target image size for resizing
            - imcrop: tuple, crop coordinates (left, top, right, bottom)
            - do_filt: bool, whether to apply Gaussian blur
            - filt_rad: float, radius for Gaussian blur
            - cnorm: float, normalization constant

    Returns:
        Tuple of:
            - dh: np.ndarray, normalized difference array
            - dm: np.ndarray, magnitude of differences
    """
    ims = pars["imsize"]

    im_0 = Image.open(impair[0]).convert("RGB")
    im_1 = Image.open(impair[1]).convert("RGB")
    im_2 = Image.open(impair[2]).convert("RGB")

    (l, t, r, b) = pars["imcrop"]
    if np.any(pars["imcrop"]):
        (l, t, r, b) = pars["imcrop"]
        im_0 = im_0.crop((l, t, r, b))
        im_1 = im_1.crop((l, t, r, b))
        im_2 = im_2.crop((l, t, r, b))

    im_0 = im_0.resize((ims, ims), Image.BILINEAR)
    im_1 = im_1.resize((ims, ims), Image.BILINEAR)
    im_2 = im_2.resize((ims, ims), Image.BILINEAR)

    if pars["do_filt"]:
        im_0 = im_0.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_1 = im_1.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_2 = im_2.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))

    im_0 = np.array(im_0) / 255.0
    im_1 = np.array(im_1) / 255.0
    im_2 = np.array(im_2) / 255.0

    im_0 = exposure.match_histograms(im_0, im_1, channel_axis=2)
    im_2 = exposure.match_histograms(im_2, im_1, channel_axis=2)

    d1 = im_1 - im_0
    d2 = im_2 - im_1
    dh = (1 + d1 + d2) / 2
    dm = np.linalg.norm(dh, axis=2) / pars["cnorm"]
    return dh, dm


# %%
def magnitude_score(
    impair: List[Path], pars: Dict[str, Any]
) -> Tuple[str, float, float, float, float]:
    """
    Compute magnitude-based change detection scores for a triplet of consecutive frames.

    Args:
        impair: List of three Path objects representing consecutive image files
        pars: Dictionary containing processing parameters:
            - imsize: int, target image size for resizing
            - imcrop: tuple, crop coordinates (left, top, right, bottom)
            - do_filt: bool, whether to apply Gaussian blur
            - filt_rad: float, radius for Gaussian blur
            - cnorm: float, normalization constant

    Returns:
        Tuple containing:
            - filename: str, name of the middle frame
            - std_dev: float, standard deviation of magnitude differences
            - euclidean: float, mean squared magnitude
            - median_deviation: float, mean absolute deviation from median
            - iqr: float, interquartile range of magnitudes
    """
    ims = pars["imsize"]
    try:
        im_0 = Image.open(impair[0]).convert("RGB")
        im_1 = Image.open(impair[1]).convert("RGB")
        im_2 = Image.open(impair[2]).convert("RGB")

    except OSError:
        print(f"one of {impair} is kaputt")
        return (
            impair[1].name,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    (l, t, r, b) = pars["imcrop"]
    if np.any(pars["imcrop"]):
        (l, t, r, b) = pars["imcrop"]
        im_0 = im_0.crop((l, t, r, b))
        im_1 = im_1.crop((l, t, r, b))
        im_2 = im_2.crop((l, t, r, b))

    im_0 = im_0.resize((ims, ims), Image.BILINEAR)
    im_1 = im_1.resize((ims, ims), Image.BILINEAR)
    im_2 = im_2.resize((ims, ims), Image.BILINEAR)

    if pars["do_filt"]:
        im_0 = im_0.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_1 = im_1.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_2 = im_2.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))

    im_0 = np.array(im_0) / 255.0
    im_1 = np.array(im_1) / 255.0
    im_2 = np.array(im_2) / 255.0

    im_0 = exposure.match_histograms(im_0, im_1, channel_axis=2)
    im_2 = exposure.match_histograms(im_2, im_1, channel_axis=2)

    d1 = im_1 - im_0
    d2 = im_2 - im_1
    dh = (1 + d1 + d2) / 2
    dm = np.linalg.norm(dh, axis=2) / pars["cnorm"]
    dm = dm.ravel()

    q3, q1 = np.percentile(dm, [75, 25])
    iqr = q3 - q1

    med = np.mean(np.abs(dm - 0.5))
    euc = np.mean(dm**2)
    return (
        impair[1].name,
        np.std(dm),
        euc,
        med,
        iqr,
    )


def magnitude_score_v2(
    impair: List[Path], pars: Dict[str, Any]
) -> Tuple[str, float, float, float, float]:
    """
    Compute magnitude-based change detection scores for a triplet of consecutive frames (version 2).

    This version uses absolute differences instead of raw differences and multichannel=True
    for histogram matching.

    Args:
        impair: List of three Path objects representing consecutive image files
        pars: Dictionary containing processing parameters:
            - imsize: int, target image size for resizing
            - imcrop: tuple, crop coordinates (left, top, right, bottom)
            - do_filt: bool, whether to apply Gaussian blur
            - filt_rad: float, radius for Gaussian blur
            - cnorm: float, normalization constant

    Returns:
        Tuple containing:
            - filename: str, name of the middle frame
            - std_dev: float, standard deviation of magnitude differences
            - euclidean: float, mean squared magnitude
            - median_deviation: float, mean absolute deviation from median
            - iqr: float, interquartile range of magnitudes
    """
    ims = pars["imsize"]

    try:
        im_0 = Image.open(impair[0]).convert("RGB")
        im_1 = Image.open(impair[1]).convert("RGB")
        im_2 = Image.open(impair[2]).convert("RGB")

    except OSError:
        print(f"one of {impair} is kaputt")
        return (
            impair[1].name,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    (l, t, r, b) = pars["imcrop"]
    if np.any(pars["imcrop"]):
        (l, t, r, b) = pars["imcrop"]
        im_0 = im_0.crop((l, t, r, b))
        im_1 = im_1.crop((l, t, r, b))
        im_2 = im_2.crop((l, t, r, b))

    im_0 = im_0.resize((ims, ims), Image.BILINEAR)
    im_1 = im_1.resize((ims, ims), Image.BILINEAR)
    im_2 = im_2.resize((ims, ims), Image.BILINEAR)

    if pars["do_filt"]:
        im_0 = im_0.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_1 = im_1.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))
        im_2 = im_2.filter(ImageFilter.GaussianBlur(radius=pars["filt_rad"]))

    im_0 = np.array(im_0) / 255.0
    im_1 = np.array(im_1) / 255.0
    im_2 = np.array(im_2) / 255.0

    im_0 = exposure.match_histograms(im_0, im_1, multichannel=True)
    im_2 = exposure.match_histograms(im_2, im_1, multichannel=True)

    d1 = im_1 - im_0
    d2 = im_2 - im_1
    # dh = (1 + d1 + d2) / 2
    dh = (1 + np.abs(d1) + np.abs(d2)) / 2

    dm = np.linalg.norm(dh, axis=2) / pars["cnorm"]
    dm = dm.ravel()

    q3, q1 = np.percentile(dm, [75, 25])
    iqr = q3 - q1

    med = np.mean(np.abs(dm - 0.5))
    euc = np.mean(dm**2)
    return (
        impair[1].name,
        np.std(dm),
        euc,
        med,
        iqr,
    )


# %% setup parallel pool
def main_triplet_difference(
    folder_frames: Path, save_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Process all frames in a folder using triplet difference analysis in parallel.

    This function processes consecutive triplets of frames to detect changes using
    magnitude-based scoring. It uses all available CPU cores for parallel processing.

    Args:
        folder_frames: Path object pointing to the folder containing frame images
        save_csv: Optional path to save the results as CSV file

    Returns:
        pd.DataFrame containing the analysis results with columns:
            - fname: filename of the middle frame in each triplet
            - mag_std: standard deviation of magnitude differences
            - mag_euc: euclidean (mean squared) magnitude
            - mag_med: median absolute deviation
            - mag_iqr: interquartile range of magnitudes
    """
    num_cores = multiprocessing.cpu_count()
    pool = Parallel(n_jobs=num_cores)
    print(f"Parsing triplet difference of {folder_frames} on {num_cores} cores")

    # Run on data in all subfolders
    pars = {
        "thr_mag": 0.75,  # threshold on norm \propto anomaly score
        "thr_count": 0.02,  # percentage of out-of-threshold pixels
        "cnorm": np.sqrt(3),  # maximum value of norm
        "delta_frame": 1,  # how many frame apart from base frame t0 (default: 1 = consecutive triplets)
        "imsize": 512,
        "imcrop": (
            0,
            0,
            1280,
            700,
        ),  # (720, 1280) original image, black band is 20px. Leave all 0s for no crop
        "do_filt": True,
        "filt_rad": 3,
    }
    data_ = sorted(list(folder_frames.glob("*.jpg")))

    dt = pars["delta_frame"]
    dlist = [
        [x, y, z] for x, y, z in zip(data_[: -2 * dt], data_[dt:-dt], data_[2 * dt :])
    ]

    outs = pool(delayed(magnitude_score)(imp, pars) for imp in tqdm(dlist))
    # outs = pool(delayed(magnitude_score_v2)(imp, pars) for imp in tqdm(dlist))

    outs = np.concatenate(
        (
            pars["delta_frame"] * [["frame_0.jpg", 0, 0, 0, 0]],
            outs,
            pars["delta_frame"] * [["frame_99999.jpg", 0, 0, 0, 0]],
        ),
        axis=0,
    )

    score = pd.DataFrame(
        outs,
        columns=["fname", "mag_std", "mag_euc", "mag_med", "mag_iqr"],
    )
    score.append
    if save_csv:
        score.to_csv(save_csv, index=False)

    return score
    # %%
