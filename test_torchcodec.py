# %%
import numpy as np
import pandas as pd
import time
import cv2
from PIL import Image
from skimage import exposure
from joblib import Parallel, delayed
import matplotlib.pyplot as plt  # Ensure this import is present for plotting
from torchcodec.decoders import VideoDecoder
from scipy.stats import skew


def center_crop(arr, crop_size):
    """Crop the center region of arr to crop_size (h, w)."""
    if crop_size is None:
        return arr
    h, w = arr.shape[:2]
    ch, cw = crop_size
    top = max((h - ch) // 2, 0)
    left = max((w - cw) // 2, 0)
    return arr[top : top + ch, left : left + cw, :]


def preprocess_frame(frame, pars):
    ims = pars.get("imsize", None)
    crop_size = pars.get("imcrop", None)
    frame = frame.cpu().numpy()
    frame = np.transpose(frame, (1, 2, 0))
    # im = Image.fromarray((frame * 255).astype(np.uint8))
    arr = np.array(frame)
    # --- Resize first ---
    if ims is not None and isinstance(ims, (tuple, list)) and len(ims) == 2:
        arr = cv2.resize(arr, (ims[0], ims[1]), interpolation=cv2.INTER_LINEAR)
    elif ims is not None and isinstance(ims, int):
        arr = cv2.resize(arr, (ims, ims), interpolation=cv2.INTER_LINEAR)
    # --- Then center crop ---
    arr = center_crop(arr, crop_size)

    if pars["do_filt"]:
        arr = cv2.GaussianBlur(arr, (0, 0), pars["filt_rad"])
    return arr / 255.0


def magnitude_score_from_preprocessed(all_frames, frame_indices, pars):
    crop_size = pars.get("imcrop", None)
    try:
        im_0, im_1, im_2 = [
            center_crop(all_frames[i], crop_size) for i in frame_indices
        ]
    except Exception as e:
        print(f"one of frames {frame_indices} is kaputt: {e}")
        return (str(frame_indices[1]), np.nan, np.nan, np.nan, np.nan)
    im_0 = exposure.match_histograms(im_0, im_1, channel_axis=2)
    im_2 = exposure.match_histograms(im_2, im_1, channel_axis=2)
    d1 = np.abs(im_1 - im_0)
    d2 = np.abs(im_2 - im_1)
    # dh = (1 + d1 + d2) / 2
    dh = (d1 + d2) / 2
    dm = np.linalg.norm(dh, axis=2) / pars["cnorm"]
    dm = dm.ravel()
    q3, q1 = np.percentile(dm, [75, 25])
    iqr = q3 - q1
    med = np.mean(np.abs(dm - 0.5))
    euc = np.mean(dm**2)
    ske = skew(dm, nan_policy="omit")

    # Add HSV transformation and compute indices base on Value
    im_0_hsv = cv2.cvtColor((im_0 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    im_1_hsv = cv2.cvtColor((im_1 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    im_2_hsv = cv2.cvtColor((im_2 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_frames = [im_0_hsv[:, :, -1], im_1_hsv[:, :, -1], im_2_hsv[:, :, -1]]
    hsv_frames = [
        (frame - frame.min()) / (frame.max() - frame.min()) for frame in hsv_frames
    ]
    v_diff = np.abs(hsv_frames[1] - hsv_frames[0]) + np.abs(
        hsv_frames[2] - hsv_frames[1]
    )
    # v_diff = (v_diff - v_diff.min()) / (v_diff.max() - v_diff.min())
    hsv_std = np.std(v_diff)
    return (str(frame_indices[1]), np.std(dm), euc, med, iqr, ske, hsv_std)


def just_im_difference(impair, pars, decoder=None):
    # cnorm = pars["cnorm"]
    imsize = pars.get("imsize", None)
    crop_size = pars.get("imcrop", None)

    def get_arr(idx_or_path):
        if decoder is not None and isinstance(idx_or_path, int):
            arr = decoder[idx_or_path].cpu().numpy()
            arr = np.transpose(arr, (1, 2, 0))
            # arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.array(Image.open(idx_or_path).convert("RGB"))
        # --- Resize first ---
        if (
            imsize is not None
            and isinstance(imsize, (tuple, list))
            and len(imsize) == 2
        ):
            arr = cv2.resize(
                arr, (imsize[0], imsize[1]), interpolation=cv2.INTER_LINEAR
            )
        elif imsize is not None and isinstance(imsize, int):
            arr = cv2.resize(arr, (imsize, imsize), interpolation=cv2.INTER_LINEAR)
        # --- Then center crop ---
        arr = center_crop(arr, crop_size)
        arr = arr.astype(np.float32) / 255.0
        return arr

    im_0 = get_arr(impair[0])
    im_1 = get_arr(impair[1])
    im_2 = get_arr(impair[2])
    im_0 = exposure.match_histograms(im_0, im_1, channel_axis=2)
    im_2 = exposure.match_histograms(im_2, im_1, channel_axis=2)
    d1 = np.abs(im_1 - im_0)
    d2 = np.abs(im_2 - im_1)
    dh = (d1 + d2) / 2
    dm = np.linalg.norm(dh, axis=2) / pars["cnorm"]
    return dh, dm


# --- Main script ---
# %%
decoder = VideoDecoder("data/FH303_01.avi")
redims = decoder[0].shape  # Get the dimensions of the first frame
redims = (
    int(np.ceil(redims[2] * 0.66)),
    int(np.ceil(redims[1] * 0.66)),
)  # Convert to (width, height) for OpenCV

pars = {
    "imsize": redims,  # or (12, 12) for smaller images
    "imcrop": (400, 400),  # or (left, top, right, bottom)
    "do_filt": False,
    "filt_rad": 5,
    "cnorm": 1.0,
    # "crop_size": (512, 512),  # Uncomment and set for center crop
}

# Preprocess all frames once
len_video = len(decoder)
t0 = time.time()
all_frames = [preprocess_frame(decoder[i], pars) for i in range(len_video - 2)]
preprocess_time = time.time() - t0
print(f"Preprocessing {len(all_frames)} frames took {preprocess_time:.2f} seconds.")
print(
    f"Shape of preprocessed frames: {all_frames[0].shape if all_frames else 'No frames'}"
)
# %%
# Parallel processing
t0 = time.time()
num_frames = len(all_frames)
triplets = [[i, i + 1, i + 2] for i in range(len_video - 4)]
results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(magnitude_score_from_preprocessed)(all_frames, triplet, pars)
    for triplet in triplets
)
parallel_time = time.time() - t0
print(
    f"Parallel processing of {len(triplets)} triplets took {parallel_time:.2f} seconds."
)

df = pd.DataFrame(
    results, columns=["frame_id", "std", "euc", "med", "iqr", "ske", "hsv_std"]
)

# %% --- Visualization ---
Nl = 10
column_to_plot = "hsv_std"  # Change to "euc", "med", "iqr", or "ske" as needed
if not df[column_to_plot].isnull().all():
    top5 = df.nlargest(Nl, column_to_plot)
    fig, axes = plt.subplots(1, Nl, figsize=(20, 4))
    for ax, (_, row) in zip(axes, top5.iterrows()):
        idx = int(row["frame_id"])
        if 0 <= idx < len(all_frames):
            img = all_frames[idx]  # Convert BGR to RGB for display
            ax.imshow((img * 255).astype(np.uint8))
            ax.set_title(f"Frame {idx}\n{column_to_plot}={row[column_to_plot]:.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# --- Overlay dh heatmap on top 5 std frames at the end of the script ---
if not df[column_to_plot].isnull().all():
    top5 = df.nlargest(Nl, column_to_plot)
    fig, axes = plt.subplots(1, Nl, figsize=(20, 4))
    for ax, (_, row) in zip(axes, top5.iterrows()):
        idx = int(row["frame_id"])
        triplet = [idx - 1, idx, idx + 1]
        if triplet[0] >= 0 and triplet[2] < len(all_frames):
            dh, _ = just_im_difference(triplet, pars, decoder=decoder)
            img = all_frames[idx]
            ax.imshow((img * 255).astype(np.uint8))
            ax.imshow(np.linalg.norm(dh, axis=2), cmap="jet", alpha=0.6)
            ax.set_title(f"Frame {idx}\n{column_to_plot}={row[column_to_plot]:.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# %%
def process_frames_to_hsv(triplet, decoder):
    """
    Read three consecutive frames from the decoder, perform histogram matching,
    transform them into the HSV colorspace

    Parameters:
        triplet (list): List of three consecutive frame indices.
        decoder (VideoDecoder): The video decoder object.

    Returns:
        tuple: Three frames as arrays in HSV colorspace
    """

    def get_frame(idx):
        frame = decoder[idx].cpu().numpy()
        frame = np.transpose(frame, (1, 2, 0))  # Convert to HWC format
        return frame / 255.0  # Normalize to [0, 1]

    # Read frames
    im_0 = get_frame(triplet[0])
    im_1 = get_frame(triplet[1])
    im_2 = get_frame(triplet[2])

    # Perform histogram matching
    im_0 = exposure.match_histograms(im_0, im_1, channel_axis=2)
    im_2 = exposure.match_histograms(im_2, im_1, channel_axis=2)

    # Convert to HSV colorspace
    im_0_hsv = cv2.cvtColor((im_0 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    im_1_hsv = cv2.cvtColor((im_1 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    im_2_hsv = cv2.cvtColor((im_2 * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

    return im_0_hsv, im_1_hsv, im_2_hsv


# Process frames starting from frame number 5999
start_frame = 14599
# start_frame = 18720
triplet = [start_frame - 1, start_frame, start_frame + 1]

im_0_hsv, im_1_hsv, im_2_hsv = process_frames_to_hsv(triplet, decoder)

# %%
# Plot each HSV image and each custom image in normalized RGB
for ch in range(3):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    hsv_frames = [im_0_hsv[:, :, ch], im_1_hsv[:, :, ch], im_2_hsv[:, :, ch]]
    # Plot HSV images
    for i, (hsv_frame, ax) in enumerate(zip(hsv_frames, axes[0])):
        ax.imshow(hsv_frame)
        ax.set_title(f"HSV Frame {i}")
        ax.axis("off")

    # Compute differences and sum of absolute differences
    diff_frames = [
        np.abs(hsv_frames[1] - hsv_frames[0]),
        np.abs(hsv_frames[2] - hsv_frames[1]),
        np.abs(hsv_frames[1] - hsv_frames[0]) + np.abs(hsv_frames[2] - hsv_frames[1]),
    ]
    titles = ["Diff Frame 0-1", "Diff Frame 1-2", "Sum of Abs Diff"]

    # Plot difference images
    for diff_frame, title, ax in zip(diff_frames, titles, axes[1]):
        diff_frame = (diff_frame - diff_frame.min()) / (
            diff_frame.max() - diff_frame.min()
        )
        ax.imshow(diff_frame, cmap="jet")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# %%
