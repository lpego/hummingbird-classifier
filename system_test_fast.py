import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Need to fix this jaz

import torch

torch.hub.set_dir("../hummingbird-classifier-models/models/")

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import exposure
from torchcodec.decoders import VideoDecoder
from src.HummingbirdModel import HummingbirdModel
from src.HummingbirdLoader import CustomCrop
import torchvision.transforms as T
from PIL import Image
import pytorch_lightning as pl
import cv2  # Import once at the top


def center_crop(arr, crop_size):
    if crop_size is None:
        return arr
    h, w = arr.shape[:2]
    ch, cw = crop_size
    top = max((h - ch) // 2, 0)
    left = max((w - cw) // 2, 0)
    return arr[top : top + ch, left : left + cw, :]


def preprocess_frame(arr, crop_size=None, imsize=None, do_filt=False, filt_rad=5):
    # frame = frame.numpy()
    # frame = np.transpose(frame, (1, 2, 0))
    # arr = np.array(frame)
    # Accept imsize as int or tuple/list of two ints
    if imsize is not None:
        if isinstance(imsize, (tuple, list)) and len(imsize) == 2:
            arr = cv2.resize(
                arr, (imsize[0], imsize[1]), interpolation=cv2.INTER_LINEAR
            )
        elif isinstance(imsize, int):
            arr = cv2.resize(arr, (imsize, imsize), interpolation=cv2.INTER_LINEAR)
    arr = center_crop(arr, crop_size=[500, 500])
    if do_filt:
        arr = cv2.GaussianBlur(arr, (0, 0), filt_rad)
    return arr / 255.0


def magnitude_score_from_preprocessed(
    all_frames, frame_indices, cnorm=1.0, crop_size=None
):
    try:
        im_0, im_1, im_2 = [
            center_crop(all_frames[i], crop_size) for i in frame_indices
        ]
    except Exception as e:
        return np.nan

    im_0 = preprocess_frame(im_0)
    im_1 = preprocess_frame(im_1)
    im_2 = preprocess_frame(im_2)

    im_0 = exposure.match_histograms(im_0, im_1, channel_axis=2)
    im_2 = exposure.match_histograms(im_2, im_1, channel_axis=2)
    d1 = np.abs(im_1 - im_0)
    d2 = np.abs(im_2 - im_1)
    dh = (d1 + d2) / 2
    dm = np.linalg.norm(dh, axis=2) / cnorm
    return np.std(dm)


# --- Main script ---
def main():
    parser = argparse.ArgumentParser(
        description="Fast system test: difference scores and model inference."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to video file"
    )
    parser.add_argument(
        "--model_ckpt", type=str, required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    # parser.add_argument(
    # "--crop_size", type=int, nargs=2, default=None, help="Center crop size (h w)"
    # )
    # parser.add_argument(
    #     "--imsize",
    #     type=int,
    #     nargs="+",
    #     default=None,
    #     help="Resize to (w h) or single int",
    # )
    parser.add_argument("--stride", type=int, default=1, help="Stride for triplets")
    args = parser.parse_args()

    device = torch.device(args.device)
    # --- Load model ---
    model = HummingbirdModel.load_from_checkpoint(args.model_ckpt)
    model.eval()
    model.to(device)

    # Use model.transform_ts if available, else fallback
    transform = None
    if hasattr(model, "transform_ts"):
        transform = model.transform_ts
    if transform is None:
        transform = T.Compose(
            [
                CustomCrop((100, 1, 1180, 700), p=1.0),
                T.Resize(
                    (224, 224),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),  # AT LEAST 224
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    # --- Read video ---
    decoder = VideoDecoder(args.video_path)
    len_video = len(decoder)
    # imsize = None
    # if args.imsize is not None:
    #     if len(args.imsize) == 1:
    #         imsize = args.imsize[0]
    #     elif len(args.imsize) == 2:
    #         imsize = tuple(args.imsize)

    triplets = [[i, i + 1, i + 2] for i in range(0, len_video - 2, args.stride)]
    results = []
    prev_prob = None
    ccrop = CustomCrop((100, 1, 1180, 700), p=1.0)
    # Increase batch size for faster inference if memory allows
    batch_size = 128  # Try 32, 64, 128, or higher if you have enough RAM/GPU

    # If you don't want to use joblib, just increase batch_size as above.
    for batch_start in tqdm(
        range(0, len(triplets[:5000]), batch_size),
        desc="Batch processing",
        leave=True,
        dynamic_ncols=True,
    ):
        batch_triplets = triplets[batch_start : batch_start + batch_size]
        # Prepare batch of central frames for inference
        input_tensors = []
        triplet_frames_float = []
        for triplet in batch_triplets:
            frames = []
            frames_float = []
            for idx in triplet:
                frame = decoder[idx]
                arr = frame.cpu().numpy()
                arr = np.transpose(arr, (1, 2, 0))
                # arr = ccrop(arr)

                # if imsize is not None:
                #     if isinstance(imsize, (tuple, list)) and len(imsize) == 2:
                #         arr = cv2.resize(
                #             arr, (imsize[0], imsize[1]), interpolation=cv2.INTER_LINEAR
                #         )
                #     elif isinstance(imsize, int):
                #         arr = cv2.resize(
                #             arr, (imsize, imsize), interpolation=cv2.INTER_LINEAR
                #         )
                # if args.crop_size is not None:
                #     h, w = arr.shape[:2]
                #     ch, cw = args.crop_size
                #     top = max((h - ch) // 2, 0)
                #     left = max((w - cw) // 2, 0)
                #     arr = arr[top : top + ch, left : left + cw, :]
                frames.append(arr.astype(np.uint8))
                frames_float.append(arr.astype(np.float32))
            # For inference, only need the central frame
            input_tensors.append(transform(Image.fromarray(frames[0])).unsqueeze(0))
            triplet_frames_float.append(frames_float)
        input_batch = torch.cat(input_tensors, dim=0).to(device)
        with torch.no_grad():
            logits_batch = model(input_batch)
            if isinstance(logits_batch, (tuple, list)):
                logits_batch = logits_batch[0]
            probs_batch = torch.softmax(logits_batch, dim=1).cpu().numpy()
        # Now process each triplet in the batch
        for i, frames_float in enumerate(triplet_frames_float):
            diff_score = magnitude_score_from_preprocessed(
                frames_float, [0, 1, 2], crop_size=None
            )
            pos_prob = (
                probs_batch[i][1] if probs_batch.shape[1] > 1 else probs_batch[i][0]
            )
            prob_diff = pos_prob - prev_prob if prev_prob is not None else np.nan
            results.append(
                {
                    "frame_idx": batch_triplets[i][1],
                    "diff_score": diff_score,
                    "pos_prob": pos_prob,
                    "prob_diff": prob_diff,
                }
            )
            prev_prob = pos_prob
        # Optional: free CUDA memory if using GPU
        if device.type == "cuda":
            torch.cuda.empty_cache()

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    df = pd.DataFrame(results)
    print(df.head())
    df.to_csv(f"./processed_{video_name}.csv", index=False)
    print("Results saved to system_test_fast_results.csv")


if __name__ == "__main__":
    main()


# python system_test_fast.py --video_path data/FH303_01.avi --model_ckpt <path_to_model_checkpoint> --device cpu
