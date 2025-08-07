import argparse
import pandas as pd
import os
from pathlib import Path
import shutil


def filter_and_copy_images(
    input_csv: Path, threshold: float, out_folder: Path, verbose=False
) -> None:
    """
    Filters images based on a score_class threshold from a CSV file and copies them to an output directory.

    Parameters:
    ----------

    input_csv : Path
        Path to the CSV file containing image paths and their corresponding score_class.

    threshold : float
        Threshold for filtering images based on their score_class.

    out_folder : Path
        Path to the output directory where filtered images will be copied.

    verbose : bool
        If True, print detailed information about the copying process.

    """
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Filter rows based on the score_class threshold
    filtered_df = df[df["score_class"] > threshold]

    # Ensure the output directory exists
    os.makedirs(out_folder, exist_ok=True)

    # Copy the files to the output directory
    for image_path in filtered_df["image_path"]:
        # Extract the filename from the image path
        filename = Path(image_path).name

        # Copy the image to the output folder
        shutil.copy(image_path, os.path.join(out_folder, filename))
        if verbose:
            print(f"Copied {image_path} to {out_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter and copy images based on model predictions."
    )
    parser.add_argument(
        "--input_csv", type=str, help="Path to the CSV file with model predictions."
    )
    parser.add_argument(
        "--threshold", type=float, help="Threshold for score_class to filter images."
    )
    parser.add_argument(
        "--out_folder", type=str, help="Output folder to copy the filtered images."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=bool,
        required=False,
        default=False,
        help="Print more details.",
    )

    args = parser.parse_args()

    exit(
        filter_and_copy_images(
            args.input_csv, args.threshold, args.out_folder, args.verbose
        )
    )
