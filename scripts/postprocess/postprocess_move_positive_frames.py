import argparse
import pandas as pd
import os
from pathlib import Path
import shutil

def filter_and_copy_images(input_csv, threshold, out_folder, verbose=False):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Filter rows based on the score_class threshold
    filtered_df = df[df['score_class'] > threshold]
    
    # Ensure the output directory exists
    os.makedirs(out_folder, exist_ok=True)
    
    # Copy the files to the output directory
    for image_path in filtered_df['image_path']:
        # Extract the filename from the image path
        filename = Path(image_path).name
        
        # Copy the image to the output folder
        shutil.copy(image_path, os.path.join(out_folder, filename))
        if verbose:
            print(f"Copied {image_path} to {out_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and copy images based on model predictions.")
    parser.add_argument(
        "--input_csv", 
        type=str, 
        help="Path to the CSV file with model predictions."
        )
    parser.add_argument(
        "--threshold", 
        type=float, 
        help="Threshold for score_class to filter images."
        )
    parser.add_argument(
        "--out_folder", 
        type=str, 
        help="Output folder to copy the filtered images."
        )
    parser.add_argument(
        "--verbose",
        "-v",
        type=bool,
        required=False,
        default=False,
        help="Print more details."
    )
    
    args = parser.parse_args()
    
    filter_and_copy_images(args.input_csv, 
                           args.threshold, 
                           args.out_folder,
                           args.verbose
                           )