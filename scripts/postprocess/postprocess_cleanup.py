import argparse
import pandas as pd
import os
from pathlib import Path
import shutil

def delete_images_in_directory(input_dir, input_csv=None, threshold=None, out_folder=None):
    # Get list of all image files in the input directory
    image_files = list(Path(input_dir).glob('*'))
    
    # Calculate total size of files
    total_size = sum(f.stat().st_size for f in image_files)  # in bytes
    if total_size < 1024:
        size_str = f"{total_size:.2f} B"
    elif total_size < 1024 ** 2:
        size_str = f"{total_size / 1024:.2f} KB"
    elif total_size < 1024 ** 3:
        size_str = f"{total_size / (1024 ** 2):.2f} MB"
    else:
        size_str = f"{total_size / (1024 ** 3):.2f} GB"
    
    # Ask for user confirmation
    print(f"######################### WARNING #########################")
    print(f"You're about to delete {len(image_files)} files, for a total of {size_str}; this is irreversible.")
    proceed = input("Do you wish to proceed? yes/no: ").strip().lower()
    
    if proceed != 'yes':
        print("Operation cancelled.")
        return
    
    # If input_csv is provided, check that images above threshold have been copied to out_folder
    if input_csv and threshold is not None and out_folder:
        df = pd.read_csv(input_csv)
        filtered_df = df[df['score_class'] > threshold]
        for image_path in filtered_df['image_path']:
            filename = Path(image_path).name
            if not (Path(out_folder) / filename).exists():
                print(f"Warning: {filename} has not been copied to {out_folder}. Operation aborted.")
                return
    
    # Delete the files
    for image_file in image_files:
        os.remove(image_file)
        print(f"Deleted {image_file}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete images based on model predictions.")
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
    # parser.add_argument(
    #     "--verbose",
    #     "-v",
    #     type=bool,
    #     required=False,
    #     default=False,
    #     help="Print more details."
    # )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing images to be deleted."
    )
    
    args = parser.parse_args()
    
    delete_images_in_directory(args.input_dir, args.input_csv, args.threshold, args.out_folder)