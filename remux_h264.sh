#!/usr/bin/env bash
#
# remux_h264.sh
# Usage: ./remux_h264.sh [input_dir] [output_dir] [framerate]
#
# Defaults:
#   input_dir  = ./data/insects
#   output_dir = ./data/insects/remuxed
#   framerate  = 30

set -euo pipefail

INPUT_DIR="${1:-./data/insects}"
OUTPUT_DIR="${2:-./data/insects/remuxed}"
FRAMERATE="${3:-30}"

echo "Remuxing H.264 files from: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Frame rate: $FRAMERATE fps"
echo

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Enable nullglob to handle case where no .h264 files exist
shopt -s nullglob

# Counter for processed files
count=0

for src in "$INPUT_DIR"/*.h264; do
  if [[ ! -f "$src" ]]; then
    continue
  fi
  
  base="$(basename "$src" .h264)"
  dst="$OUTPUT_DIR/${base}.mkv"
  
  echo "[$((++count))] Remuxing: $(basename "$src") → $(basename "$dst")"
  
  ffmpeg -hide_banner -loglevel warning \
    -framerate "$FRAMERATE" \
    -i "$src" \
    -c copy \
    -avoid_negative_ts make_zero \
    "$dst"
    
  if [[ $? -eq 0 ]]; then
    echo "✓ Success: $dst"
  else
    echo "✗ Failed: $src"
  fi
  echo
done

if [[ $count -eq 0 ]]; then
  echo "No .h264 files found in $INPUT_DIR"
else
  echo "Completed: $count files processed"
  echo "Output files are in: $OUTPUT_DIR"
fi