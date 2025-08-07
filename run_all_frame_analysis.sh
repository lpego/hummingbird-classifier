#!/bin/bash

# run_all_frame_analysis.sh
# 
# Comprehensive script to run all three frame difference analysis methods:
# 1. Color histogram differences (_video_frame_chist_diff.py)
# 2. Triplet frame differences (video_frame_diff_triplet.py)
# 3. Running mean background subtraction (video_frame_diff_running_mean.py)
#
# Usage: ./run_all_frame_analysis.sh [OPTIONS]
#
# Options:
#   -o, --output-dir DIR     Output directory for results (default: results_all_methods)
#   -c, --crop-box X Y W H   Crop box coordinates (default: 0 0 1280 700)
#   -f, --frame-skip N       Frame skip for histogram and triplet analysis (default: 1)
#   -r, --running-mean N     Buffer size for running mean (default: 20)
#   -v, --visualize          Enable visualization
#   -h, --help              Show this help message
#
# Example:
#   ./run_all_frame_analysis.sh -o results_test -f 2 -r 30 -v

set -e  # Exit on any error

# Default parameters
OUTPUT_DIR="./results_test_sanity/hummingbird"
CROP_BOX=(0 0 1280 700)
FRAME_SKIP=1
RUNNING_MEAN_N=20
VISUALIZE=""
HELP=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
run_all_frame_analysis.sh - Run all frame difference analysis methods

USAGE:
    ./run_all_frame_analysis.sh [OPTIONS]

OPTIONS:
    -o, --output-dir DIR     Output directory for results (default: results_all_methods)
    -c, --crop-box X Y W H   Crop box coordinates (default: 0 0 1280 700)
    -f, --frame-skip N       Frame skip for histogram and triplet analysis (default: 1)
    -r, --running-mean N     Buffer size for running mean (default: 20)
    -v, --visualize          Enable visualization
    -h, --help               Show this help message

METHODS:
    1. Color Histogram Differences - Compares color histograms between frames
    2. Triplet Frame Differences - Compares frame triplets at different time intervals
    3. Running Mean Background Subtraction - Compares frames against running background

EXAMPLE:
    ./run_all_frame_analysis.sh -o results_test -f 2 -r 30 -v

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--crop-box)
            CROP_BOX=("$2" "$3" "$4" "$5")
            shift 5
            ;;
        -f|--frame-skip)
            FRAME_SKIP="$2"
            shift 2
            ;;
        -r|--running-mean)
            RUNNING_MEAN_N="$2"
            shift 2
            ;;
        -v|--visualize)
            VISUALIZE="--visualize"
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    show_help
    exit 0
fi

# Check if Python scripts exist
TRIPLET_SCRIPT="video_frame_diff_triplet.py"
RUNNING_MEAN_SCRIPT="video_frame_diff_running_mean.py"
CHIST_SCRIPT="video_frame_diff_colorhist.py"
WASSERSTEIN_SCRIPT="video_frame_diff_colorhist.py"
CHI_SQUARE_SCRIPT="video_frame_diff_colorhist.py"

# Create main output directory
mkdir -p "$OUTPUT_DIR"

# Create subdirectories for each method
CHIST_DIR="$OUTPUT_DIR/euclidean"
TRIPLET_DIR="$OUTPUT_DIR/triplet_analysis"
RUNNING_MEAN_DIR="$OUTPUT_DIR/running_mean"
WASSERSTEIN_DIR="$OUTPUT_DIR/wasserstein"
CHI_SQUARE_DIR="$OUTPUT_DIR/chi_square"  # chi_square is in parent results directory

mkdir -p "$CHIST_DIR" "$TRIPLET_DIR" "$RUNNING_MEAN_DIR" "$WASSERSTEIN_DIR" "$CHI_SQUARE_DIR"

print_status "Starting comprehensive frame difference analysis..."
print_status "Output directory: $OUTPUT_DIR"
print_status "Crop box: ${CROP_BOX[*]}"
print_status "Frame skip (triplet): $FRAME_SKIP"
print_status "Running mean buffer size: $RUNNING_MEAN_N"
print_status "Visualization: $([ -n "$VISUALIZE" ] && echo "enabled" || echo "disabled")"

echo ""
echo "============================================"
echo "ANALYSIS METHODS TO BE EXECUTED:"
echo "============================================"
echo "1. Color Histogram Differences"
echo "   - Variants: 1A Euclidean, 1B Wasserstein, 1C Chi-Square"
echo "2. Triplet Frame Differences"  
echo "3. Running Mean Background Subtraction"
echo "============================================"
echo ""

# Function to run analysis with error handling
run_analysis() {
    local method_name="$1"
    local script="$2"
    local output_dir="$3"
    local extra_args="$4"
    
    print_status "Starting $method_name..."
    
    # Build command
    local cmd="python $script --output-folder $output_dir --crop-box ${CROP_BOX[*]} $VISUALIZE $extra_args"
    
    print_status "Command: $cmd"
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run the analysis
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$method_name completed successfully in ${duration}s"
        return 0
    else
        print_error "$method_name failed!"
        return 1
    fi
}

# Track overall success
OVERALL_SUCCESS=true

echo "Starting analysis pipeline..."
echo ""

# 1. Color Histogram Differences
echo "📊 METHOD 1A: COLOR HISTOGRAM DIFFERENCES (EUCLIDEAN)"
echo "===================================================="
if ! run_analysis "Color Histogram Analysis (Euclidean)" "$CHIST_SCRIPT" "$CHIST_DIR" "--frame-skip $FRAME_SKIP --distance-metric euclidean"; then
    OVERALL_SUCCESS=false
fi
echo ""

echo "📊 METHOD 1B: COLOR HISTOGRAM DIFFERENCES (WASSERSTEIN)"
echo "======================================================"
if ! run_analysis "Color Histogram Analysis (Wasserstein)" "$WASSERSTEIN_SCRIPT" "$WASSERSTEIN_DIR" "--frame-skip $FRAME_SKIP --distance-metric wasserstein"; then
    OVERALL_SUCCESS=false
fi
echo ""

echo "📊 METHOD 1C: COLOR HISTOGRAM DIFFERENCES (CHI-SQUARE)"
echo "====================================================="
if ! run_analysis "Color Histogram Analysis (Chi-Square)" "$CHI_SQUARE_SCRIPT" "$CHI_SQUARE_DIR" "--frame-skip $FRAME_SKIP --distance-metric chi_square"; then
    OVERALL_SUCCESS=false
fi
echo ""

# 2. Triplet Frame Differences
echo "🔀 METHOD 2: TRIPLET FRAME DIFFERENCES"
echo "======================================"
if ! run_analysis "Triplet Frame Analysis" "$TRIPLET_SCRIPT" "$TRIPLET_DIR" "--frame-skip $FRAME_SKIP"; then
    OVERALL_SUCCESS=false
fi
echo ""

# 3. Running Mean Background Subtraction
echo "📈 METHOD 3: RUNNING MEAN BACKGROUND SUBTRACTION"
echo "==============================================="
if ! run_analysis "Running Mean Analysis" "$RUNNING_MEAN_SCRIPT" "$RUNNING_MEAN_DIR" "--running-mean-N $RUNNING_MEAN_N"; then
    OVERALL_SUCCESS=false
fi
echo ""


# Summary
echo "============================================"
echo "ANALYSIS PIPELINE SUMMARY"
echo "============================================"

if [ "$OVERALL_SUCCESS" = true ]; then
    print_success "All analyses completed successfully!"
    echo ""
    print_status "Results saved to:"
    echo "  📊 Color Histogram: $CHIST_DIR"
    echo "  📊 Wasserstein Analysis: $WASSERSTEIN_DIR"
    echo "  📊 Chi-Square Analysis: $CHI_SQUARE_DIR"
    echo "  🔀 Triplet Analysis: $TRIPLET_DIR"
    echo "  📈 Running Mean: $RUNNING_MEAN_DIR"
    echo ""
    print_status "Next steps:"
    echo "  1. Review the CSV files for each method"
    echo "  2. Compare results across methods"
    echo "  3. Run precision/recall analysis if ground truth is available"
    echo ""
else
    print_error "Some analyses failed! Check the output above for details."
    exit 1
fi

# Optional: List output files
echo "Generated files:"
find "$OUTPUT_DIR" -name "*.csv" -o -name "*.yaml" | sort | while read file; do
    echo "  📄 $file"
done

print_success "Analysis pipeline completed!"