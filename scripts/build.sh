#!/bin/bash
# Build script for PoseBYTE tracker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
BUILD_TYPE="Release"
CLEAN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --help)
            echo "Usage: $0 [--debug] [--clean]"
            echo "  --debug  Build in debug mode"
            echo "  --clean  Clean build directory before building"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ $CLEAN -eq 1 ] && [ -d "build" ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure
echo "Configuring with CMake (${BUILD_TYPE})..."
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "Build complete!"
echo ""
echo "Executables:"
echo "  ./build/posebyte_demo    - Main demo application"
echo "  ./build/export_engine    - ONNX to TensorRT exporter"
echo "  ./build/benchmark        - Performance benchmarks"
echo ""
echo "Quick start:"
echo "  1. Download model: python scripts/setup_model.py"
echo "  2. Export engine:  ./build/export_engine -m models/yolov8n-pose.onnx -o models/yolov8n-pose.engine -p fp16"
echo "  3. Download video: bash scripts/download_video.sh"
echo "  4. Run demo:       ./build/posebyte_demo -e models/yolov8n-pose.engine -i data/dance_video.mp4 -d"
