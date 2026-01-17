# PoseBYTE: Fully GPU-Native Pose Tracker

A high-performance multi-person pose tracking system combining YOLO-Pose detection with a **fully GPU-native** PoseBYTE tracker. The entire pipeline runs on the GPU with only a single device-to-host transfer for visualization.

## Demo

https://github.com/user-attachments/assets/REPLACE_WITH_YOUR_VIDEO_ID

*Multi-person pose tracking at 200+ FPS on RTX 3080 Ti*

> **To add your demo video:** Edit this README on GitHub, drag your video file into the editor, and replace the URL above with the generated link.

## What's Novel

This implementation pushes pose tracking entirely to the GPU:

1. **Zero-Copy GPU Pipeline**: Detection → Postprocess → Tracking all happen on GPU memory. No intermediate CPU transfers. Only the final tracked poses are copied to host for visualization.

2. **CUDA Auction Algorithm**: Linear assignment solved via parallel auction algorithm instead of Hungarian. Scales better on GPU.

3. **GPU-Native Multi-Tier Association**: Full ByteTrack-style two-tier matching (high-conf → low-conf → lost recovery) implemented in CUDA kernels.

4. **OKS-Based Pose Matching**: Uses Object Keypoint Similarity instead of bounding box IoU. More robust for pose tracking since it considers individual keypoint positions and visibility.

5. **Spatial Gating**: Reduces O(T×D) matching complexity by pre-filtering unlikely pairs based on bounding box distance.

6. **Visibility-Masked OKS**: Only visible keypoints contribute to similarity. Falls back to torso-only matching when limbs are occluded.

## Features

- **TensorRT Inference**: YOLO-Pose (v8/v11) with FP16 and INT8 quantization
- **CUDA Kalman Filter**: Smooth pose prediction between frames
- **OKS-Based Association**: Object Keypoint Similarity instead of IoU for robust pose matching
- **GPU-Accelerated NMS**: Pose-aware non-maximum suppression
- **ByteTrack Algorithm**: Two-stage association handling high and low confidence detections
- **Configurable Track Persistence**: Tune how aggressively tracks are dropped via `--max-age`

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        Video Frame                            │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                TensorRT YOLO-Pose (FP16/INT8)                 │
│                    GPU Inference Engine                       │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                       CUDA Pose NMS                           │
│              OKS-based Non-Maximum Suppression                │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                     PoseBYTE Tracker                          │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐  │
│  │ CUDA Kalman │  │  CUDA OKS   │  │   Linear Assignment   │  │
│  │   Filter    │  │  Distance   │  │   (Auction/Greedy)    │  │
│  │ (3rd order) │  │   Matrix    │  │                       │  │
│  └─────────────┘  └─────────────┘  └───────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                       Tracked Poses                           │
│                 ID, Keypoints, Confidence                     │
└───────────────────────────────────────────────────────────────┘
```

## Requirements

- CUDA 11.0+
- TensorRT 8.0+
- OpenCV 4.x
- CMake 3.18+
- C++17 compiler

### Ubuntu/Debian

```bash
# CUDA and TensorRT (install from NVIDIA)
# https://developer.nvidia.com/tensorrt

# OpenCV
sudo apt-get install libopencv-dev

# Build tools
sudo apt-get install cmake build-essential
```

## Building

```bash
# Clone repository
cd /path/to/yolo

# Build
bash scripts/build.sh

# Or manually:
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Quick Start

### 1. Download and Export Model

```bash
# Download YOLO-Pose and export to ONNX
python scripts/setup_model.py --model yolov8n-pose

# Convert to TensorRT (FP16)
./build/export_engine -m models/yolov8n-pose.onnx -o models/yolov8n-pose_fp16.engine -p fp16

# Convert to TensorRT (INT8) - requires calibration for best results
./build/export_engine -m models/yolov8n-pose.onnx -o models/yolov8n-pose_int8.engine -p int8
```

### 2. Download Test Video

```bash
bash scripts/download_video.sh
```

### 3. Run Demo

```bash
# With display
./build/posebyte_demo -e models/yolov8n-pose_fp16.engine -i data/dance_video.mp4 -d

# Save to file
./build/posebyte_demo -e models/yolov8n-pose_fp16.engine -i data/dance_video.mp4 -o output.mp4

# With custom thresholds
./build/posebyte_demo -e models/yolov8n-pose_fp16.engine -i data/dance_video.mp4 -c 0.3 -n 0.7 -t 0.85 -o output.mp4
```

### 4. Run Benchmarks

```bash
# CUDA components only
./build/benchmark -n 1000

# With TensorRT engine
./build/benchmark -e models/yolov8n-pose_fp16.engine -n 100
```

## Usage

```
PoseBYTE GPU-Native Tracker Demo
Usage: posebyte_demo [options]

Options:
  -e, --engine PATH    TensorRT engine file (required)
  -i, --input PATH     Input video file (required)
  -o, --output PATH    Output video file (optional)
  -c, --conf FLOAT     Confidence threshold (default: 0.30)
  -n, --nms FLOAT      NMS threshold (default: 0.65)
  -t, --track FLOAT    Track match threshold (default: 0.3)
  -a, --max-age INT    Max frames a track can be undetected before deletion (default: 10)
  -d, --display        Display output in window
  -v, --verbose        Show per-frame detection/tracking details
  -h, --help           Show this help message
```

### Track Persistence (`--max-age`)

The `--max-age` parameter controls how many **consecutive frames** a track can go **undetected** (not matched to any detection) before it's deleted from memory:

```
Person detected → Track ID 5 created
Frame 100: detected ✓
Frame 101: detected ✓
Frame 102: NOT detected (age=1)
Frame 103: NOT detected (age=2)
...
Frame 111: NOT detected (age=10) → Track ID 5 DELETED
Frame 112: Person reappears → NEW Track ID 6 created
```

**Recommended values:**
- **3-5**: Strict - tracks dropped quickly. Best for fast-paced videos, crowds, music videos.
- **10-15**: Balanced - handles brief occlusions (person behind object momentarily).
- **20-30**: Permissive - tracks survive longer gaps. Risk: same ID assigned to different people.

**Note**: Without appearance features (Re-ID), pure geometry-based tracking can't distinguish between different people in similar poses. Use lower values if you see ID persistence issues.

## Performance

### GPU-Native Tracker Benchmarks (RTX 3080 Ti Laptop)

The system uses a fully GPU-native pipeline where pose data stays on the GPU throughout detection and tracking, with only a single D2H copy for visualization.

#### FP16 Precision

| Model | Params | GFLOPs | Engine Size | Avg FPS | Detect (ms) | Track (ms) | Total/Frame |
|-------|--------|--------|-------------|---------|-------------|------------|-------------|
| YOLOv8n-pose | 3.3M | 9.2 | 8.5 MB | **420** | 1.28 | 0.49 | 2.32 ms |
| YOLOv8s-pose | 11.6M | 30.2 | 26 MB | **327** | 1.87 | 0.56 | 2.98 ms |
| YOLOv8m-pose | 26.4M | 81.0 | 55 MB | **196** | 3.89 | 0.64 | 5.10 ms |
| YOLOv8l-pose | 44.5M | 168.6 | 88 MB | **136** | 6.14 | 0.70 | 7.37 ms |
| YOLOv8x-pose | 69.5M | 263.2 | 137 MB | **84** | 10.49 | 0.73 | 11.83 ms |

#### INT8 Precision (with partial quantization)

All models now support INT8 quantization using our partial quantization approach (early backbone in FP16, rest in INT8).

| Model | Engine Size | Avg FPS | Detect (ms) | Track (ms) | Total/Frame | vs FP16 |
|-------|-------------|---------|-------------|------------|-------------|---------|
| YOLOv8n-pose | **4.5 MB** | **352** | 1.67 | 0.45 | 2.73 ms | -16%* |
| YOLOv8s-pose | **13 MB** | **409** | 1.26 | 0.46 | 2.28 ms | +25% faster |
| YOLOv8m-pose | **28 MB** | **309** | 2.14 | 0.46 | 3.09 ms | +58% faster |
| YOLOv8l-pose | **46 MB** | **235** | 3.06 | 0.52 | 4.07 ms | +73% faster |
| YOLOv8x-pose | **70 MB** | **150** | 5.04 | 0.64 | 6.38 ms | +79% faster |

*Note: Smaller models may show less INT8 benefit due to partial quantization overhead

#### Key Performance Insights

- **GPU-native tracking overhead is minimal**: ~0.4-0.7 ms regardless of model size
- **All models exceed real-time**: Even YOLOv8x-pose runs at 150 FPS with INT8
- **INT8 provides significant benefits for larger models**: 58-79% speed improvement
- **Engine size reduction**: ~48% smaller engines with INT8
- **Theoretical max with YOLOv8s-pose INT8**: 439 FPS

### INT8 Quantization Notes

INT8 quantization for YOLO models requires **partial quantization** - keeping the early backbone layers (model.0-model.4) in FP16 while the rest runs in INT8. This is because TensorRT lacks INT8 implementations for certain fused Conv+SiLU operations in the early layers.

Our export tool automatically applies this partial quantization when building INT8 engines:
- ~65-86 layers forced to FP16 (depending on model size)
- Remaining layers use INT8 for faster inference
- Calibration uses video frames extracted from your dataset

**Alternative Solutions:**
1. **QAT (Quantization-Aware Training)**: Train with fake quantization nodes using [nvidia-pytorch-quantization](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
2. **Explicit Q/DQ nodes**: Use ONNX quantization tools to insert QuantizeLinear/DequantizeLinear nodes before TensorRT conversion

## Key Differences from Original ByteTrack

| Component | ByteTrack | PoseBYTE |
|-----------|-----------|----------|
| State | Bounding box (8D) | Keypoints (136D) |
| Motion Model | Linear (2nd order) | Third order (pos, vel, acc, jerk) |
| Association | IoU | OKS (Object Keypoint Similarity) |
| Re-ID | Appearance features | Pure geometric (no appearance) |

## File Structure

```
yolo/
├── include/
│   ├── types.h                    # Common types and constants
│   ├── cuda/
│   │   ├── kalman_filter.h        # CUDA Kalman filter
│   │   ├── oks_distance.h         # OKS distance computation
│   │   ├── hungarian.h            # Linear assignment
│   │   └── nms.h                  # Non-maximum suppression
│   ├── tensorrt/
│   │   └── yolo_pose_engine.h     # TensorRT inference
│   ├── tracker/
│   │   └── posebyte_tracker.h     # Main tracker
│   └── utils/
│       └── video_utils.h          # Video I/O and visualization
├── src/
│   ├── cuda/                      # CUDA kernel implementations
│   ├── tensorrt/                  # TensorRT engine implementation
│   ├── tracker/                   # Tracker implementation
│   ├── utils/                     # Utility implementations
│   ├── main.cpp                   # Demo application
│   ├── export_engine.cpp          # Model export tool
│   └── benchmark.cpp              # Performance benchmarks
├── scripts/
│   ├── build.sh                   # Build script
│   ├── download_video.sh          # Video download
│   └── setup_model.py             # Model setup
├── models/                        # Model files
├── data/                          # Test data
└── CMakeLists.txt                 # Build configuration
```

## License

MIT License

## References

- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Original ByteTrack implementation
- [PoseBYTE](https://github.com/RM-8vt13r/PoseBYTE) - Pose tracking modification
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO-Pose models
- [Object Keypoint Similarity](https://cocodataset.org/#keypoints-eval) - OKS metric
