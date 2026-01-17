# GPU-Native PoseBYTE Tracker Architecture

## Current Problems (Why It's Slow)

1. **H2D/D2H copies every frame** - Poses copied to GPU for distance, back to CPU for assignment
2. **Multiple sync points per frame** - Each OKS/IoU computation has cudaStreamSynchronize
3. **Sequential CPU operations** - Kalman predict/update done track-by-track
4. **No batching** - Each track processed independently
5. **Postprocess on CPU** - TensorRT output copied to host for NMS

## Target Architecture: Zero-Copy Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GPU DEVICE MEMORY                                  │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │ Input Frame  │───▶│ Preprocess   │───▶│ TensorRT Inference           │  │
│  │ (pinned)     │    │ CUDA Kernel  │    │ Output stays on device       │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘  │
│                                                        │                     │
│                                                        ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    GPU Postprocess & NMS                              │  │
│  │  - Decode YOLO output (batched kernel)                               │  │
│  │  - Score filtering (thrust::copy_if)                                 │  │
│  │  - OKS-based NMS (adjacency matrix approach)                         │  │
│  │  Output: d_detections[MAX_DETS, 57]                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                        │                     │
│                                                        ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    PERSISTENT TRACK STATE                             │  │
│  │                                                                       │  │
│  │  d_kalman_mean[MAX_TRACKS, 136]      - All track means                │  │
│  │  d_kalman_cov[MAX_TRACKS, 136, 136]  - All track covariances         │  │
│  │  d_track_poses[MAX_TRACKS, 17, 3]    - Current smoothed poses         │  │
│  │  d_track_meta[MAX_TRACKS]            - IDs, states, hit counts        │  │
│  │  d_active_mask[MAX_TRACKS]           - Which tracks are active        │  │
│  │                                                                       │  │
│  │  THESE NEVER LEAVE THE GPU UNTIL VISUALIZATION                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                        │                     │
│                                                        ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    BATCHED TRACKING KERNELS                           │  │
│  │                                                                       │  │
│  │  Kernel 1: batchedKalmanPredict ────────────────────────────────────▶ │  │
│  │     All tracks predicted in ONE kernel launch                         │  │
│  │     Each thread handles one track                                     │  │
│  │                                                                       │  │
│  │  Kernel 2: batchedOKSDistanceMatrix ─────────────────────────────────▶│  │
│  │     Compute full [tracks × detections] cost matrix                    │  │
│  │     Each thread handles one (track, detection) pair                   │  │
│  │     Output: d_cost_matrix[MAX_TRACKS, MAX_DETS]                       │  │
│  │                                                                       │  │
│  │  Kernel 3: gpuAuctionAssignment ─────────────────────────────────────▶│  │
│  │     GPU auction algorithm (avoids O(n³) Hungarian)                    │  │
│  │     Iterative bidding until convergence                               │  │
│  │     Output: d_assignments[MAX_TRACKS]                                 │  │
│  │                                                                       │  │
│  │  Kernel 4: batchedKalmanUpdate ──────────────────────────────────────▶│  │
│  │     All matched tracks updated in ONE kernel launch                   │  │
│  │     Each thread handles one track                                     │  │
│  │                                                                       │  │
│  │  ONE cudaStreamSynchronize at the end                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                        │                     │
│                                                        ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    OUTPUT (only when needed)                          │  │
│  │  D2H copy only for visualization/saving                               │  │
│  │  Or: render directly on GPU with OpenGL interop                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Persistent Device State

Track state lives on GPU for entire track lifetime:

```cpp
struct GPUTrackPool {
    // Kalman filter state (136D = 17 keypoints × 8 state vars)
    float* d_mean;           // [MAX_TRACKS, 136]
    float* d_covariance;     // [MAX_TRACKS, 136, 136]

    // Current pose estimate
    float* d_poses;          // [MAX_TRACKS, 17, 3] (x, y, conf per keypoint)

    // Track metadata
    int* d_track_ids;        // [MAX_TRACKS]
    int* d_states;           // [MAX_TRACKS] (New/Tracked/Lost)
    int* d_hits;             // [MAX_TRACKS]
    int* d_time_lost;        // [MAX_TRACKS]

    // Active track indices (compacted)
    int* d_active_indices;   // [MAX_TRACKS]
    int* d_num_active;       // Scalar
    int* d_num_lost;         // Scalar
};
```

### 2. Batched Kalman Operations

All track predictions in ONE kernel:

```cuda
__global__ void batchedKalmanPredict(
    float* __restrict__ mean,          // [num_tracks, STATE_DIM]
    float* __restrict__ covariance,    // [num_tracks, STATE_DIM, STATE_DIM]
    const float* __restrict__ F,       // [STATE_DIM, STATE_DIM] transition matrix
    const float* __restrict__ Q,       // [STATE_DIM, STATE_DIM] process noise
    const int* __restrict__ active_idx,// Which tracks to predict
    int num_tracks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tracks) return;

    int track = active_idx[tid];

    // Load mean into registers
    float m[STATE_DIM];
    #pragma unroll
    for (int i = 0; i < STATE_DIM; i++) {
        m[i] = mean[track * STATE_DIM + i];
    }

    // x' = F @ x  (state transition)
    float m_new[STATE_DIM] = {0};
    #pragma unroll
    for (int i = 0; i < STATE_DIM; i++) {
        #pragma unroll
        for (int j = 0; j < STATE_DIM; j++) {
            m_new[i] += F[i * STATE_DIM + j] * m[j];
        }
    }

    // Store back
    #pragma unroll
    for (int i = 0; i < STATE_DIM; i++) {
        mean[track * STATE_DIM + i] = m_new[i];
    }

    // P' = F @ P @ F^T + Q  (covariance update)
    // ... similar batched matrix operations
}
```

### 3. Single OKS Distance Matrix Kernel

Compute entire cost matrix in one launch:

```cuda
__global__ void batchedOKSDistanceMatrix(
    const float* __restrict__ track_poses,  // [num_tracks, 17, 3]
    const float* __restrict__ det_poses,    // [num_dets, 17, 3]
    float* __restrict__ cost_matrix,        // [num_tracks, num_dets]
    const int* __restrict__ active_tracks,
    int num_tracks,
    int num_dets
) {
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= num_tracks || det_idx >= num_dets) return;

    int track = active_tracks[track_idx];

    // Compute OKS between track and detection
    float oks = computeOKS(
        &track_poses[track * 17 * 3],
        &det_poses[det_idx * 17 * 3]
    );

    cost_matrix[track_idx * num_dets + det_idx] = 1.0f - oks;
}
```

### 4. GPU Auction Algorithm

Replace Hungarian O(n³) with GPU-friendly auction:

```cuda
// Auction algorithm: iterative bidding
// Each track "bids" on detections, highest bidder wins
// Converges in O(n) iterations typically

__global__ void auctionBidding(
    const float* cost_matrix,  // [tracks, dets]
    float* prices,             // [dets] - current prices
    int* track_to_det,         // [tracks] - current assignment
    int* det_to_track,         // [dets] - reverse mapping
    float epsilon,             // Bid increment
    int* changed,              // Did anything change?
    int num_tracks,
    int num_dets
) {
    int track = blockIdx.x * blockDim.x + threadIdx.x;
    if (track >= num_tracks) return;
    if (track_to_det[track] >= 0) return;  // Already assigned

    // Find best and second-best detection
    float best_value = -1e9f, second_value = -1e9f;
    int best_det = -1;

    for (int d = 0; d < num_dets; d++) {
        float value = -cost_matrix[track * num_dets + d] - prices[d];
        if (value > best_value) {
            second_value = best_value;
            best_value = value;
            best_det = d;
        } else if (value > second_value) {
            second_value = value;
        }
    }

    if (best_det >= 0) {
        // Compute bid = best_value - second_value + epsilon
        float bid = best_value - second_value + epsilon;

        // Atomic update (contention handled by GPU)
        atomicAdd(&prices[best_det], bid);

        // Try to win this detection
        int prev = atomicExch(&det_to_track[best_det], track);
        if (prev >= 0) {
            track_to_det[prev] = -1;  // Kick out previous owner
        }
        track_to_det[track] = best_det;

        atomicExch(changed, 1);
    }
}
```

### 5. CUDA Graph Capture (CUDA 12.8)

Capture entire tracking pipeline as graph for minimal launch overhead:

```cpp
void captureTrackingGraph() {
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    // All tracking kernels recorded
    batchedKalmanPredict<<<...>>>();
    batchedOKSDistanceMatrix<<<...>>>();

    for (int iter = 0; iter < MAX_AUCTION_ITERS; iter++) {
        auctionBidding<<<...>>>();
    }

    batchedKalmanUpdate<<<...>>>();

    cudaStreamEndCapture(stream_, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
}

// Per-frame: just launch the captured graph
void track(/* ... */) {
    cudaGraphLaunch(graph_exec_, stream_);
}
```

### 6. TensorRT Output Chain

Keep inference output on device, postprocess directly:

```cpp
// TensorRT 10: setTensorAddress with device pointers
context_->setTensorAddress("output", d_raw_output_);
context_->enqueueV3(stream_);

// Postprocess on same stream, no sync needed
decodeYOLOOutput<<<...>>>(d_raw_output_, d_detections_, stream_);
poseNMS<<<...>>>(d_detections_, d_nms_output_, stream_);

// Chain to tracking (still on GPU)
batchedKalmanPredict<<<...>>>(d_track_state_, stream_);
// ... rest of tracking pipeline
```

## Memory Layout for Coalesced Access

```
Track poses: [NUM_TRACKS, NUM_KEYPOINTS, 3]
             Structure of Arrays for coalesced reads

             All X coordinates contiguous: d_poses + 0
             All Y coordinates contiguous: d_poses + NUM_TRACKS * NUM_KEYPOINTS
             All confidences contiguous:   d_poses + 2 * NUM_TRACKS * NUM_KEYPOINTS
```

## Expected Performance

With this architecture:
- **0 H2D copies per frame** (input via pinned memory DMA)
- **0 D2H copies per frame** (only for visualization output)
- **1 sync point per frame** (end of tracking pipeline)
- **~50-100 kernels batched into CUDA graph**
- **Target: 500+ FPS with 30+ tracks** (bottleneck becomes TensorRT inference)

## Implementation Priority

1. GPU postprocess (decode + NMS on device)
2. Persistent track state pool
3. Batched Kalman kernels
4. Batched OKS distance kernel
5. GPU auction algorithm
6. CUDA graph capture
7. OpenGL interop for visualization (optional)

## References

- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [TensorRT 10 setTensorAddress](https://github.com/NVIDIA/TensorRT/issues/4096)
- [GPU Hungarian Algorithm](https://github.com/paclopes/HungarianGPU)
- [Work-Efficient Parallel NMS](https://arxiv.org/html/2502.00535v1)
- [GPU Kalman Filter](https://arxiv.org/abs/2105.01796)
