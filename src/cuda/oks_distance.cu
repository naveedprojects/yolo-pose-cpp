#include "cuda/oks_distance.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace posebyte {
namespace cuda {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CUDA Kernels for OKS Distance Computation
// ============================================================================

// Kernel: Compute OKS between all track-detection pairs
// Each thread computes one cell of the cost matrix
__global__ void kernelOKSDistance(
    const float* tracks,       // [num_tracks, NUM_KEYPOINTS * 3]
    const float* detections,   // [num_detections, NUM_KEYPOINTS * 3]
    const float* sigmas,       // [NUM_KEYPOINTS]
    float* costs,              // [num_tracks, num_detections]
    int num_tracks,
    int num_detections,
    int num_keypoints
) {
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= num_tracks || det_idx >= num_detections) return;

    int track_offset = track_idx * num_keypoints * 3;
    int det_offset = det_idx * num_keypoints * 3;

    // Compute scale (area) of the detection pose
    float det_min_x = 1e9f, det_min_y = 1e9f;
    float det_max_x = -1e9f, det_max_y = -1e9f;
    int det_valid_count = 0;

    for (int kp = 0; kp < num_keypoints; kp++) {
        float det_conf = detections[det_offset + kp * 3 + 2];
        if (det_conf > 0.1f) {  // Lowered from 0.0f for more stability
            float x = detections[det_offset + kp * 3 + 0];
            float y = detections[det_offset + kp * 3 + 1];
            det_min_x = fminf(det_min_x, x);
            det_min_y = fminf(det_min_y, y);
            det_max_x = fmaxf(det_max_x, x);
            det_max_y = fmaxf(det_max_y, y);
            det_valid_count++;
        }
    }

    // Also compute track scale for averaging
    float track_min_x = 1e9f, track_min_y = 1e9f;
    float track_max_x = -1e9f, track_max_y = -1e9f;

    for (int kp = 0; kp < num_keypoints; kp++) {
        float track_conf = tracks[track_offset + kp * 3 + 2];
        if (track_conf > 0.1f) {
            float x = tracks[track_offset + kp * 3 + 0];
            float y = tracks[track_offset + kp * 3 + 1];
            track_min_x = fminf(track_min_x, x);
            track_min_y = fminf(track_min_y, y);
            track_max_x = fmaxf(track_max_x, x);
            track_max_y = fmaxf(track_max_y, y);
        }
    }

    // Average scale from both track and detection for more stability
    float det_scale_sq = (det_max_x - det_min_x) * (det_max_y - det_min_y);
    float track_scale_sq = (track_max_x - track_min_x) * (track_max_y - track_min_y);
    float scale_sq = (det_scale_sq + track_scale_sq) * 0.5f;

    // Minimum scale to prevent excessive sensitivity for small poses
    // A person ~50 pixels tall has area ~2500, so 1000 is reasonable minimum
    const float MIN_SCALE_SQ = 1000.0f;
    if (scale_sq < MIN_SCALE_SQ) {
        scale_sq = MIN_SCALE_SQ;
    }

    // Handle degenerate cases
    if (det_valid_count < 2) {
        costs[track_idx * num_detections + det_idx] = 1.0f;  // Max cost
        return;
    }

    // Compute OKS
    float oks_sum = 0.0f;
    int oks_count = 0;

    // Lower confidence threshold for matching - we want to match more keypoints
    const float kp_conf_threshold = 0.2f;

    for (int kp = 0; kp < num_keypoints; kp++) {
        float det_conf = detections[det_offset + kp * 3 + 2];
        float track_conf = tracks[track_offset + kp * 3 + 2];

        // Only compare keypoints that are visible in BOTH track and detection
        if (det_conf > kp_conf_threshold && track_conf > kp_conf_threshold) {
            float det_x = detections[det_offset + kp * 3 + 0];
            float det_y = detections[det_offset + kp * 3 + 1];
            float track_x = tracks[track_offset + kp * 3 + 0];
            float track_y = tracks[track_offset + kp * 3 + 1];

            // Squared distance
            float dx = det_x - track_x;
            float dy = det_y - track_y;
            float dist_sq = dx * dx + dy * dy;

            // OKS formula: exp(-d^2 / (2 * s^2 * sigma^2))
            // Using larger sigma multiplier for more tolerance to movement
            float sigma = sigmas[kp] * 2.0f;  // 2x sigma for more tolerance
            float sigma_sq = sigma * sigma;

            float exponent = -dist_sq / (2.0f * scale_sq * sigma_sq);
            float oks_kp = expf(exponent);

            oks_sum += oks_kp;
            oks_count++;
        }
    }

    // Average OKS across visible keypoints
    float oks;
    if (oks_count >= 3) {  // Need at least 3 keypoint matches for valid OKS
        oks = oks_sum / (float)oks_count;
    } else {
        // Fallback: if too few high-confidence keypoints, try lower threshold
        oks_sum = 0.0f;
        oks_count = 0;
        for (int kp = 0; kp < num_keypoints; kp++) {
            float det_conf = detections[det_offset + kp * 3 + 2];
            float track_conf = tracks[track_offset + kp * 3 + 2];
            if (det_conf > 0.05f && track_conf > 0.05f) {
                float det_x = detections[det_offset + kp * 3 + 0];
                float det_y = detections[det_offset + kp * 3 + 1];
                float track_x = tracks[track_offset + kp * 3 + 0];
                float track_y = tracks[track_offset + kp * 3 + 1];

                float dx = det_x - track_x;
                float dy = det_y - track_y;
                float dist_sq = dx * dx + dy * dy;

                float sigma = sigmas[kp] * 2.0f;
                float sigma_sq = sigma * sigma;
                float exponent = -dist_sq / (2.0f * scale_sq * sigma_sq);
                oks_sum += expf(exponent);
                oks_count++;
            }
        }
        oks = (oks_count > 0) ? (oks_sum / (float)oks_count) : 0.0f;
    }

    // Cost = 1 - OKS (lower is better)
    costs[track_idx * num_detections + det_idx] = 1.0f - oks;
}

// Kernel: Compute IoU between track and detection bounding boxes
__global__ void kernelIoUDistance(
    const float* track_bboxes,    // [num_tracks, 4] - x1, y1, x2, y2
    const float* det_bboxes,      // [num_detections, 4]
    float* costs,                  // [num_tracks, num_detections]
    int num_tracks,
    int num_detections
) {
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= num_tracks || det_idx >= num_detections) return;

    // Load bounding boxes
    float t_x1 = track_bboxes[track_idx * 4 + 0];
    float t_y1 = track_bboxes[track_idx * 4 + 1];
    float t_x2 = track_bboxes[track_idx * 4 + 2];
    float t_y2 = track_bboxes[track_idx * 4 + 3];

    float d_x1 = det_bboxes[det_idx * 4 + 0];
    float d_y1 = det_bboxes[det_idx * 4 + 1];
    float d_x2 = det_bboxes[det_idx * 4 + 2];
    float d_y2 = det_bboxes[det_idx * 4 + 3];

    // Intersection
    float inter_x1 = fmaxf(t_x1, d_x1);
    float inter_y1 = fmaxf(t_y1, d_y1);
    float inter_x2 = fminf(t_x2, d_x2);
    float inter_y2 = fminf(t_y2, d_y2);

    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    // Union
    float t_area = (t_x2 - t_x1) * (t_y2 - t_y1);
    float d_area = (d_x2 - d_x1) * (d_y2 - d_y1);
    float union_area = t_area + d_area - inter_area;

    // IoU
    float iou = (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;

    // Cost = 1 - IoU
    costs[track_idx * num_detections + det_idx] = 1.0f - iou;
}

// Kernel: Extract bounding boxes from poses
__global__ void kernelExtractBboxes(
    const float* poses,    // [num_poses, NUM_KEYPOINTS * 3]
    float* bboxes,         // [num_poses, 4]
    int num_poses,
    int num_keypoints
) {
    int pose_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pose_idx >= num_poses) return;

    int offset = pose_idx * num_keypoints * 3;

    float min_x = 1e9f, min_y = 1e9f;
    float max_x = -1e9f, max_y = -1e9f;

    for (int kp = 0; kp < num_keypoints; kp++) {
        float conf = poses[offset + kp * 3 + 2];
        if (conf > 0.0f) {
            float x = poses[offset + kp * 3 + 0];
            float y = poses[offset + kp * 3 + 1];
            min_x = fminf(min_x, x);
            min_y = fminf(min_y, y);
            max_x = fmaxf(max_x, x);
            max_y = fmaxf(max_y, y);
        }
    }

    // Add small margin
    float margin = 10.0f;
    bboxes[pose_idx * 4 + 0] = min_x - margin;
    bboxes[pose_idx * 4 + 1] = min_y - margin;
    bboxes[pose_idx * 4 + 2] = max_x + margin;
    bboxes[pose_idx * 4 + 3] = max_y + margin;
}

// Kernel: Combine OKS and IoU costs
__global__ void kernelCombineCosts(
    const float* oks_costs,
    const float* iou_costs,
    float* combined_costs,
    int num_tracks,
    int num_detections,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tracks * num_detections;
    if (idx >= total) return;

    combined_costs[idx] = alpha * oks_costs[idx] + (1.0f - alpha) * iou_costs[idx];
}

// ============================================================================
// OKSDistanceCUDA Class Implementation
// ============================================================================

OKSDistanceCUDA::OKSDistanceCUDA(int max_tracks, int max_detections)
    : max_tracks_(max_tracks), max_detections_(max_detections),
      d_tracks_(nullptr), d_detections_(nullptr), d_costs_(nullptr),
      d_sigmas_(nullptr), d_track_bboxes_(nullptr), d_det_bboxes_(nullptr),
      h_tracks_buffer_(nullptr), h_detections_buffer_(nullptr) {

    // Allocate device memory
    size_t pose_size = NUM_KEYPOINTS * 3 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_tracks_, max_tracks * pose_size));
    CUDA_CHECK(cudaMalloc(&d_detections_, max_detections * pose_size));
    CUDA_CHECK(cudaMalloc(&d_costs_, max_tracks * max_detections * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigmas_, NUM_KEYPOINTS * sizeof(float)));

    // Pre-allocate bbox buffers (avoid per-frame allocation)
    CUDA_CHECK(cudaMalloc(&d_track_bboxes_, max_tracks * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_det_bboxes_, max_detections * 4 * sizeof(float)));

    // Copy sigma values to device
    CUDA_CHECK(cudaMemcpy(d_sigmas_, COCO_SIGMAS, NUM_KEYPOINTS * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Pre-allocate pinned host buffers (faster H2D/D2H transfers)
    CUDA_CHECK(cudaMallocHost(&h_tracks_buffer_, max_tracks * NUM_KEYPOINTS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_detections_buffer_, max_detections * NUM_KEYPOINTS * 3 * sizeof(float)));

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

OKSDistanceCUDA::~OKSDistanceCUDA() {
    cudaFree(d_tracks_);
    cudaFree(d_detections_);
    cudaFree(d_costs_);
    cudaFree(d_sigmas_);
    cudaFree(d_track_bboxes_);
    cudaFree(d_det_bboxes_);
    cudaFreeHost(h_tracks_buffer_);
    cudaFreeHost(h_detections_buffer_);
    cudaStreamDestroy(stream_);
}

void OKSDistanceCUDA::computeOKSDistance(
    const PoseDetection* tracks,
    const PoseDetection* detections,
    float* out_costs,
    int num_tracks,
    int num_detections
) {
    if (num_tracks == 0 || num_detections == 0) return;

    // Flatten tracks using pre-allocated buffer (no vector allocation!)
    for (int t = 0; t < num_tracks; t++) {
        for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
            h_tracks_buffer_[t * NUM_KEYPOINTS * 3 + kp * 3 + 0] = tracks[t].keypoints[kp].x;
            h_tracks_buffer_[t * NUM_KEYPOINTS * 3 + kp * 3 + 1] = tracks[t].keypoints[kp].y;
            h_tracks_buffer_[t * NUM_KEYPOINTS * 3 + kp * 3 + 2] = tracks[t].keypoints[kp].confidence;
        }
    }

    // Flatten detections using pre-allocated buffer
    for (int d = 0; d < num_detections; d++) {
        for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
            h_detections_buffer_[d * NUM_KEYPOINTS * 3 + kp * 3 + 0] = detections[d].keypoints[kp].x;
            h_detections_buffer_[d * NUM_KEYPOINTS * 3 + kp * 3 + 1] = detections[d].keypoints[kp].y;
            h_detections_buffer_[d * NUM_KEYPOINTS * 3 + kp * 3 + 2] = detections[d].keypoints[kp].confidence;
        }
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpyAsync(d_tracks_, h_tracks_buffer_,
                               num_tracks * NUM_KEYPOINTS * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_detections_, h_detections_buffer_,
                               num_detections * NUM_KEYPOINTS * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((num_detections + block.x - 1) / block.x,
              (num_tracks + block.y - 1) / block.y);

    kernelOKSDistance<<<grid, block, 0, stream_>>>(
        d_tracks_, d_detections_, d_sigmas_, d_costs_,
        num_tracks, num_detections, NUM_KEYPOINTS
    );

    // Copy results back
    CUDA_CHECK(cudaMemcpyAsync(out_costs, d_costs_,
                               num_tracks * num_detections * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void OKSDistanceCUDA::computeIoUDistance(
    const PoseDetection* tracks,
    const PoseDetection* detections,
    float* out_costs,
    int num_tracks,
    int num_detections
) {
    if (num_tracks == 0 || num_detections == 0) return;

    // Flatten poses using pre-allocated buffers (no vector allocation!)
    for (int t = 0; t < num_tracks; t++) {
        for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
            h_tracks_buffer_[t * NUM_KEYPOINTS * 3 + kp * 3 + 0] = tracks[t].keypoints[kp].x;
            h_tracks_buffer_[t * NUM_KEYPOINTS * 3 + kp * 3 + 1] = tracks[t].keypoints[kp].y;
            h_tracks_buffer_[t * NUM_KEYPOINTS * 3 + kp * 3 + 2] = tracks[t].keypoints[kp].confidence;
        }
    }

    for (int d = 0; d < num_detections; d++) {
        for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
            h_detections_buffer_[d * NUM_KEYPOINTS * 3 + kp * 3 + 0] = detections[d].keypoints[kp].x;
            h_detections_buffer_[d * NUM_KEYPOINTS * 3 + kp * 3 + 1] = detections[d].keypoints[kp].y;
            h_detections_buffer_[d * NUM_KEYPOINTS * 3 + kp * 3 + 2] = detections[d].keypoints[kp].confidence;
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(d_tracks_, h_tracks_buffer_,
                               num_tracks * NUM_KEYPOINTS * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_detections_, h_detections_buffer_,
                               num_detections * NUM_KEYPOINTS * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));

    // Extract bounding boxes
    int block_size = 256;
    int grid_tracks = (num_tracks + block_size - 1) / block_size;
    int grid_dets = (num_detections + block_size - 1) / block_size;

    kernelExtractBboxes<<<grid_tracks, block_size, 0, stream_>>>(
        d_tracks_, d_track_bboxes_, num_tracks, NUM_KEYPOINTS
    );
    kernelExtractBboxes<<<grid_dets, block_size, 0, stream_>>>(
        d_detections_, d_det_bboxes_, num_detections, NUM_KEYPOINTS
    );

    // Compute IoU
    dim3 block(16, 16);
    dim3 grid((num_detections + block.x - 1) / block.x,
              (num_tracks + block.y - 1) / block.y);

    kernelIoUDistance<<<grid, block, 0, stream_>>>(
        d_track_bboxes_, d_det_bboxes_, d_costs_,
        num_tracks, num_detections
    );

    // Copy results
    CUDA_CHECK(cudaMemcpyAsync(out_costs, d_costs_,
                               num_tracks * num_detections * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void OKSDistanceCUDA::computeCombinedDistance(
    const PoseDetection* tracks,
    const PoseDetection* detections,
    float* out_costs,
    int num_tracks,
    int num_detections,
    float alpha
) {
    if (num_tracks == 0 || num_detections == 0) return;

    // Allocate temporary buffers
    float* d_oks_costs;
    float* d_iou_costs;
    CUDA_CHECK(cudaMalloc(&d_oks_costs, num_tracks * num_detections * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_iou_costs, num_tracks * num_detections * sizeof(float)));

    // Compute OKS
    std::vector<float> h_oks(num_tracks * num_detections);
    std::vector<float> h_iou(num_tracks * num_detections);

    computeOKSDistance(tracks, detections, h_oks.data(), num_tracks, num_detections);
    computeIoUDistance(tracks, detections, h_iou.data(), num_tracks, num_detections);

    // Copy to device
    CUDA_CHECK(cudaMemcpyAsync(d_oks_costs, h_oks.data(),
                               h_oks.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_iou_costs, h_iou.data(),
                               h_iou.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));

    // Combine
    int total = num_tracks * num_detections;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    kernelCombineCosts<<<grid_size, block_size, 0, stream_>>>(
        d_oks_costs, d_iou_costs, d_costs_,
        num_tracks, num_detections, alpha
    );

    // Copy results
    CUDA_CHECK(cudaMemcpyAsync(out_costs, d_costs_,
                               total * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    cudaFree(d_oks_costs);
    cudaFree(d_iou_costs);
}

// ============================================================================
// GPU-Native Operations (no H2D/D2H copies, async)
// ============================================================================

void OKSDistanceCUDA::computeOKSDistanceDeviceAsync(
    const float* d_tracks,
    const float* d_detections,
    float* d_out_costs,
    int num_tracks,
    int num_detections,
    cudaStream_t stream
) {
    if (num_tracks == 0 || num_detections == 0) return;

    cudaStream_t s = stream ? stream : stream_;

    // Launch kernel directly on device data
    dim3 block(16, 16);
    dim3 grid((num_detections + block.x - 1) / block.x,
              (num_tracks + block.y - 1) / block.y);

    kernelOKSDistance<<<grid, block, 0, s>>>(
        d_tracks, d_detections, d_sigmas_, d_out_costs,
        num_tracks, num_detections, NUM_KEYPOINTS
    );
    // No sync - caller decides when to sync
}

void OKSDistanceCUDA::computeIoUDistanceDeviceAsync(
    const float* d_track_poses,
    const float* d_det_poses,
    float* d_out_costs,
    int num_tracks,
    int num_detections,
    cudaStream_t stream
) {
    if (num_tracks == 0 || num_detections == 0) return;

    cudaStream_t s = stream ? stream : stream_;

    // Extract bounding boxes from poses
    int block_size = 256;
    int grid_tracks = (num_tracks + block_size - 1) / block_size;
    int grid_dets = (num_detections + block_size - 1) / block_size;

    kernelExtractBboxes<<<grid_tracks, block_size, 0, s>>>(
        d_track_poses, d_track_bboxes_, num_tracks, NUM_KEYPOINTS
    );
    kernelExtractBboxes<<<grid_dets, block_size, 0, s>>>(
        d_det_poses, d_det_bboxes_, num_detections, NUM_KEYPOINTS
    );

    // Compute IoU
    dim3 block(16, 16);
    dim3 grid((num_detections + block.x - 1) / block.x,
              (num_tracks + block.y - 1) / block.y);

    kernelIoUDistance<<<grid, block, 0, s>>>(
        d_track_bboxes_, d_det_bboxes_, d_out_costs,
        num_tracks, num_detections
    );
    // No sync - caller decides when to sync
}

}  // namespace cuda
}  // namespace posebyte
