#include "cuda/gpu_postprocess.h"
#include <cstdio>
#include <vector>

namespace posebyte {
namespace cuda {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// COCO keypoint sigmas for OKS
__constant__ float c_sigmas[17] = {
    0.026f, 0.025f, 0.025f, 0.035f, 0.035f,  // nose, eyes, ears
    0.079f, 0.079f, 0.072f, 0.072f,          // shoulders, elbows
    0.062f, 0.062f, 0.107f, 0.107f,          // wrists, hips
    0.087f, 0.087f, 0.089f, 0.089f           // knees, ankles
};

// ============================================================================
// Kernel: Decode YOLO output and filter by confidence
// Input: raw_output [56, 8400] (transposed YOLO format)
// Output: filtered detections with poses, bboxes, scores
// ============================================================================
__global__ void kernelDecodeAndFilter(
    const float* __restrict__ raw_output,  // [56, 8400]
    float* __restrict__ det_poses,          // [max_dets, 17, 3]
    float* __restrict__ det_bboxes,         // [max_dets, 4]
    float* __restrict__ det_scores,         // [max_dets]
    int* __restrict__ det_indices,          // [max_dets]
    int* __restrict__ num_dets,             // Scalar
    int num_anchors,
    int max_dets,
    float conf_threshold
) {
    int anchor_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor_idx >= num_anchors) return;

    // YOLO output layout: [56, 8400]
    // Row 0-3: bbox (cx, cy, w, h)
    // Row 4: confidence
    // Row 5-55: keypoints (17 * 3 = 51)

    float conf = raw_output[4 * num_anchors + anchor_idx];

    if (conf < conf_threshold) return;

    // Atomically get slot
    int slot = atomicAdd(num_dets, 1);
    if (slot >= max_dets) {
        atomicSub(num_dets, 1);
        return;
    }

    // Decode bbox (cx, cy, w, h) -> (x1, y1, x2, y2)
    float cx = raw_output[0 * num_anchors + anchor_idx];
    float cy = raw_output[1 * num_anchors + anchor_idx];
    float w = raw_output[2 * num_anchors + anchor_idx];
    float h = raw_output[3 * num_anchors + anchor_idx];

    det_bboxes[slot * 4 + 0] = cx - w * 0.5f;  // x1
    det_bboxes[slot * 4 + 1] = cy - h * 0.5f;  // y1
    det_bboxes[slot * 4 + 2] = cx + w * 0.5f;  // x2
    det_bboxes[slot * 4 + 3] = cy + h * 0.5f;  // y2

    det_scores[slot] = conf;
    det_indices[slot] = anchor_idx;

    // Decode keypoints
    for (int kp = 0; kp < 17; kp++) {
        int kp_offset = 5 + kp * 3;
        det_poses[slot * 17 * 3 + kp * 3 + 0] = raw_output[(kp_offset + 0) * num_anchors + anchor_idx];  // x
        det_poses[slot * 17 * 3 + kp * 3 + 1] = raw_output[(kp_offset + 1) * num_anchors + anchor_idx];  // y
        det_poses[slot * 17 * 3 + kp * 3 + 2] = raw_output[(kp_offset + 2) * num_anchors + anchor_idx];  // conf
    }
}

// ============================================================================
// Kernel: Compute pairwise IoU + OKS for NMS (SYMMETRIC - no score check)
// Each thread computes overlap for one (i, j) pair
// Output: bit mask indicating overlap (score-independent)
// ============================================================================
__global__ void kernelComputeNMSMask(
    const float* __restrict__ det_poses,   // [num_dets, 17, 3]
    const float* __restrict__ det_bboxes,  // [num_dets, 4]
    const float* __restrict__ det_scores,  // [num_dets] (unused - symmetric mask)
    unsigned long long* __restrict__ nms_mask,  // [num_dets, num_dets/64]
    int num_dets,
    float iou_threshold,
    float oks_threshold
) {
    // 2D grid: each thread handles one (i, j) pair where j is packed into 64-bit words
    int i = blockIdx.y;  // Row (detection to check)
    int j_word = blockIdx.x * blockDim.x + threadIdx.x;  // Column word

    if (i >= num_dets) return;

    int words_per_row = (num_dets + 63) / 64;
    if (j_word >= words_per_row) return;

    unsigned long long mask = 0;

    // Process 64 detections at once
    for (int bit = 0; bit < 64; bit++) {
        int j = j_word * 64 + bit;
        if (j >= num_dets || j == i) continue;  // Skip self only

        // Compute IoU
        float xi1 = det_bboxes[i * 4 + 0], yi1 = det_bboxes[i * 4 + 1];
        float xi2 = det_bboxes[i * 4 + 2], yi2 = det_bboxes[i * 4 + 3];
        float xj1 = det_bboxes[j * 4 + 0], yj1 = det_bboxes[j * 4 + 1];
        float xj2 = det_bboxes[j * 4 + 2], yj2 = det_bboxes[j * 4 + 3];

        float inter_x1 = fmaxf(xi1, xj1);
        float inter_y1 = fmaxf(yi1, yj1);
        float inter_x2 = fminf(xi2, xj2);
        float inter_y2 = fminf(yi2, yj2);

        float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
        float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
        float inter_area = inter_w * inter_h;

        float area_i = (xi2 - xi1) * (yi2 - yi1);
        float area_j = (xj2 - xj1) * (yj2 - yj1);
        float union_area = area_i + area_j - inter_area;
        float iou = (union_area > 0) ? (inter_area / union_area) : 0.0f;

        // High IoU -> mark as overlapping
        if (iou > iou_threshold) {
            mask |= (1ULL << bit);
            continue;
        }

        // Compute OKS for pose similarity
        float scale_sq = fmaxf(area_i, area_j);
        if (scale_sq < 32.0f * 32.0f) scale_sq = 32.0f * 32.0f;

        float oks_sum = 0.0f;
        int oks_count = 0;

        for (int kp = 0; kp < 17; kp++) {
            float conf_i = det_poses[i * 17 * 3 + kp * 3 + 2];
            float conf_j = det_poses[j * 17 * 3 + kp * 3 + 2];

            if (conf_i > 0.2f && conf_j > 0.2f) {
                float dx = det_poses[i * 17 * 3 + kp * 3 + 0] - det_poses[j * 17 * 3 + kp * 3 + 0];
                float dy = det_poses[i * 17 * 3 + kp * 3 + 1] - det_poses[j * 17 * 3 + kp * 3 + 1];
                float dist_sq = dx * dx + dy * dy;

                float sigma = c_sigmas[kp];
                float oks_kp = expf(-dist_sq / (2.0f * scale_sq * 4.0f * sigma * sigma));
                oks_sum += oks_kp;
                oks_count++;
            }
        }

        if (oks_count >= 3) {
            float oks = oks_sum / oks_count;
            // High pose similarity -> mark as overlapping
            if (oks > oks_threshold || (oks > 0.4f && iou > 0.2f)) {
                mask |= (1ULL << bit);
            }
        }
    }

    nms_mask[i * words_per_row + j_word] = mask;
}

// ============================================================================
// Kernel: Sort detection indices by score (descending) - simple insertion sort
// Works well for small arrays (< 1024 detections)
// ============================================================================
__global__ void kernelSortByScore(
    const float* __restrict__ det_scores,
    int* __restrict__ sorted_indices,
    int num_dets
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    // Initialize indices
    for (int i = 0; i < num_dets; i++) {
        sorted_indices[i] = i;
    }

    // Insertion sort by score (descending)
    for (int i = 1; i < num_dets; i++) {
        int key_idx = sorted_indices[i];
        float key_score = det_scores[key_idx];
        int j = i - 1;

        while (j >= 0 && det_scores[sorted_indices[j]] < key_score) {
            sorted_indices[j + 1] = sorted_indices[j];
            j--;
        }
        sorted_indices[j + 1] = key_idx;
    }
}

// ============================================================================
// Kernel: Apply NMS mask to get final detections
// Processes in SCORE ORDER (highest first) for correct suppression
// ============================================================================
__global__ void kernelApplyNMSMask(
    const float* __restrict__ det_scores,
    const unsigned long long* __restrict__ nms_mask,
    const int* __restrict__ sorted_indices,  // Indices sorted by score (descending)
    int* __restrict__ keep_indices,
    int* __restrict__ num_keep,
    int num_dets,
    int words_per_row
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;  // Single thread

    unsigned long long suppressed[16] = {0};  // Up to 1024 detections

    // Process detections in SCORE ORDER (highest first)
    for (int rank = 0; rank < num_dets && *num_keep < 256; rank++) {
        int i = sorted_indices[rank];  // Get detection with rank-th highest score

        int word = i / 64;
        int bit = i % 64;

        // Check if suppressed by higher-scoring detection
        if (suppressed[word] & (1ULL << bit)) continue;

        // Keep this detection
        int slot = atomicAdd(num_keep, 1);
        keep_indices[slot] = i;

        // Suppress all detections that overlap with this one
        for (int w = 0; w < words_per_row; w++) {
            suppressed[w] |= nms_mask[i * words_per_row + w];
        }
    }
}

// ============================================================================
// Kernel: Compact detections after NMS - copy kept detections to front of arrays
// This ensures indices 0 to num_keep-1 contain the actual kept detections
// ============================================================================
__global__ void kernelCompactDetections(
    float* __restrict__ det_poses,          // [max_dets, 17, 3] - will be compacted in-place
    float* __restrict__ det_bboxes,         // [max_dets, 4]
    float* __restrict__ det_scores,         // [max_dets]
    const int* __restrict__ keep_indices,   // [num_keep] - indices of detections to keep
    float* __restrict__ temp_poses,         // Temporary buffer for poses
    float* __restrict__ temp_bboxes,        // Temporary buffer for bboxes
    float* __restrict__ temp_scores,        // Temporary buffer for scores
    int num_keep
) {
    int keep_idx = blockIdx.x;  // Which kept detection (0 to num_keep-1)
    int kp_idx = threadIdx.x;   // Which keypoint (0-16) or bbox element

    if (keep_idx >= num_keep) return;

    int src_idx = keep_indices[keep_idx];  // Original detection index

    // Copy pose keypoints (17 threads per detection)
    if (kp_idx < 17) {
        int src_offset = src_idx * 17 * 3 + kp_idx * 3;
        int dst_offset = keep_idx * 17 * 3 + kp_idx * 3;

        temp_poses[dst_offset + 0] = det_poses[src_offset + 0];
        temp_poses[dst_offset + 1] = det_poses[src_offset + 1];
        temp_poses[dst_offset + 2] = det_poses[src_offset + 2];
    }

    // Copy bbox and score (only thread 0)
    if (kp_idx == 0) {
        temp_bboxes[keep_idx * 4 + 0] = det_bboxes[src_idx * 4 + 0];
        temp_bboxes[keep_idx * 4 + 1] = det_bboxes[src_idx * 4 + 1];
        temp_bboxes[keep_idx * 4 + 2] = det_bboxes[src_idx * 4 + 2];
        temp_bboxes[keep_idx * 4 + 3] = det_bboxes[src_idx * 4 + 3];
        temp_scores[keep_idx] = det_scores[src_idx];
    }
}

// Kernel: Copy compacted data back to main arrays
__global__ void kernelCopyBack(
    float* __restrict__ det_poses,
    float* __restrict__ det_bboxes,
    float* __restrict__ det_scores,
    const float* __restrict__ temp_poses,
    const float* __restrict__ temp_bboxes,
    const float* __restrict__ temp_scores,
    int num_keep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy poses
    if (idx < num_keep * 17 * 3) {
        det_poses[idx] = temp_poses[idx];
    }

    // Copy bboxes (handled by separate range)
    int bbox_idx = idx - num_keep * 17 * 3;
    if (bbox_idx >= 0 && bbox_idx < num_keep * 4) {
        det_bboxes[bbox_idx] = temp_bboxes[bbox_idx];
    }

    // Copy scores
    int score_idx = idx - num_keep * 17 * 3 - num_keep * 4;
    if (score_idx >= 0 && score_idx < num_keep) {
        det_scores[score_idx] = temp_scores[score_idx];
    }
}

// ============================================================================
// GPUPostprocess Implementation
// ============================================================================

GPUPostprocess::GPUPostprocess(int max_detections, int num_anchors)
    : max_detections_(max_detections)
    , num_anchors_(num_anchors)
{
    // Allocate detection buffers
    CUDA_CHECK(cudaMalloc(&d_det_poses_, max_detections * 17 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_det_bboxes_, max_detections * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_det_scores_, max_detections * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_det_indices_, max_detections * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_dets_, sizeof(int)));

    // NMS buffers
    int words_per_row = (max_detections + 63) / 64;
    CUDA_CHECK(cudaMalloc(&d_nms_mask_, max_detections * words_per_row * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_keep_indices_, max_detections * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_keep_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted_indices_, max_detections * sizeof(int)));

    // Compaction temporary buffers
    CUDA_CHECK(cudaMalloc(&d_temp_poses_, max_detections * 17 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_bboxes_, max_detections * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp_scores_, max_detections * sizeof(float)));

    // Copy sigmas to device
    CUDA_CHECK(cudaMalloc(&d_sigmas_, 17 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sigmas_, COCO_SIGMAS, 17 * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

GPUPostprocess::~GPUPostprocess() {
    cudaFree(d_det_poses_);
    cudaFree(d_det_bboxes_);
    cudaFree(d_det_scores_);
    cudaFree(d_det_indices_);
    cudaFree(d_num_dets_);
    cudaFree(d_nms_mask_);
    cudaFree(d_keep_indices_);
    cudaFree(d_num_keep_);
    cudaFree(d_sorted_indices_);
    cudaFree(d_temp_poses_);
    cudaFree(d_temp_bboxes_);
    cudaFree(d_temp_scores_);
    cudaFree(d_sigmas_);
    cudaStreamDestroy(stream_);
}

int GPUPostprocess::process(
    const float* d_raw_output,
    float conf_threshold,
    float nms_threshold,
    cudaStream_t stream
) {
    cudaStream_t s = stream ? stream : stream_;

    // Reset counters
    CUDA_CHECK(cudaMemsetAsync(d_num_dets_, 0, sizeof(int), s));
    CUDA_CHECK(cudaMemsetAsync(d_num_keep_, 0, sizeof(int), s));

    // Step 1: Decode and filter by confidence
    int block_size = 256;
    int grid_size = (num_anchors_ + block_size - 1) / block_size;

    kernelDecodeAndFilter<<<grid_size, block_size, 0, s>>>(
        d_raw_output,
        d_det_poses_,
        d_det_bboxes_,
        d_det_scores_,
        d_det_indices_,
        d_num_dets_,
        num_anchors_,
        max_detections_,
        conf_threshold
    );

    // Get number of detections (need to sync for this)
    int num_dets;
    CUDA_CHECK(cudaMemcpyAsync(&num_dets, d_num_dets_, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    if (num_dets == 0) return 0;
    if (num_dets > max_detections_) num_dets = max_detections_;

    // Step 2: Compute NMS mask
    int words_per_row = (num_dets + 63) / 64;
    CUDA_CHECK(cudaMemsetAsync(d_nms_mask_, 0,
                                num_dets * words_per_row * sizeof(unsigned long long), s));

    dim3 nms_block(32);
    dim3 nms_grid(words_per_row, num_dets);

    kernelComputeNMSMask<<<nms_grid, nms_block, 0, s>>>(
        d_det_poses_,
        d_det_bboxes_,
        d_det_scores_,
        d_nms_mask_,
        num_dets,
        nms_threshold,  // Use the passed threshold
        nms_threshold   // Use same threshold for OKS
    );

    // Step 3: Sort detections by score (descending)
    kernelSortByScore<<<1, 1, 0, s>>>(
        d_det_scores_,
        d_sorted_indices_,
        num_dets
    );

    // Step 4: Apply NMS mask (process in score order)
    kernelApplyNMSMask<<<1, 1, 0, s>>>(
        d_det_scores_,
        d_nms_mask_,
        d_sorted_indices_,
        d_keep_indices_,
        d_num_keep_,
        num_dets,
        words_per_row
    );

    // Get final count
    int num_keep;
    CUDA_CHECK(cudaMemcpyAsync(&num_keep, d_num_keep_, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    if (num_keep == 0) return 0;

    // Step 5: Compact detections - copy kept detections to indices 0 to num_keep-1
    // This is critical: d_keep_indices_ contains the original indices of kept detections
    // We need to move them to the front so the tracker gets the right data
    kernelCompactDetections<<<num_keep, 17, 0, s>>>(
        d_det_poses_,
        d_det_bboxes_,
        d_det_scores_,
        d_keep_indices_,
        d_temp_poses_,
        d_temp_bboxes_,
        d_temp_scores_,
        num_keep
    );

    // Copy compacted data back to main arrays
    int total_elements = num_keep * 17 * 3 + num_keep * 4 + num_keep;
    int block = 256;
    int grid = (total_elements + block - 1) / block;
    kernelCopyBack<<<grid, block, 0, s>>>(
        d_det_poses_,
        d_det_bboxes_,
        d_det_scores_,
        d_temp_poses_,
        d_temp_bboxes_,
        d_temp_scores_,
        num_keep
    );

    CUDA_CHECK(cudaStreamSynchronize(s));

    return num_keep;
}

void GPUPostprocess::debugDumpDetections(int num_dets) {
    if (num_dets <= 0) return;

    std::vector<float> poses(num_dets * 17 * 3);
    std::vector<float> scores(num_dets);
    std::vector<float> bboxes(num_dets * 4);

    CUDA_CHECK(cudaMemcpy(poses.data(), d_det_poses_, num_dets * 17 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scores.data(), d_det_scores_, num_dets * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bboxes.data(), d_det_bboxes_, num_dets * 4 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n=== DEBUG: %d detections after NMS ===\n", num_dets);
    for (int i = 0; i < std::min(num_dets, 3); i++) {
        printf("Det %d: score=%.3f, bbox=[%.1f, %.1f, %.1f, %.1f]\n",
               i, scores[i],
               bboxes[i*4+0], bboxes[i*4+1], bboxes[i*4+2], bboxes[i*4+3]);
        printf("  All 17 keypoints (x, y, conf):\n");
        for (int kp = 0; kp < 17; kp++) {
            printf("    kp%d: (%.1f, %.1f, %.4f)\n", kp,
                   poses[i*17*3 + kp*3 + 0],
                   poses[i*17*3 + kp*3 + 1],
                   poses[i*17*3 + kp*3 + 2]);
        }
    }
    printf("===================================\n\n");
}

std::vector<TrackOutput> GPUPostprocess::getRawDetections(int num_dets) {
    std::vector<TrackOutput> detections;
    if (num_dets <= 0) return detections;

    std::vector<float> poses(num_dets * 17 * 3);
    std::vector<float> scores(num_dets);
    std::vector<float> bboxes(num_dets * 4);

    CUDA_CHECK(cudaMemcpy(poses.data(), d_det_poses_, num_dets * 17 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scores.data(), d_det_scores_, num_dets * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bboxes.data(), d_det_bboxes_, num_dets * 4 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_dets; i++) {
        TrackOutput det;
        det.track_id = i;  // Use detection index as "track id"
        det.score = scores[i];
        det.bbox[0] = bboxes[i*4 + 0];
        det.bbox[1] = bboxes[i*4 + 1];
        det.bbox[2] = bboxes[i*4 + 2];
        det.bbox[3] = bboxes[i*4 + 3];

        for (int kp = 0; kp < 17; kp++) {
            det.keypoints[kp].x = poses[i*17*3 + kp*3 + 0];
            det.keypoints[kp].y = poses[i*17*3 + kp*3 + 1];
            det.keypoints[kp].confidence = poses[i*17*3 + kp*3 + 2];
        }
        detections.push_back(det);
    }
    return detections;
}

int GPUPostprocess::getNumDetectionsHost() {
    int count;
    CUDA_CHECK(cudaMemcpy(&count, d_num_keep_, sizeof(int), cudaMemcpyDeviceToHost));
    return count;
}

}  // namespace cuda
}  // namespace posebyte
