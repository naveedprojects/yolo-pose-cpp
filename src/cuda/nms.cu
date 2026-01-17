#include "cuda/nms.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <cmath>

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

// ============================================================================
// CUDA Kernels for NMS
// ============================================================================

// Kernel: Compute pairwise OKS matrix
__global__ void kernelComputeOKSMatrix(
    const float* poses,    // [num_detections, NUM_KEYPOINTS * 3]
    const float* sigmas,   // [NUM_KEYPOINTS]
    float* oks_matrix,     // [num_detections, num_detections]
    int num_detections,
    int num_keypoints
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_detections || j >= num_detections) return;

    if (i == j) {
        oks_matrix[i * num_detections + j] = 1.0f;
        return;
    }

    if (i > j) return;

    int offset_i = i * num_keypoints * 3;
    int offset_j = j * num_keypoints * 3;

    // Compute scale from both poses and use the larger one
    float min_xi = 1e9f, min_yi = 1e9f, max_xi = -1e9f, max_yi = -1e9f;
    float min_xj = 1e9f, min_yj = 1e9f, max_xj = -1e9f, max_yj = -1e9f;
    int valid_i = 0, valid_j = 0;

    for (int kp = 0; kp < num_keypoints; kp++) {
        float conf_i = poses[offset_i + kp * 3 + 2];
        float conf_j = poses[offset_j + kp * 3 + 2];

        if (conf_i > 0.2f) {
            float x = poses[offset_i + kp * 3 + 0];
            float y = poses[offset_i + kp * 3 + 1];
            min_xi = fminf(min_xi, x);
            min_yi = fminf(min_yi, y);
            max_xi = fmaxf(max_xi, x);
            max_yi = fmaxf(max_yi, y);
            valid_i++;
        }
        if (conf_j > 0.2f) {
            float x = poses[offset_j + kp * 3 + 0];
            float y = poses[offset_j + kp * 3 + 1];
            min_xj = fminf(min_xj, x);
            min_yj = fminf(min_yj, y);
            max_xj = fmaxf(max_xj, x);
            max_yj = fmaxf(max_yj, y);
            valid_j++;
        }
    }

    // Use larger pose area for scale (COCO standard)
    float area_i = (max_xi - min_xi) * (max_yi - min_yi);
    float area_j = (max_xj - min_xj) * (max_yj - min_yj);
    float scale_sq = fmaxf(area_i, area_j);

    if (scale_sq < 32.0f * 32.0f || valid_i < 3 || valid_j < 3) {
        oks_matrix[i * num_detections + j] = 0.0f;
        oks_matrix[j * num_detections + i] = 0.0f;
        return;
    }

    // Compute OKS
    float oks_sum = 0.0f;
    int oks_count = 0;

    for (int kp = 0; kp < num_keypoints; kp++) {
        float conf_i = poses[offset_i + kp * 3 + 2];
        float conf_j = poses[offset_j + kp * 3 + 2];

        if (conf_i > 0.2f && conf_j > 0.2f) {
            float x_i = poses[offset_i + kp * 3 + 0];
            float y_i = poses[offset_i + kp * 3 + 1];
            float x_j = poses[offset_j + kp * 3 + 0];
            float y_j = poses[offset_j + kp * 3 + 1];

            float dx = x_i - x_j;
            float dy = y_i - y_j;
            float dist_sq = dx * dx + dy * dy;

            float sigma = sigmas[kp];
            // OKS formula with k=2*sigma
            float oks_kp = expf(-dist_sq / (2.0f * scale_sq * 4.0f * sigma * sigma));
            oks_sum += oks_kp;
            oks_count++;
        }
    }

    float oks = (oks_count >= 3) ? (oks_sum / (float)oks_count) : 0.0f;

    oks_matrix[i * num_detections + j] = oks;
    oks_matrix[j * num_detections + i] = oks;
}

// ============================================================================
// NMSCuda Class Implementation
// ============================================================================

NMSCuda::NMSCuda(int max_detections) : max_detections_(max_detections) {
    CUDA_CHECK(cudaMalloc(&d_poses_, max_detections * NUM_KEYPOINTS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores_, max_detections * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_oks_matrix_, max_detections * max_detections * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_keep_, max_detections * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices_, max_detections * sizeof(int)));

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

NMSCuda::~NMSCuda() {
    cudaFree(d_poses_);
    cudaFree(d_scores_);
    cudaFree(d_oks_matrix_);
    cudaFree(d_keep_);
    cudaFree(d_indices_);
    cudaStreamDestroy(stream_);
}

std::vector<int> NMSCuda::apply(
    const PoseDetection* detections,
    int num_detections,
    float oks_threshold,
    float score_threshold
) {
    std::vector<int> keep_indices;
    if (num_detections == 0) return keep_indices;

    // Filter by score threshold and sort
    std::vector<std::pair<int, float>> scored_indices;
    for (int i = 0; i < num_detections; i++) {
        if (detections[i].score >= score_threshold) {
            scored_indices.push_back({i, detections[i].score});
        }
    }

    if (scored_indices.empty()) return keep_indices;

    // Sort by score descending
    std::sort(scored_indices.begin(), scored_indices.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Helper: compute IoU between two bboxes
    auto computeIoU = [](const float* bbox1, const float* bbox2) -> float {
        float x1 = std::max(bbox1[0], bbox2[0]);
        float y1 = std::max(bbox1[1], bbox2[1]);
        float x2 = std::min(bbox1[2], bbox2[2]);
        float y2 = std::min(bbox1[3], bbox2[3]);

        float inter_w = std::max(0.0f, x2 - x1);
        float inter_h = std::max(0.0f, y2 - y1);
        float inter_area = inter_w * inter_h;

        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float union_area = area1 + area2 - inter_area;

        return (union_area > 0) ? (inter_area / union_area) : 0.0f;
    };

    // Helper: compute OKS between two poses
    auto computeOKS = [](const PoseDetection& det1, const PoseDetection& det2) -> float {
        // Compute scale from both poses
        float min_x1 = 1e9f, min_y1 = 1e9f, max_x1 = -1e9f, max_y1 = -1e9f;
        float min_x2 = 1e9f, min_y2 = 1e9f, max_x2 = -1e9f, max_y2 = -1e9f;
        int valid1 = 0, valid2 = 0;

        for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
            if (det1.keypoints[kp].confidence > 0.2f) {
                min_x1 = std::min(min_x1, det1.keypoints[kp].x);
                min_y1 = std::min(min_y1, det1.keypoints[kp].y);
                max_x1 = std::max(max_x1, det1.keypoints[kp].x);
                max_y1 = std::max(max_y1, det1.keypoints[kp].y);
                valid1++;
            }
            if (det2.keypoints[kp].confidence > 0.2f) {
                min_x2 = std::min(min_x2, det2.keypoints[kp].x);
                min_y2 = std::min(min_y2, det2.keypoints[kp].y);
                max_x2 = std::max(max_x2, det2.keypoints[kp].x);
                max_y2 = std::max(max_y2, det2.keypoints[kp].y);
                valid2++;
            }
        }

        if (valid1 < 3 || valid2 < 3) return 0.0f;

        // Use larger pose area for scale (COCO standard)
        float area1 = (max_x1 - min_x1) * (max_y1 - min_y1);
        float area2 = (max_x2 - min_x2) * (max_y2 - min_y2);
        float scale_sq = std::max(area1, area2);
        if (scale_sq < 32.0f * 32.0f) scale_sq = 32.0f * 32.0f;

        // Compute OKS
        float oks_sum = 0.0f;
        int oks_count = 0;

        for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
            if (det1.keypoints[kp].confidence > 0.2f &&
                det2.keypoints[kp].confidence > 0.2f) {
                float dx = det1.keypoints[kp].x - det2.keypoints[kp].x;
                float dy = det1.keypoints[kp].y - det2.keypoints[kp].y;
                float dist_sq = dx * dx + dy * dy;
                float sigma = COCO_SIGMAS[kp];
                // OKS formula: exp(-d^2 / (2 * s^2 * k^2)) where k = 2*sigma
                float oks_kp = expf(-dist_sq / (2.0f * scale_sq * 4.0f * sigma * sigma));
                oks_sum += oks_kp;
                oks_count++;
            }
        }

        return (oks_count >= 3) ? (oks_sum / oks_count) : 0.0f;
    };

    // Greedy NMS with aggressive OKS-based suppression
    int n = static_cast<int>(scored_indices.size());
    std::vector<bool> suppressed(n, false);

    // Balanced NMS thresholds
    float iou_suppress_thresh = 0.55f;     // Suppress if IoU > 0.55
    float oks_suppress_thresh = 0.5f;      // Suppress if OKS > 0.5 (similar poses)
    float combined_iou_thresh = 0.2f;      // If moderate IoU overlap...
    float combined_oks_thresh = 0.4f;      // ...and OKS > 0.4, suppress

    for (int i = 0; i < n; i++) {
        if (suppressed[i]) continue;

        int idx = scored_indices[i].first;
        keep_indices.push_back(idx);
        const auto& det1 = detections[idx];

        for (int j = i + 1; j < n; j++) {
            if (suppressed[j]) continue;

            int other_idx = scored_indices[j].first;
            const auto& det2 = detections[other_idx];

            // Check IoU
            float iou = computeIoU(det1.bbox, det2.bbox);

            // High IoU -> definitely same person
            if (iou > iou_suppress_thresh) {
                suppressed[j] = true;
                continue;
            }

            // Compute OKS (pose similarity)
            float oks = computeOKS(det1, det2);

            // High OKS -> similar poses, suppress lower score
            if (oks > oks_suppress_thresh) {
                suppressed[j] = true;
                continue;
            }

            // Combined check: moderate IoU + moderate OKS -> likely same person
            if (iou > combined_iou_thresh && oks > combined_oks_thresh) {
                suppressed[j] = true;
                continue;
            }

            // Additional check: if poses have significant keypoint overlap,
            // compute center distance normalized by scale
            float cx1 = (det1.bbox[0] + det1.bbox[2]) / 2.0f;
            float cy1 = (det1.bbox[1] + det1.bbox[3]) / 2.0f;
            float cx2 = (det2.bbox[0] + det2.bbox[2]) / 2.0f;
            float cy2 = (det2.bbox[1] + det2.bbox[3]) / 2.0f;

            float w1 = det1.bbox[2] - det1.bbox[0];
            float h1 = det1.bbox[3] - det1.bbox[1];
            float scale = std::max(w1, h1);
            if (scale < 32.0f) scale = 32.0f;

            float center_dist = sqrtf((cx1-cx2)*(cx1-cx2) + (cy1-cy2)*(cy1-cy2));
            float normalized_dist = center_dist / scale;

            // If centers are close (within 0.3 of scale) and any OKS similarity
            if (normalized_dist < 0.3f && oks > 0.15f) {
                suppressed[j] = true;
            }
        }
    }

    return keep_indices;
}

std::vector<std::vector<int>> NMSCuda::applyBatch(
    const PoseDetection* detections,
    const int* num_per_image,
    int num_images,
    float oks_threshold,
    float score_threshold
) {
    std::vector<std::vector<int>> results(num_images);

    int offset = 0;
    for (int img = 0; img < num_images; img++) {
        results[img] = apply(detections + offset, num_per_image[img],
                            oks_threshold, score_threshold);

        for (int& idx : results[img]) {
            idx += offset;
        }

        offset += num_per_image[img];
    }

    return results;
}

}  // namespace cuda
}  // namespace posebyte
