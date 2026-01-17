#pragma once

#include "types.h"
#include <cuda_runtime.h>

namespace posebyte {
namespace cuda {

// GPU-native YOLO pose postprocessing
// All operations stay on device - no D2H copies
class GPUPostprocess {
public:
    GPUPostprocess(int max_detections = 1024, int num_anchors = 8400);
    ~GPUPostprocess();

    // Process raw TensorRT output entirely on GPU
    // Input: d_raw_output [56, 8400] from TensorRT (device memory)
    // Output: d_detections in device memory, returns count
    int process(
        const float* d_raw_output,     // [56, 8400] device pointer
        float conf_threshold,
        float nms_threshold,
        cudaStream_t stream = 0
    );

    // Get device pointers (for chaining to tracker)
    float* getDetectionPoses() { return d_det_poses_; }      // [max_dets, 17, 3]
    float* getDetectionBboxes() { return d_det_bboxes_; }    // [max_dets, 4]
    float* getDetectionScores() { return d_det_scores_; }    // [max_dets]
    int* getNumDetections() { return d_num_dets_; }          // Scalar on device

    // Copy only count to host (4 bytes) for loop control
    int getNumDetectionsHost();

    // Debug: dump detection data to console
    void debugDumpDetections(int num_dets);

    // Get raw detections as TrackOutput for visualization (bypasses tracker)
    std::vector<TrackOutput> getRawDetections(int num_dets);

private:
    int max_detections_;
    int num_anchors_;

    // Device memory for decoded detections
    float* d_det_poses_;       // [max_dets, 17, 3] - keypoints
    float* d_det_bboxes_;      // [max_dets, 4] - x1, y1, x2, y2
    float* d_det_scores_;      // [max_dets] - confidence scores
    int* d_det_indices_;       // [max_dets] - original indices (for NMS)
    int* d_num_dets_;          // Scalar - number of valid detections

    // Temporary buffers for decoding
    float* d_all_scores_;      // [num_anchors] - all confidence scores
    int* d_above_thresh_;      // [num_anchors] - indices above threshold
    int* d_num_above_thresh_;  // Scalar

    // NMS working buffers
    unsigned long long* d_nms_mask_;  // Bit mask for suppression
    int* d_keep_indices_;      // Indices to keep after NMS
    int* d_num_keep_;          // Scalar
    int* d_sorted_indices_;    // Indices sorted by score (descending)

    // Compaction temporary buffers
    float* d_temp_poses_;      // Temporary buffer for pose compaction
    float* d_temp_bboxes_;     // Temporary buffer for bbox compaction
    float* d_temp_scores_;     // Temporary buffer for score compaction

    // OKS sigmas on device
    float* d_sigmas_;

    cudaStream_t stream_;
};

}  // namespace cuda
}  // namespace posebyte
