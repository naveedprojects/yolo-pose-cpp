#pragma once

#include "types.h"
#include <cuda_runtime.h>
#include <vector>

namespace posebyte {
namespace cuda {

// CUDA Non-Maximum Suppression for Pose Detections
class NMSCuda {
public:
    NMSCuda(int max_detections = 1024);
    ~NMSCuda();

    // Apply NMS to pose detections
    // Uses OKS-based NMS instead of IoU for poses
    // Returns indices of detections to keep
    std::vector<int> apply(
        const PoseDetection* detections,
        int num_detections,
        float oks_threshold = 0.65f,
        float score_threshold = 0.25f
    );

    // Batch NMS for multiple images
    std::vector<std::vector<int>> applyBatch(
        const PoseDetection* detections,  // All detections concatenated
        const int* num_per_image,          // Number of detections per image
        int num_images,
        float oks_threshold = 0.65f,
        float score_threshold = 0.25f
    );

private:
    int max_detections_;

    float* d_poses_;      // [max_detections, NUM_KEYPOINTS * 3]
    float* d_scores_;     // [max_detections]
    float* d_oks_matrix_; // [max_detections, max_detections]
    int* d_keep_;         // [max_detections]
    int* d_indices_;      // [max_detections] sorted indices

    cudaStream_t stream_;
};

// Kernel declarations
extern "C" {
    void launchPoseNMS(
        const float* poses,      // [num_detections, NUM_KEYPOINTS * 3]
        const float* scores,     // [num_detections]
        const float* sigmas,     // [NUM_KEYPOINTS]
        int* keep,               // [num_detections] output: 1 = keep, 0 = suppress
        int num_detections,
        int num_keypoints,
        float oks_threshold,
        float score_threshold,
        cudaStream_t stream
    );
}

}  // namespace cuda
}  // namespace posebyte
