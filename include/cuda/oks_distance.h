#pragma once

#include "types.h"
#include <cuda_runtime.h>

namespace posebyte {
namespace cuda {

// CUDA OKS (Object Keypoint Similarity) Distance Computation
// Supports both legacy (host data) and GPU-native (device data) modes
class OKSDistanceCUDA {
public:
    OKSDistanceCUDA(int max_tracks = 256, int max_detections = 256);
    ~OKSDistanceCUDA();

    // ========================================================================
    // GPU-Native Operations (no H2D/D2H copies, async)
    // ========================================================================

    // Compute OKS distance matrix from device pointers - async
    // d_tracks: [num_tracks, 17*3] on device
    // d_detections: [num_detections, 17*3] on device
    // d_out_costs: [num_tracks, num_detections] on device
    void computeOKSDistanceDeviceAsync(
        const float* d_tracks,
        const float* d_detections,
        float* d_out_costs,
        int num_tracks,
        int num_detections,
        cudaStream_t stream = 0
    );

    // Compute IoU distance from device pointers - async
    // d_track_poses: [num_tracks, 17*3] on device
    // d_det_poses: [num_detections, 17*3] on device
    void computeIoUDistanceDeviceAsync(
        const float* d_track_poses,
        const float* d_det_poses,
        float* d_out_costs,
        int num_tracks,
        int num_detections,
        cudaStream_t stream = 0
    );

    // ========================================================================
    // Legacy Host-Side Operations (for compatibility)
    // ========================================================================

    // Compute OKS distance matrix between tracks and detections
    // Returns cost matrix where cost = 1 - OKS
    void computeOKSDistance(
        const PoseDetection* tracks,
        const PoseDetection* detections,
        float* out_costs,
        int num_tracks,
        int num_detections
    );

    // Compute IoU distance matrix (fallback for bbox-based matching)
    void computeIoUDistance(
        const PoseDetection* tracks,
        const PoseDetection* detections,
        float* out_costs,
        int num_tracks,
        int num_detections
    );

    // Compute combined distance: alpha * OKS + (1-alpha) * IoU
    void computeCombinedDistance(
        const PoseDetection* tracks,
        const PoseDetection* detections,
        float* out_costs,
        int num_tracks,
        int num_detections,
        float alpha = 0.7f
    );

    // ========================================================================
    // Device Memory Access
    // ========================================================================

    // Get internal device buffers
    float* getCostsDevice() { return d_costs_; }
    float* getTracksDevice() { return d_tracks_; }
    float* getDetectionsDevice() { return d_detections_; }
    float* getSigmasDevice() { return d_sigmas_; }
    float* getTrackBboxesDevice() { return d_track_bboxes_; }
    float* getDetBboxesDevice() { return d_det_bboxes_; }

    // Get limits
    int getMaxTracks() const { return max_tracks_; }
    int getMaxDetections() const { return max_detections_; }

    cudaStream_t getStream() const { return stream_; }

private:
    int max_tracks_;
    int max_detections_;

    // Device memory
    float* d_tracks_;      // [max_tracks, NUM_KEYPOINTS * 3]
    float* d_detections_;  // [max_detections, NUM_KEYPOINTS * 3]
    float* d_costs_;       // [max_tracks, max_detections]
    float* d_sigmas_;      // [NUM_KEYPOINTS] - OKS sigma values

    // Pre-allocated bbox buffers for IoU
    float* d_track_bboxes_;  // [max_tracks, 4]
    float* d_det_bboxes_;    // [max_detections, 4]

    // Pre-allocated host buffers (pinned memory)
    float* h_tracks_buffer_;      // [max_tracks, NUM_KEYPOINTS * 3]
    float* h_detections_buffer_;  // [max_detections, NUM_KEYPOINTS * 3]

    cudaStream_t stream_;
};

}  // namespace cuda
}  // namespace posebyte
