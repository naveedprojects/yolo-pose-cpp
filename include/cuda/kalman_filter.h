#pragma once

#include "types.h"
#include <cuda_runtime.h>
#include <vector>

namespace posebyte {
namespace cuda {

// GPU-native batched Kalman Filter for Pose Tracking
// All operations are batched and async - single sync point at end
// State layout per keypoint: [x, y, vx, vy, ax, ay, jx, jy] (8 dims)
// Total state per track: 17 keypoints * 8 = 136 dims
class KalmanFilterCUDA {
public:
    KalmanFilterCUDA(int max_tracks = 256);
    ~KalmanFilterCUDA();

    // ========================================================================
    // GPU-Native Batch Operations (async, no sync until explicit call)
    // ========================================================================

    // Predict all active tracks - async, no sync
    void predictAsync(int num_active_tracks, cudaStream_t stream = 0);

    // Batch update multiple tracks from device detection data
    // d_detections: [num_detections, 17, 3] on device
    // d_matches: [num_matches, 2] (track_slot, det_idx) on device
    void updateBatchAsync(
        const float* d_detections,    // Device: [num_dets, 17*3]
        const int* d_matches,          // Device: [num_matches, 2]
        int num_matches,
        cudaStream_t stream = 0
    );

    // Batch initialize multiple tracks from device detections
    // d_detections: [num_new, 17, 3] on device
    // d_track_slots: [num_new] track slot indices to initialize
    void initiateBatchAsync(
        const float* d_detections,    // Device: [num_new, 17*3]
        const int* d_track_slots,     // Device: [num_new]
        int num_new,
        cudaStream_t stream = 0
    );

    // Extract predicted poses to device buffer (for OKS computation)
    // d_out_poses: [num_tracks, 17, 3] output on device
    void extractPosesToDeviceAsync(
        float* d_out_poses,           // Device output
        const int* d_track_slots,     // Device: [num_tracks] - which slots to extract
        int num_tracks,
        cudaStream_t stream = 0
    );

    // Sync stream - call only when you need results
    void sync(cudaStream_t stream = 0);

    // ========================================================================
    // Legacy Host-Side Operations (for compatibility)
    // ========================================================================

    // Initialize filter for a new track from detection (host side)
    void initiate(int track_idx, const PoseDetection& detection);

    // Predict next state for all active tracks
    void predict(int num_tracks, float accel_memory = 0.9f, float jerk_memory = 0.9f);

    // Update tracks with matched detections (host side)
    void update(const PoseDetection* detections, const int* matches, int num_matches);

    // Get predicted keypoint positions for a track (D2H copy)
    void getPredictedPose(int track_idx, PoseDetection& out_pose) const;

    // Batch get all predicted poses (D2H copy)
    void getAllPredictedPoses(PoseDetection* out_poses, int num_tracks) const;

    // Get state mean/covariance (for debugging)
    void getState(int track_idx, float* mean, float* covariance) const;

    // Reset a track slot
    void resetTrack(int track_idx);

    // ========================================================================
    // Device Memory Access
    // ========================================================================

    // Get device pointers for GPU-native pipeline
    float* getMeansDevice() { return d_means_; }
    float* getCovariancesDevice() { return d_covariances_; }
    int getMaxTracks() const { return max_tracks_; }
    cudaStream_t getStream() const { return stream_; }

    // Set acceleration/jerk memory factors
    void setMotionParams(float accel_memory, float jerk_memory) {
        accel_memory_ = accel_memory;
        jerk_memory_ = jerk_memory;
    }

private:
    int max_tracks_;
    float accel_memory_ = 0.9f;
    float jerk_memory_ = 0.9f;

    // ========================================================================
    // Device Memory (persistent - never freed between frames)
    // ========================================================================

    // State (batch processing)
    float* d_means_;           // [max_tracks, TOTAL_STATE_DIM]
    float* d_covariances_;     // [max_tracks, TOTAL_STATE_DIM, TOTAL_STATE_DIM]

    // Transition matrix (shared across all tracks)
    float* d_transition_;      // [TOTAL_STATE_DIM, TOTAL_STATE_DIM]

    // Process/measurement noise
    float* d_process_noise_;   // [TOTAL_STATE_DIM, TOTAL_STATE_DIM]
    float* d_measurement_noise_; // [NUM_KEYPOINTS * 2, NUM_KEYPOINTS * 2]

    // Pre-allocated workspace (no per-frame alloc)
    float* d_workspace_;       // General purpose workspace
    float* d_det_buffer_;      // [max_tracks, 17*3] for detection data
    int* d_match_buffer_;      // [max_tracks, 2] for match pairs
    int* d_slot_buffer_;       // [max_tracks] for track slot indices

    // ========================================================================
    // Pinned Host Memory (for fast H2D/D2H)
    // ========================================================================
    float* h_det_pinned_;      // [max_tracks, 17*3]
    int* h_match_pinned_;      // [max_tracks, 2]
    int* h_slot_pinned_;       // [max_tracks]
    float* h_pose_pinned_;     // [max_tracks, 17*3] for output

    // Initialize matrices
    void initTransitionMatrix(float accel_memory, float jerk_memory);
    void initNoiseCovariances();

    cudaStream_t stream_;
};

}  // namespace cuda
}  // namespace posebyte
