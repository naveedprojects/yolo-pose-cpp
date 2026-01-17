#pragma once

#include "types.h"
#include "cuda/kalman_filter.h"
#include "cuda/oks_distance.h"
#include "cuda/hungarian.h"
#include "cuda/gpu_postprocess.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace posebyte {
namespace cuda {

// GPU-Native Tracker Configuration
struct GPUTrackerConfig {
    int max_tracks = 128;          // Maximum concurrent tracks
    int max_detections = 64;       // Maximum detections per frame
    float match_threshold = 0.5f;  // Cost threshold for matching (1 - OKS)
    float high_thresh = 0.30f;     // High confidence threshold for two-tier
    float low_thresh = 0.15f;      // Low confidence threshold
    float new_track_thresh = 0.30f; // Minimum conf for new tracks
    int max_age = 10;              // Frames before track goes to lost
    int min_hits = 3;              // Minimum hits before track is confirmed
    bool use_cuda_graph = false;   // Enable CUDA graph capture
};

// Per-stage timing statistics
struct TrackerTiming {
    long long predict_us = 0;
    long long gate_us = 0;
    long long high_assoc_us = 0;
    long long low_assoc_us = 0;
    long long lost_assoc_us = 0;
    long long update_us = 0;
    long long age_us = 0;
    long long new_track_us = 0;
    long long dedup_us = 0;
    long long total_us = 0;
    int frame_count = 0;
};

// GPU-Native Track State
struct GPUTrackState {
    int track_id;
    int state;           // 0=tentative, 1=confirmed, 2=lost
    int hits;
    int age;
    int last_frame;
};

// Production-grade GPU-Native Tracker
// Features:
// - Two-tier ByteTrack-style association (high/low confidence)
// - Spatial gating to reduce O(T*D) to O(T*k) where k << D
// - Lost track recovery with configurable window
// - Visibility-masked OKS with torso fallback
// - Kalman filtering for pose smoothing
// - Duplicate track suppression
// - Per-stage timing telemetry
class GPUTracker {
public:
    GPUTracker(const GPUTrackerConfig& config = GPUTrackerConfig());
    ~GPUTracker();

    // ========================================================================
    // Main Interface
    // ========================================================================

    // Update tracker with detections from GPU postprocess
    // d_det_poses: [num_dets, 17, 3] on device
    // d_det_scores: [num_dets] on device
    // Returns number of active tracks
    int update(
        const float* d_det_poses,
        const float* d_det_scores,
        int num_detections,
        int frame_id
    );

    // Get active tracks (copies to host for visualization)
    std::vector<TrackOutput> getActiveTracks();

    // Get number of active tracks
    int getNumActiveTracks() const { return num_active_tracks_; }

    // Print per-stage timing statistics
    void printTimingStats() const;

    // Get timing stats struct
    const TrackerTiming& getTiming() const { return timing_; }

    // ========================================================================
    // CUDA Graph Support (limited for complex pipeline)
    // ========================================================================

    void captureGraph();
    void executeGraph();
    bool isGraphCaptured() const { return graph_captured_; }

    // ========================================================================
    // Device Memory Access (for chaining)
    // ========================================================================

    float* getTrackPosesDevice() { return d_track_poses_; }
    float* getTrackScoresDevice() { return d_track_scores_; }
    int* getTrackStatesDevice() { return d_track_states_; }
    int* getTrackIdsDevice() { return d_track_ids_; }

    cudaStream_t getStream() const { return stream_; }

private:
    GPUTrackerConfig config_;

    // ========================================================================
    // Constants
    // ========================================================================

    static constexpr int LOST_WINDOW = 10;          // Frames to keep lost tracks
    static constexpr float GATE_THRESHOLD = 3.0f;   // Spatial gate threshold
    static constexpr float VISIBILITY_THRESHOLD = 0.2f;  // Keypoint visibility
    static constexpr float DEDUP_IOU_THRESHOLD = 0.7f;   // Duplicate suppression

    // ========================================================================
    // Device Memory (persistent)
    // ========================================================================

    // Track state
    float* d_track_poses_;        // [max_tracks, 17, 3]
    float* d_track_scores_;       // [max_tracks]
    int* d_track_states_;         // [max_tracks]
    int* d_track_ids_;            // [max_tracks]
    int* d_track_hits_;           // [max_tracks]
    int* d_track_ages_;           // [max_tracks]
    int* d_track_last_frame_;     // [max_tracks]
    int* d_track_active_;         // [max_tracks]

    // Detection buffer
    float* d_det_poses_;          // [max_detections, 17, 3]
    float* d_det_scores_;         // [max_detections]
    int* d_det_matched_;          // [max_detections]

    // Cost matrix and assignments
    float* d_cost_matrix_;        // [max_tracks, max_detections]
    int* d_row_assignments_;      // [max_tracks]
    int* d_col_assignments_;      // [max_detections]

    // Backup buffers for preserving prior tier assignments
    int* d_row_assign_backup_;    // [max_tracks]
    int* d_col_assign_backup_;    // [max_detections]

    // Counters
    int* d_num_active_tracks_;
    int* d_num_new_tracks_;

    // New track allocation
    int* d_slot_for_det_;         // [max_detections]
    int* d_next_slot_hint_;
    int* d_next_track_id_;

    // Velocity and prediction
    float* d_track_velocities_;   // [max_tracks, 17, 2]
    float* d_predicted_poses_;    // [max_tracks, 17, 3]

    // Spatial gating
    float* d_track_centers_;      // [max_tracks, 4] - cx, cy, w, h
    float* d_det_centers_;        // [max_detections, 4]
    int* d_gate_mask_;            // [max_tracks, max_detections]
    int* d_lost_gate_mask_;       // [max_tracks, max_detections] - separate for lost tier

    // Two-tier association
    int* d_high_conf_mask_;       // [max_detections]
    int* d_low_conf_mask_;        // [max_detections]

    // Duplicate suppression
    float* d_track_iou_matrix_;   // [max_tracks, max_tracks]

    // ========================================================================
    // Pinned Host Memory
    // ========================================================================

    float* h_track_poses_pinned_;
    float* h_track_scores_pinned_;
    GPUTrackState* h_track_states_pinned_;

    // ========================================================================
    // CUDA Components
    // ========================================================================

    std::unique_ptr<KalmanFilterCUDA> kalman_;
    std::unique_ptr<OKSDistanceCUDA> oks_distance_;
    std::unique_ptr<LinearAssignmentCUDA> assignment_;

    cudaStream_t stream_;
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    bool graph_captured_ = false;

    // ========================================================================
    // State
    // ========================================================================

    int num_active_tracks_ = 0;
    int next_track_id_ = 1;
    int current_frame_ = 0;
    int current_num_detections_ = 0;

    // Timing
    TrackerTiming timing_;

    // ========================================================================
    // Internal Operations
    // ========================================================================

    void predictTracks();
    void computeSpatialGating(int num_detections);
    void associateHighConfidence(int num_detections);
    void associateLowConfidence(int num_detections);
    void associateLostTracks(int num_detections);
    void updateMatchedTracks(int num_detections);
    void handleUnmatchedTracks();
    void createNewTracks(int num_detections);
    void removeDuplicateTracks();
};

}  // namespace cuda
}  // namespace posebyte
