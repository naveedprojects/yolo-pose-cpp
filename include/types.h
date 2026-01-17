#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

// Ensure fminf/fmaxf are available for host code
#ifndef __CUDA_ARCH__
using std::fminf;
using std::fmaxf;
#endif

namespace posebyte {

// COCO keypoint indices
enum CocoKeypoint {
    NOSE = 0,
    LEFT_EYE = 1,
    RIGHT_EYE = 2,
    LEFT_EAR = 3,
    RIGHT_EAR = 4,
    LEFT_SHOULDER = 5,
    RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7,
    RIGHT_ELBOW = 8,
    LEFT_WRIST = 9,
    RIGHT_WRIST = 10,
    LEFT_HIP = 11,
    RIGHT_HIP = 12,
    LEFT_KNEE = 13,
    RIGHT_KNEE = 14,
    LEFT_ANKLE = 15,
    RIGHT_ANKLE = 16,
    NUM_KEYPOINTS = 17
};

// COCO OKS falloff values (sigma * 2)
// These control per-keypoint tolerance in OKS calculation
constexpr float COCO_SIGMAS[NUM_KEYPOINTS] = {
    0.026f,  // nose
    0.025f,  // left_eye
    0.025f,  // right_eye
    0.035f,  // left_ear
    0.035f,  // right_ear
    0.079f,  // left_shoulder
    0.079f,  // right_shoulder
    0.072f,  // left_elbow
    0.072f,  // right_elbow
    0.062f,  // left_wrist
    0.062f,  // right_wrist
    0.107f,  // left_hip
    0.107f,  // right_hip
    0.087f,  // left_knee
    0.087f,  // right_knee
    0.089f,  // left_ankle
    0.089f   // right_ankle
};

// Single keypoint structure
struct Keypoint {
    float x;
    float y;
    float confidence;
};

// Pose detection result from YOLO-Pose
struct PoseDetection {
    float bbox[4];  // x1, y1, x2, y2
    float score;    // detection confidence
    Keypoint keypoints[NUM_KEYPOINTS];

    // Get pose bounding box area (for OKS normalization)
    __host__ __device__ float getPoseArea() const {
        float min_x = 1e9f, min_y = 1e9f;
        float max_x = -1e9f, max_y = -1e9f;
        int valid_count = 0;

        for (int i = 0; i < NUM_KEYPOINTS; i++) {
            if (keypoints[i].confidence > 0.0f) {
                min_x = fminf(min_x, keypoints[i].x);
                min_y = fminf(min_y, keypoints[i].y);
                max_x = fmaxf(max_x, keypoints[i].x);
                max_y = fmaxf(max_y, keypoints[i].y);
                valid_count++;
            }
        }

        if (valid_count < 2) return 0.0f;
        return (max_x - min_x) * (max_y - min_y);
    }

    // Get pose height (for Kalman noise scaling)
    __host__ __device__ float getPoseHeight() const {
        float min_y = 1e9f, max_y = -1e9f;

        for (int i = 0; i < NUM_KEYPOINTS; i++) {
            if (keypoints[i].confidence > 0.0f) {
                min_y = fminf(min_y, keypoints[i].y);
                max_y = fmaxf(max_y, keypoints[i].y);
            }
        }

        return max_y - min_y;
    }
};

// Track state enumeration
enum class TrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

// Kalman filter state dimensions
// For third-order model: position, velocity, acceleration, jerk
// Per keypoint: 2 coords (x, y) * 4 orders = 8 values
// Total: 17 keypoints * 8 = 136 dimensional state
constexpr int MOTION_ORDERS = 4;  // pos, vel, acc, jerk
constexpr int COORDS_PER_KP = 2;   // x, y
constexpr int STATE_DIM_PER_KP = MOTION_ORDERS * COORDS_PER_KP;  // 8
constexpr int TOTAL_STATE_DIM = NUM_KEYPOINTS * STATE_DIM_PER_KP;  // 136

// Kalman state for a single track
struct KalmanState {
    float mean[TOTAL_STATE_DIM];
    float covariance[TOTAL_STATE_DIM * TOTAL_STATE_DIM];

    // Initialize from detection
    void initFromDetection(const PoseDetection& det);
};

// Configuration for PoseBYTE tracker
struct TrackerConfig {
    // Detection thresholds
    float high_thresh = 0.6f;      // High confidence threshold
    float low_thresh = 0.1f;       // Low confidence threshold
    float new_track_thresh = 0.7f; // Threshold for new track creation

    // Track management
    int max_time_lost = 30;        // Frames before track removal
    int min_hits = 3;              // Hits before track confirmation

    // Matching thresholds
    float match_thresh = 0.8f;     // OKS threshold for matching
    float iou_thresh = 0.3f;       // IoU threshold (fallback)

    // Kalman filter parameters
    float accel_memory = 0.9f;     // Acceleration memory factor
    float jerk_memory = 0.9f;      // Jerk memory factor

    // NMS
    float nms_thresh = 0.65f;      // NMS IoU threshold
};

// Output track structure
struct Track {
    int id;
    TrackState state;
    PoseDetection pose;
    int age;           // Frames since creation
    int hits;          // Total successful matches
    int time_lost;     // Frames since last match
    float score;       // Average detection score
};

// Simple track output for GPU tracker
struct TrackOutput {
    int track_id;
    float score;
    float bbox[4];  // x1, y1, x2, y2
    Keypoint keypoints[NUM_KEYPOINTS];
};

}  // namespace posebyte
