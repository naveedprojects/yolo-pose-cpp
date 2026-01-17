#include "cuda/gpu_tracker.h"
#include <cstdio>
#include <algorithm>
#include <chrono>

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
// Constants
// ============================================================================

constexpr int NUM_KEYPOINTS = 17;
constexpr int TRACK_STATE_TENTATIVE = 0;
constexpr int TRACK_STATE_CONFIRMED = 1;
constexpr int TRACK_STATE_LOST = 2;

// Torso keypoints for fallback matching (more stable than limbs)
// COCO: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
// Note: These are defined inline in kernels that use them

// ============================================================================
// CUDA Kernels - Utility
// ============================================================================

// Kernel: Compact active track indices into a dense array
__global__ void kernelCompactActiveIndices(
    const int* track_active,
    const int* track_states,
    int* active_indices,        // Output: dense array of active track indices
    int* lost_indices,          // Output: dense array of lost track indices
    int* num_active,            // Output: count of active tracks
    int* num_lost,              // Output: count of lost tracks
    int max_tracks
) {
    __shared__ int s_active_count;
    __shared__ int s_lost_count;

    if (threadIdx.x == 0) {
        s_active_count = 0;
        s_lost_count = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_tracks) return;

    if (track_active[idx] == 1) {
        if (track_states[idx] == TRACK_STATE_LOST) {
            int pos = atomicAdd(&s_lost_count, 1);
            // Will be written in second pass
        } else {
            int pos = atomicAdd(&s_active_count, 1);
        }
    }
    __syncthreads();

    // Single thread writes final counts
    if (threadIdx.x == 0) {
        atomicAdd(num_active, s_active_count);
        atomicAdd(num_lost, s_lost_count);
    }
}

// Simpler version: just count and build index arrays on CPU after copy
__global__ void kernelBuildActiveIndex(
    const int* track_active,
    const int* track_states,
    int* active_mask,           // Output: 1 if active & not lost, 0 otherwise
    int* lost_mask,             // Output: 1 if lost, 0 otherwise
    int max_tracks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_tracks) return;

    active_mask[idx] = 0;
    lost_mask[idx] = 0;

    if (track_active[idx] == 1) {
        if (track_states[idx] == TRACK_STATE_LOST) {
            lost_mask[idx] = 1;
        } else {
            active_mask[idx] = 1;
        }
    }
}

// ============================================================================
// CUDA Kernels - Kalman Filter Integration
// ============================================================================

// Kernel: Kalman predict step - updates predicted poses using velocity model
__global__ void kernelKalmanPredict(
    float* predicted_poses,         // Output: [max_tracks, 17*3]
    const float* track_poses,       // Input: [max_tracks, 17*3]
    float* track_velocities,        // Input/Output: [max_tracks, 17*2]
    const int* track_active,
    const int* track_states,
    int max_tracks,
    float dt                        // Time delta (1.0 for frame-based)
) {
    int track_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (track_idx >= max_tracks) return;
    if (track_active[track_idx] == 0) return;

    // Copy and predict for all tracks (active and lost)
    for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
        int pose_offset = track_idx * NUM_KEYPOINTS * 3 + kp * 3;
        int vel_offset = track_idx * NUM_KEYPOINTS * 2 + kp * 2;

        float x = track_poses[pose_offset + 0];
        float y = track_poses[pose_offset + 1];
        float conf = track_poses[pose_offset + 2];

        float vx = track_velocities[vel_offset + 0];
        float vy = track_velocities[vel_offset + 1];

        // Predict position
        predicted_poses[pose_offset + 0] = x + vx * dt;
        predicted_poses[pose_offset + 1] = y + vy * dt;
        predicted_poses[pose_offset + 2] = conf;

        // Decay velocity slightly for lost tracks (they're less certain)
        if (track_states[track_idx] == TRACK_STATE_LOST) {
            track_velocities[vel_offset + 0] *= 0.95f;
            track_velocities[vel_offset + 1] *= 0.95f;
        }
    }
}

// Kernel: Kalman update step - updates track state with matched detection
__global__ void kernelKalmanUpdate(
    float* track_poses,             // Output: updated poses
    float* track_velocities,        // Output: updated velocities
    const float* det_poses,         // Input: detection poses
    const int* row_assignments,     // Input: track -> detection mapping
    const int* track_active,
    int max_tracks,
    float process_noise,            // Kalman process noise
    float measurement_noise         // Kalman measurement noise
) {
    int track_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (track_idx >= max_tracks) return;
    if (track_active[track_idx] == 0) return;

    int det_idx = row_assignments[track_idx];
    if (det_idx < 0) return;  // No match

    // Kalman gain (simplified constant gain for speed)
    float K = measurement_noise / (measurement_noise + process_noise);
    float alpha = 0.3f;  // Velocity smoothing

    for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
        int track_offset = track_idx * NUM_KEYPOINTS * 3 + kp * 3;
        int det_offset = det_idx * NUM_KEYPOINTS * 3 + kp * 3;
        int vel_offset = track_idx * NUM_KEYPOINTS * 2 + kp * 2;

        float old_x = track_poses[track_offset + 0];
        float old_y = track_poses[track_offset + 1];

        float det_x = det_poses[det_offset + 0];
        float det_y = det_poses[det_offset + 1];
        float det_conf = det_poses[det_offset + 2];

        // Update position with Kalman gain
        float new_x = old_x + K * (det_x - old_x);
        float new_y = old_y + K * (det_y - old_y);

        // Update velocity (exponential smoothing)
        float dx = det_x - old_x;
        float dy = det_y - old_y;
        track_velocities[vel_offset + 0] = alpha * dx + (1 - alpha) * track_velocities[vel_offset + 0];
        track_velocities[vel_offset + 1] = alpha * dy + (1 - alpha) * track_velocities[vel_offset + 1];

        // Store updated pose
        track_poses[track_offset + 0] = new_x;
        track_poses[track_offset + 1] = new_y;
        track_poses[track_offset + 2] = det_conf;
    }
}

// ============================================================================
// CUDA Kernels - Spatial Gating
// ============================================================================

// Kernel: Compute bounding box center and size for spatial gating
__global__ void kernelComputeBboxCenters(
    const float* poses,             // [num_poses, 17*3]
    float* centers,                 // Output: [num_poses, 4] - cx, cy, w, h
    int num_poses
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_poses) return;

    int offset = idx * NUM_KEYPOINTS * 3;

    float min_x = 1e9f, min_y = 1e9f;
    float max_x = -1e9f, max_y = -1e9f;
    int valid = 0;

    for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
        float conf = poses[offset + kp * 3 + 2];
        if (conf > 0.1f) {
            float x = poses[offset + kp * 3 + 0];
            float y = poses[offset + kp * 3 + 1];
            min_x = fminf(min_x, x);
            min_y = fminf(min_y, y);
            max_x = fmaxf(max_x, x);
            max_y = fmaxf(max_y, y);
            valid++;
        }
    }

    if (valid < 2) {
        centers[idx * 4 + 0] = 0;
        centers[idx * 4 + 1] = 0;
        centers[idx * 4 + 2] = 0;
        centers[idx * 4 + 3] = 0;
        return;
    }

    float w = max_x - min_x;
    float h = max_y - min_y;
    centers[idx * 4 + 0] = (min_x + max_x) * 0.5f;  // cx
    centers[idx * 4 + 1] = (min_y + max_y) * 0.5f;  // cy
    centers[idx * 4 + 2] = w;
    centers[idx * 4 + 3] = h;
}

// Kernel: Build spatial gate mask with adaptive threshold based on velocity
// Faster moving tracks get wider gates to account for motion between frames
__global__ void kernelSpatialGate(
    const float* track_centers,     // [max_tracks, 4]
    const float* det_centers,       // [num_dets, 4]
    const float* track_velocities,  // [max_tracks, 17*2] - per-keypoint velocities
    const int* track_active,
    const int* track_states,
    int* gate_mask,                 // Output: [max_tracks, num_dets] - 1 if should compute
    int max_tracks,
    int num_dets,
    float gate_threshold            // Base max center distance as ratio of bbox size
) {
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= max_tracks || det_idx >= num_dets) return;

    int mask_idx = track_idx * num_dets + det_idx;

    // Skip inactive tracks
    if (track_active[track_idx] == 0) {
        gate_mask[mask_idx] = 0;
        return;
    }

    float t_cx = track_centers[track_idx * 4 + 0];
    float t_cy = track_centers[track_idx * 4 + 1];
    float t_w = track_centers[track_idx * 4 + 2];
    float t_h = track_centers[track_idx * 4 + 3];

    float d_cx = det_centers[det_idx * 4 + 0];
    float d_cy = det_centers[det_idx * 4 + 1];
    float d_w = det_centers[det_idx * 4 + 2];
    float d_h = det_centers[det_idx * 4 + 3];

    // If centers are invalid (couldn't be computed from keypoints),
    // ALLOW matching to proceed - let OKS decide based on actual keypoints
    if (t_w < 1.0f || t_h < 1.0f || d_w < 1.0f || d_h < 1.0f) {
        gate_mask[mask_idx] = 1;  // Allow matching
        return;
    }

    // Compute center distance
    float dx = t_cx - d_cx;
    float dy = t_cy - d_cy;
    float dist = sqrtf(dx * dx + dy * dy);

    // Compute average velocity magnitude from torso keypoints (5,6,11,12)
    // These are the most stable for velocity estimation
    const int torso_kps[] = {5, 6, 11, 12};
    float avg_velocity = 0.0f;
    for (int i = 0; i < 4; i++) {
        int kp = torso_kps[i];
        int vel_offset = track_idx * NUM_KEYPOINTS * 2 + kp * 2;
        float vx = track_velocities[vel_offset + 0];
        float vy = track_velocities[vel_offset + 1];
        avg_velocity += sqrtf(vx * vx + vy * vy);
    }
    avg_velocity *= 0.25f;

    // Gate based on average size
    float avg_size = (t_w + t_h + d_w + d_h) * 0.25f;
    float ratio = dist / (avg_size + 1e-6f);

    // Adaptive gate threshold based on velocity
    // - Base threshold for slow tracks
    // - Increase by velocity/size ratio for faster tracks
    // - Cap at 3x base threshold to avoid too wide gates
    float velocity_factor = 1.0f + fminf(avg_velocity / (avg_size + 1e-6f), 2.0f);
    float threshold = gate_threshold * velocity_factor;

    // For lost tracks, use even wider gate (they may have moved more)
    if (track_states[track_idx] == TRACK_STATE_LOST) {
        threshold *= 2.0f;
    }

    gate_mask[mask_idx] = (ratio < threshold) ? 1 : 0;
}

// ============================================================================
// CUDA Kernels - Visibility-Masked OKS
// ============================================================================

// COCO keypoint sigmas
__constant__ float d_sigmas[17] = {
    0.026f, 0.025f, 0.025f, 0.035f, 0.035f,  // nose, eyes, ears
    0.079f, 0.079f, 0.072f, 0.072f,          // shoulders, elbows
    0.062f, 0.062f, 0.107f, 0.107f,          // wrists, hips
    0.087f, 0.087f, 0.089f, 0.089f           // knees, ankles
};

// Kernel: Compute visibility-masked OKS with spatial gating
// NOTE: Preserves locked pairs (gate_mask == 0 means already matched in previous tier)
__global__ void kernelOKSWithGating(
    const float* track_poses,       // [max_tracks, 17*3] - predicted poses
    const float* det_poses,         // [num_dets, 17*3]
    const int* gate_mask,           // [max_tracks, num_dets]
    const int* track_active,
    float* cost_matrix,             // Output: [max_tracks, num_dets]
    int max_tracks,
    int num_dets,
    float visibility_threshold      // Min confidence to consider keypoint visible
) {
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= max_tracks || det_idx >= num_dets) return;

    int cost_idx = track_idx * num_dets + det_idx;

    // Skip if not gated or inactive - PRESERVE existing cost for locked pairs
    if (track_active[track_idx] == 0) {
        cost_matrix[cost_idx] = 1.0f;  // Inactive track
        return;
    }
    if (gate_mask[cost_idx] == 0) {
        // Don't overwrite - this pair may be locked from previous tier
        return;
    }

    int track_offset = track_idx * NUM_KEYPOINTS * 3;
    int det_offset = det_idx * NUM_KEYPOINTS * 3;

    // Compute scale from detection (use both for stability)
    float det_min_x = 1e9f, det_min_y = 1e9f;
    float det_max_x = -1e9f, det_max_y = -1e9f;
    float track_min_x = 1e9f, track_min_y = 1e9f;
    float track_max_x = -1e9f, track_max_y = -1e9f;

    for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
        float d_conf = det_poses[det_offset + kp * 3 + 2];
        float t_conf = track_poses[track_offset + kp * 3 + 2];

        if (d_conf > 0.1f) {
            float x = det_poses[det_offset + kp * 3 + 0];
            float y = det_poses[det_offset + kp * 3 + 1];
            det_min_x = fminf(det_min_x, x);
            det_min_y = fminf(det_min_y, y);
            det_max_x = fmaxf(det_max_x, x);
            det_max_y = fmaxf(det_max_y, y);
        }
        if (t_conf > 0.1f) {
            float x = track_poses[track_offset + kp * 3 + 0];
            float y = track_poses[track_offset + kp * 3 + 1];
            track_min_x = fminf(track_min_x, x);
            track_min_y = fminf(track_min_y, y);
            track_max_x = fmaxf(track_max_x, x);
            track_max_y = fmaxf(track_max_y, y);
        }
    }

    float det_area = (det_max_x - det_min_x) * (det_max_y - det_min_y);
    float track_area = (track_max_x - track_min_x) * (track_max_y - track_min_y);
    float scale_sq = fmaxf((det_area + track_area) * 0.5f, 1000.0f);  // Min scale

    // Compute visibility-masked OKS
    float oks_sum = 0.0f;
    int oks_count = 0;

    for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
        float d_conf = det_poses[det_offset + kp * 3 + 2];
        float t_conf = track_poses[track_offset + kp * 3 + 2];

        // Only use keypoints visible in BOTH (visibility masking)
        if (d_conf > visibility_threshold && t_conf > visibility_threshold) {
            float d_x = det_poses[det_offset + kp * 3 + 0];
            float d_y = det_poses[det_offset + kp * 3 + 1];
            float t_x = track_poses[track_offset + kp * 3 + 0];
            float t_y = track_poses[track_offset + kp * 3 + 1];

            float dx = d_x - t_x;
            float dy = d_y - t_y;
            float dist_sq = dx * dx + dy * dy;

            float sigma = d_sigmas[kp] * 2.0f;  // Relaxed sigma
            float sigma_sq = sigma * sigma;

            float oks_kp = expf(-dist_sq / (2.0f * scale_sq * sigma_sq));
            oks_sum += oks_kp;
            oks_count++;
        }
    }

    float oks = (oks_count >= 3) ? (oks_sum / oks_count) : 0.0f;
    cost_matrix[cost_idx] = 1.0f - oks;
}

// Kernel: Compute torso-only OKS for low-confidence fallback
// NOTE: Preserves locked pairs from previous tiers
__global__ void kernelTorsoOKS(
    const float* track_poses,
    const float* det_poses,
    const int* gate_mask,
    const int* track_active,
    float* cost_matrix,
    int max_tracks,
    int num_dets
) {
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= max_tracks || det_idx >= num_dets) return;

    int cost_idx = track_idx * num_dets + det_idx;

    // Skip inactive tracks
    if (track_active[track_idx] == 0) {
        cost_matrix[cost_idx] = 1.0f;
        return;
    }
    // Preserve locked pairs from previous tiers
    if (gate_mask[cost_idx] == 0) {
        return;  // Don't overwrite
    }

    int track_offset = track_idx * NUM_KEYPOINTS * 3;
    int det_offset = det_idx * NUM_KEYPOINTS * 3;

    // Compute scale
    float scale_sq = 10000.0f;  // Default scale for torso

    // Only use torso keypoints (5, 6, 11, 12)
    float oks_sum = 0.0f;
    int oks_count = 0;

    const int torso_kps[] = {5, 6, 11, 12};
    for (int i = 0; i < 4; i++) {
        int kp = torso_kps[i];
        float d_conf = det_poses[det_offset + kp * 3 + 2];
        float t_conf = track_poses[track_offset + kp * 3 + 2];

        if (d_conf > 0.1f && t_conf > 0.1f) {
            float d_x = det_poses[det_offset + kp * 3 + 0];
            float d_y = det_poses[det_offset + kp * 3 + 1];
            float t_x = track_poses[track_offset + kp * 3 + 0];
            float t_y = track_poses[track_offset + kp * 3 + 1];

            float dx = d_x - t_x;
            float dy = d_y - t_y;
            float dist_sq = dx * dx + dy * dy;

            float sigma = d_sigmas[kp] * 3.0f;  // Very relaxed for torso
            float oks_kp = expf(-dist_sq / (2.0f * scale_sq * sigma * sigma));
            oks_sum += oks_kp;
            oks_count++;
        }
    }

    float oks = (oks_count >= 2) ? (oks_sum / oks_count) : 0.0f;
    cost_matrix[cost_idx] = 1.0f - oks;
}

// ============================================================================
// CUDA Kernels - Match Locking and State Filtering
// ============================================================================

// Kernel: Mask out tracks by state (for tier-specific association)
// exclude_state: tracks with this state will have gate_mask set to 0
__global__ void kernelMaskTracksByState(
    int* gate_mask,                 // [max_tracks, num_dets] - modified in place
    const int* track_states,
    const int* track_active,
    int max_tracks,
    int num_dets,
    int exclude_state              // State to exclude (e.g., TRACK_STATE_LOST)
) {
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= max_tracks || det_idx >= num_dets) return;

    // If this track has the excluded state, mask it out
    if (track_active[track_idx] == 1 && track_states[track_idx] == exclude_state) {
        gate_mask[track_idx * num_dets + det_idx] = 0;
    }
}

// Kernel: Mask to ONLY include tracks with specific state
// include_state: only tracks with this state will have gate_mask = 1
__global__ void kernelMaskOnlyState(
    int* gate_mask,                 // [max_tracks, num_dets] - modified in place
    const int* track_states,
    const int* track_active,
    int max_tracks,
    int num_dets,
    int include_state              // Only include this state
) {
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= max_tracks || det_idx >= num_dets) return;

    // If this track doesn't have the included state, mask it out
    if (track_active[track_idx] == 0 || track_states[track_idx] != include_state) {
        gate_mask[track_idx * num_dets + det_idx] = 0;
    }
}

// Kernel: Lock matched pairs by setting their costs to infinity
// This prevents later tiers from reassigning already-matched pairs
__global__ void kernelLockMatchedPairs(
    float* cost_matrix,             // [max_tracks, num_dets] - modified in place
    int* gate_mask,                 // [max_tracks, num_dets] - modified in place
    const int* row_assignments,     // [max_tracks] - current track->det assignments
    const int* col_assignments,     // [num_dets] - current det->track assignments
    const int* track_active,
    int max_tracks,
    int num_dets
) {
    int track_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (track_idx >= max_tracks || det_idx >= num_dets) return;

    int idx = track_idx * num_dets + det_idx;

    // If this track is already matched, lock entire row
    if (row_assignments[track_idx] >= 0) {
        cost_matrix[idx] = 1e9f;
        gate_mask[idx] = 0;
    }

    // If this detection is already matched, lock entire column
    if (col_assignments[det_idx] >= 0) {
        cost_matrix[idx] = 1e9f;
        gate_mask[idx] = 0;
    }
}

// ============================================================================
// CUDA Kernels - Assignment Preservation
// ============================================================================

// Kernel: Merge assignment results, preserving prior matches
// If prior_assign[i] >= 0, keep it; otherwise use new_assign[i]
__global__ void kernelMergeAssignments(
    const int* prior_assign,
    int* merged_assign,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Preserve prior match, only accept new if prior was unmatched
    if (prior_assign[idx] >= 0) {
        merged_assign[idx] = prior_assign[idx];
    }
    // else: keep new_assign (already in merged_assign)
}

// ============================================================================
// CUDA Kernels - Two-Tier Association
// ============================================================================

// Kernel: Split detections into high and low confidence
__global__ void kernelSplitDetections(
    const float* det_scores,
    int* high_conf_mask,            // Output: 1 if high confidence
    int* low_conf_mask,             // Output: 1 if low confidence
    int num_dets,
    float high_threshold,
    float low_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dets) return;

    float score = det_scores[idx];
    high_conf_mask[idx] = (score >= high_threshold) ? 1 : 0;
    low_conf_mask[idx] = (score >= low_threshold && score < high_threshold) ? 1 : 0;
}

// Kernel: Apply assignment results and update track states
__global__ void kernelUpdateMatchedTracks(
    float* track_poses,
    float* track_scores,
    int* track_hits,
    int* track_ages,
    int* track_states,
    int* track_last_frame,
    int* track_active,
    const float* det_poses,
    const float* det_scores,
    const int* row_assignments,
    int max_tracks,
    int frame_id,
    int min_hits
) {
    int track_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (track_idx >= max_tracks) return;
    if (track_active[track_idx] == 0) return;

    int det_idx = row_assignments[track_idx];
    if (det_idx < 0) return;  // No match

    // Update track metadata
    track_scores[track_idx] = det_scores[det_idx];
    track_hits[track_idx]++;
    track_ages[track_idx] = 0;
    track_last_frame[track_idx] = frame_id;

    // State transitions
    int state = track_states[track_idx];
    if (state == TRACK_STATE_TENTATIVE && track_hits[track_idx] >= min_hits) {
        track_states[track_idx] = TRACK_STATE_CONFIRMED;
    } else if (state == TRACK_STATE_LOST) {
        // Reactivate lost track
        track_states[track_idx] = TRACK_STATE_CONFIRMED;
    }
}

// Kernel: Handle unmatched tracks - age them or mark as lost
__global__ void kernelAgeUnmatchedTracks(
    int* track_ages,
    int* track_states,
    int* track_active,
    const int* row_assignments,
    int max_tracks,
    int max_age,
    int lost_window              // Frames to keep in lost state before removal
) {
    int track_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (track_idx >= max_tracks) return;
    if (track_active[track_idx] == 0) return;

    int det_idx = row_assignments[track_idx];
    if (det_idx >= 0) return;  // Was matched

    // Age the track
    track_ages[track_idx]++;
    int age = track_ages[track_idx];
    int state = track_states[track_idx];

    if (state == TRACK_STATE_TENTATIVE) {
        // Tentative tracks are removed immediately if not matched
        if (age > 2) {
            track_active[track_idx] = 0;
        }
    } else if (state == TRACK_STATE_CONFIRMED) {
        // Confirmed tracks go to lost state
        if (age > max_age) {
            track_states[track_idx] = TRACK_STATE_LOST;
        }
    } else if (state == TRACK_STATE_LOST) {
        // Lost tracks are removed after lost_window
        if (age > max_age + lost_window) {
            track_active[track_idx] = 0;
        }
    }
}

// ============================================================================
// CUDA Kernels - New Track Creation
// ============================================================================

// Kernel: Allocate slots for new tracks using atomic operations
__global__ void kernelAllocateNewTrackSlots(
    int* track_active,
    const int* col_assignments,
    const float* det_scores,
    int* slot_for_det,
    int* next_slot_hint,
    int max_tracks,
    int num_dets,
    float conf_threshold
) {
    int det_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (det_idx >= num_dets) return;

    slot_for_det[det_idx] = -1;

    // Skip if already matched or low confidence
    if (col_assignments[det_idx] >= 0) return;
    if (det_scores[det_idx] < conf_threshold) return;

    // Find free slot
    int start = atomicAdd(next_slot_hint, 1) % max_tracks;
    for (int i = 0; i < max_tracks; i++) {
        int slot = (start + i) % max_tracks;
        int old = atomicCAS(&track_active[slot], 0, 1);
        if (old == 0) {
            slot_for_det[det_idx] = slot;
            return;
        }
    }
}

// Kernel: Initialize new tracks
__global__ void kernelInitNewTracks(
    float* track_poses,
    float* track_scores,
    int* track_ids,
    int* track_hits,
    int* track_ages,
    int* track_states,
    int* track_last_frame,
    float* track_velocities,
    const float* det_poses,
    const float* det_scores,
    int* col_assignments,
    const int* slot_for_det,
    int* next_track_id,
    int num_dets,
    int frame_id,
    float conf_threshold
) {
    int det_idx = blockIdx.x;
    int kp_idx = threadIdx.x;

    if (det_idx >= num_dets) return;
    if (col_assignments[det_idx] >= 0) return;
    if (det_scores[det_idx] < conf_threshold) return;

    int slot = slot_for_det[det_idx];
    if (slot < 0) return;

    __shared__ int new_id;
    if (kp_idx == 0) {
        new_id = atomicAdd(next_track_id, 1);
        track_ids[slot] = new_id;
        track_scores[slot] = det_scores[det_idx];
        track_hits[slot] = 1;
        track_ages[slot] = 0;
        track_states[slot] = TRACK_STATE_TENTATIVE;
        track_last_frame[slot] = frame_id;
        col_assignments[det_idx] = slot;  // Link detection to new track
    }
    __syncthreads();

    if (kp_idx < NUM_KEYPOINTS) {
        int track_offset = slot * NUM_KEYPOINTS * 3 + kp_idx * 3;
        int det_offset = det_idx * NUM_KEYPOINTS * 3 + kp_idx * 3;
        int vel_offset = slot * NUM_KEYPOINTS * 2 + kp_idx * 2;

        track_poses[track_offset + 0] = det_poses[det_offset + 0];
        track_poses[track_offset + 1] = det_poses[det_offset + 1];
        track_poses[track_offset + 2] = det_poses[det_offset + 2];

        track_velocities[vel_offset + 0] = 0.0f;
        track_velocities[vel_offset + 1] = 0.0f;
    }
}

// ============================================================================
// CUDA Kernels - Duplicate Suppression
// ============================================================================

// Kernel: Compute IoU between all track pairs for duplicate detection
// NOTE: Only computes IoU for CONFIRMED tracks (not LOST or TENTATIVE with few hits)
__global__ void kernelTrackIoU(
    const float* track_centers,     // [max_tracks, 4] - cx, cy, w, h
    const int* track_active,
    const int* track_states,
    const int* track_hits,
    float* iou_matrix,              // Output: [max_tracks, max_tracks]
    int max_tracks,
    int min_hits_for_dedup          // Minimum hits to consider for dedup
) {
    int t1 = blockIdx.y * blockDim.y + threadIdx.y;
    int t2 = blockIdx.x * blockDim.x + threadIdx.x;

    if (t1 >= max_tracks || t2 >= max_tracks || t1 >= t2) {
        if (t1 < max_tracks && t2 < max_tracks) {
            iou_matrix[t1 * max_tracks + t2] = 0.0f;
        }
        return;
    }

    // Skip inactive tracks
    if (track_active[t1] == 0 || track_active[t2] == 0) {
        iou_matrix[t1 * max_tracks + t2] = 0.0f;
        return;
    }

    // Skip LOST tracks - they should not be deduplicated
    if (track_states[t1] == TRACK_STATE_LOST || track_states[t2] == TRACK_STATE_LOST) {
        iou_matrix[t1 * max_tracks + t2] = 0.0f;
        return;
    }

    // Skip tracks with too few hits (unstable for dedup)
    if (track_hits[t1] < min_hits_for_dedup || track_hits[t2] < min_hits_for_dedup) {
        iou_matrix[t1 * max_tracks + t2] = 0.0f;
        return;
    }

    // Convert center format to corner format
    float cx1 = track_centers[t1 * 4 + 0];
    float cy1 = track_centers[t1 * 4 + 1];
    float w1 = track_centers[t1 * 4 + 2];
    float h1 = track_centers[t1 * 4 + 3];

    float cx2 = track_centers[t2 * 4 + 0];
    float cy2 = track_centers[t2 * 4 + 1];
    float w2 = track_centers[t2 * 4 + 2];
    float h2 = track_centers[t2 * 4 + 3];

    float x1_min = cx1 - w1 * 0.5f, x1_max = cx1 + w1 * 0.5f;
    float y1_min = cy1 - h1 * 0.5f, y1_max = cy1 + h1 * 0.5f;
    float x2_min = cx2 - w2 * 0.5f, x2_max = cx2 + w2 * 0.5f;
    float y2_min = cy2 - h2 * 0.5f, y2_max = cy2 + h2 * 0.5f;

    float inter_x1 = fmaxf(x1_min, x2_min);
    float inter_y1 = fmaxf(y1_min, y2_min);
    float inter_x2 = fminf(x1_max, x2_max);
    float inter_y2 = fminf(y1_max, y2_max);

    float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area1 = w1 * h1;
    float area2 = w2 * h2;
    float union_area = area1 + area2 - inter_area;

    float iou = (union_area > 0) ? (inter_area / union_area) : 0.0f;
    iou_matrix[t1 * max_tracks + t2] = iou;
    iou_matrix[t2 * max_tracks + t1] = iou;  // Symmetric
}

// Kernel: Remove duplicate tracks (keep one with more hits)
// NOTE: Only removes CONFIRMED tracks - LOST tracks are preserved for reactivation
__global__ void kernelRemoveDuplicates(
    const float* iou_matrix,
    int* track_active,
    const int* track_states,
    const int* track_hits,
    const int* track_ids,
    int max_tracks,
    float iou_threshold
) {
    int t1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (t1 >= max_tracks) return;
    if (track_active[t1] == 0) return;

    // Never remove LOST tracks in dedup - they need to stay for reactivation
    if (track_states[t1] == TRACK_STATE_LOST) return;

    for (int t2 = t1 + 1; t2 < max_tracks; t2++) {
        if (track_active[t2] == 0) continue;

        // Skip LOST tracks as duplicate candidates
        if (track_states[t2] == TRACK_STATE_LOST) continue;

        float iou = iou_matrix[t1 * max_tracks + t2];
        if (iou > iou_threshold) {
            // Remove track with fewer hits (or higher ID if tied)
            if (track_hits[t1] < track_hits[t2] ||
                (track_hits[t1] == track_hits[t2] && track_ids[t1] > track_ids[t2])) {
                track_active[t1] = 0;
                return;
            } else {
                track_active[t2] = 0;
            }
        }
    }
}

// ============================================================================
// CUDA Kernels - Count
// ============================================================================

__global__ void kernelCountActive(
    const int* track_active,
    int* count,
    int max_tracks
) {
    __shared__ int block_count;
    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < max_tracks && track_active[idx] == 1) {
        atomicAdd(&block_count, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}

// ============================================================================
// GPUTracker Implementation
// ============================================================================

GPUTracker::GPUTracker(const GPUTrackerConfig& config) : config_(config) {
    int max_t = config.max_tracks;
    int max_d = config.max_detections;

    // Allocate device memory for tracks
    CUDA_CHECK(cudaMalloc(&d_track_poses_, max_t * 17 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_track_scores_, max_t * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_track_states_, max_t * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_track_ids_, max_t * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_track_hits_, max_t * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_track_ages_, max_t * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_track_last_frame_, max_t * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_track_active_, max_t * sizeof(int)));

    // Detection buffers
    CUDA_CHECK(cudaMalloc(&d_det_poses_, max_d * 17 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_det_scores_, max_d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_det_matched_, max_d * sizeof(int)));

    // Cost matrix and assignments
    CUDA_CHECK(cudaMalloc(&d_cost_matrix_, max_t * max_d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_assignments_, max_t * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_assignments_, max_d * sizeof(int)));

    // Backup buffers for preserving prior tier assignments
    CUDA_CHECK(cudaMalloc(&d_row_assign_backup_, max_t * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_assign_backup_, max_d * sizeof(int)));

    // Counters and allocation
    CUDA_CHECK(cudaMalloc(&d_num_active_tracks_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_new_tracks_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_slot_for_det_, max_d * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_slot_hint_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_track_id_, sizeof(int)));

    // Velocity and prediction
    CUDA_CHECK(cudaMalloc(&d_track_velocities_, max_t * 17 * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predicted_poses_, max_t * 17 * 3 * sizeof(float)));

    // Spatial gating
    CUDA_CHECK(cudaMalloc(&d_track_centers_, max_t * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_det_centers_, max_d * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate_mask_, max_t * max_d * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lost_gate_mask_, max_t * max_d * sizeof(int)));

    // Confidence masks for two-tier
    CUDA_CHECK(cudaMalloc(&d_high_conf_mask_, max_d * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_low_conf_mask_, max_d * sizeof(int)));

    // Duplicate suppression
    CUDA_CHECK(cudaMalloc(&d_track_iou_matrix_, max_t * max_t * sizeof(float)));

    // Pinned host memory
    CUDA_CHECK(cudaMallocHost(&h_track_poses_pinned_, max_t * 17 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_track_scores_pinned_, max_t * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_track_states_pinned_, max_t * sizeof(GPUTrackState)));

    // Initialize
    CUDA_CHECK(cudaMemset(d_track_active_, 0, max_t * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_track_states_, 0, max_t * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_track_velocities_, 0, max_t * 17 * 2 * sizeof(float)));

    int one = 1;
    CUDA_CHECK(cudaMemcpy(d_next_track_id_, &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_next_slot_hint_, 0, sizeof(int)));

    // Create components
    kalman_ = std::make_unique<KalmanFilterCUDA>(max_t);
    oks_distance_ = std::make_unique<OKSDistanceCUDA>(max_t, max_d);
    assignment_ = std::make_unique<LinearAssignmentCUDA>(std::max(max_t, max_d));

    CUDA_CHECK(cudaStreamCreate(&stream_));

    // Initialize timing
    memset(&timing_, 0, sizeof(timing_));

    printf("[GPUTracker] Production tracker initialized:\n");
    printf("  - max_tracks=%d, max_dets=%d\n", max_t, max_d);
    printf("  - match_threshold=%.2f, high_thresh=%.2f, low_thresh=%.2f\n",
           config.match_threshold, config.high_thresh, config.low_thresh);
    printf("  - min_hits=%d, max_age=%d\n", config.min_hits, config.max_age);
    printf("  - Two-tier matching: ENABLED\n");
    printf("  - Spatial gating: ENABLED\n");
    printf("  - Lost track recovery: ENABLED (window=%d frames)\n", LOST_WINDOW);
    printf("  - Duplicate suppression: ENABLED\n");
}

GPUTracker::~GPUTracker() {
    // Free device memory
    cudaFree(d_track_poses_);
    cudaFree(d_track_scores_);
    cudaFree(d_track_states_);
    cudaFree(d_track_ids_);
    cudaFree(d_track_hits_);
    cudaFree(d_track_ages_);
    cudaFree(d_track_last_frame_);
    cudaFree(d_track_active_);
    cudaFree(d_det_poses_);
    cudaFree(d_det_scores_);
    cudaFree(d_det_matched_);
    cudaFree(d_cost_matrix_);
    cudaFree(d_row_assignments_);
    cudaFree(d_col_assignments_);
    cudaFree(d_row_assign_backup_);
    cudaFree(d_col_assign_backup_);
    cudaFree(d_num_active_tracks_);
    cudaFree(d_num_new_tracks_);
    cudaFree(d_slot_for_det_);
    cudaFree(d_next_slot_hint_);
    cudaFree(d_next_track_id_);
    cudaFree(d_track_velocities_);
    cudaFree(d_predicted_poses_);
    cudaFree(d_track_centers_);
    cudaFree(d_det_centers_);
    cudaFree(d_gate_mask_);
    cudaFree(d_lost_gate_mask_);
    cudaFree(d_high_conf_mask_);
    cudaFree(d_low_conf_mask_);
    cudaFree(d_track_iou_matrix_);

    cudaFreeHost(h_track_poses_pinned_);
    cudaFreeHost(h_track_scores_pinned_);
    cudaFreeHost(h_track_states_pinned_);

    if (graph_captured_) {
        cudaGraphExecDestroy(graph_exec_);
        cudaGraphDestroy(graph_);
    }

    cudaStreamDestroy(stream_);
}

int GPUTracker::update(
    const float* d_det_poses,
    const float* d_det_scores,
    int num_detections,
    int frame_id
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    current_frame_ = frame_id;
    current_num_detections_ = std::min(num_detections, config_.max_detections);

    if (current_num_detections_ > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_det_poses_, d_det_poses,
                                   current_num_detections_ * 17 * 3 * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_det_scores_, d_det_scores,
                                   current_num_detections_ * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));
    }

    // Reset assignments
    CUDA_CHECK(cudaMemsetAsync(d_row_assignments_, 0xFF, config_.max_tracks * sizeof(int), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_col_assignments_, 0xFF, current_num_detections_ * sizeof(int), stream_));

    // Count active tracks at start of frame (critical for gating/matching)
    CUDA_CHECK(cudaMemsetAsync(d_num_active_tracks_, 0, sizeof(int), stream_));
    kernelCountActive<<<(config_.max_tracks + 255) / 256, 256, 0, stream_>>>(
        d_track_active_, d_num_active_tracks_, config_.max_tracks
    );
    CUDA_CHECK(cudaMemcpyAsync(&num_active_tracks_, d_num_active_tracks_,
                               sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto t_copy = std::chrono::high_resolution_clock::now();

    // Stage 1: Predict track positions
    predictTracks();
    auto t_predict = std::chrono::high_resolution_clock::now();

    // Stage 2: Compute spatial gating
    computeSpatialGating(current_num_detections_);
    auto t_gate = std::chrono::high_resolution_clock::now();

    // Stage 3: Two-tier association
    // First: Match high-confidence detections to confirmed+tentative tracks
    associateHighConfidence(current_num_detections_);
    auto t_high = std::chrono::high_resolution_clock::now();

    // Second: Match low-confidence detections to remaining unmatched tracks
    associateLowConfidence(current_num_detections_);
    auto t_low = std::chrono::high_resolution_clock::now();

    // Stage 4: Match remaining detections against lost tracks (reactivation)
    associateLostTracks(current_num_detections_);
    auto t_lost = std::chrono::high_resolution_clock::now();

    // Stage 5: Update matched tracks with Kalman filter
    updateMatchedTracks(current_num_detections_);
    auto t_update = std::chrono::high_resolution_clock::now();

    // Stage 6: Handle unmatched tracks (age/mark lost)
    handleUnmatchedTracks();
    auto t_age = std::chrono::high_resolution_clock::now();

    // Stage 7: Create new tracks from unmatched high-confidence detections
    createNewTracks(current_num_detections_);
    auto t_new = std::chrono::high_resolution_clock::now();

    // Stage 8: Remove duplicate tracks
    removeDuplicateTracks();
    auto t_dedup = std::chrono::high_resolution_clock::now();

    // Count active tracks
    CUDA_CHECK(cudaMemsetAsync(d_num_active_tracks_, 0, sizeof(int), stream_));
    kernelCountActive<<<(config_.max_tracks + 255) / 256, 256, 0, stream_>>>(
        d_track_active_, d_num_active_tracks_, config_.max_tracks
    );
    CUDA_CHECK(cudaMemcpyAsync(&num_active_tracks_, d_num_active_tracks_,
                               sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto t_end = std::chrono::high_resolution_clock::now();

    // Update timing stats
    auto us = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };

    timing_.predict_us += us(t_copy, t_predict);
    timing_.gate_us += us(t_predict, t_gate);
    timing_.high_assoc_us += us(t_gate, t_high);
    timing_.low_assoc_us += us(t_high, t_low);
    timing_.lost_assoc_us += us(t_low, t_lost);
    timing_.update_us += us(t_lost, t_update);
    timing_.age_us += us(t_update, t_age);
    timing_.new_track_us += us(t_age, t_new);
    timing_.dedup_us += us(t_new, t_dedup);
    timing_.total_us += us(t_start, t_end);
    timing_.frame_count++;

    return num_active_tracks_;
}

void GPUTracker::predictTracks() {
    if (num_active_tracks_ == 0) return;

    int block = 256;
    int grid = (config_.max_tracks + block - 1) / block;

    kernelKalmanPredict<<<grid, block, 0, stream_>>>(
        d_predicted_poses_,
        d_track_poses_,
        d_track_velocities_,
        d_track_active_,
        d_track_states_,
        config_.max_tracks,
        1.0f  // dt = 1 frame
    );
}

void GPUTracker::computeSpatialGating(int num_detections) {
    if (num_active_tracks_ == 0 || num_detections == 0) return;

    int block = 256;

    // Compute bbox centers for tracks and detections
    int grid_t = (config_.max_tracks + block - 1) / block;
    int grid_d = (num_detections + block - 1) / block;

    kernelComputeBboxCenters<<<grid_t, block, 0, stream_>>>(
        d_predicted_poses_, d_track_centers_, config_.max_tracks
    );
    kernelComputeBboxCenters<<<grid_d, block, 0, stream_>>>(
        d_det_poses_, d_det_centers_, num_detections
    );

    // Build gate mask with adaptive velocity-based threshold
    dim3 block2d(16, 16);
    dim3 grid2d((num_detections + 15) / 16, (config_.max_tracks + 15) / 16);

    kernelSpatialGate<<<grid2d, block2d, 0, stream_>>>(
        d_track_centers_,
        d_det_centers_,
        d_track_velocities_,
        d_track_active_,
        d_track_states_,
        d_gate_mask_,
        config_.max_tracks,
        num_detections,
        GATE_THRESHOLD
    );
}

void GPUTracker::associateHighConfidence(int num_detections) {
    if (num_active_tracks_ == 0 || num_detections == 0) return;

    // Split detections by confidence
    int block = 256;
    int grid = (num_detections + block - 1) / block;

    kernelSplitDetections<<<grid, block, 0, stream_>>>(
        d_det_scores_,
        d_high_conf_mask_,
        d_low_conf_mask_,
        num_detections,
        config_.high_thresh,
        config_.low_thresh
    );

    dim3 block2d(16, 16);
    dim3 grid2d((num_detections + 15) / 16, (config_.max_tracks + 15) / 16);

    // IMPORTANT: Exclude LOST tracks from high-confidence tier
    // LOST tracks are handled separately in associateLostTracks
    kernelMaskTracksByState<<<grid2d, block2d, 0, stream_>>>(
        d_gate_mask_,
        d_track_states_,
        d_track_active_,
        config_.max_tracks,
        num_detections,
        TRACK_STATE_LOST  // Exclude lost tracks
    );

    // Compute OKS cost matrix with visibility masking and gating
    kernelOKSWithGating<<<grid2d, block2d, 0, stream_>>>(
        d_predicted_poses_,
        d_det_poses_,
        d_gate_mask_,
        d_track_active_,
        d_cost_matrix_,
        config_.max_tracks,
        num_detections,
        VISIBILITY_THRESHOLD
    );

    // Solve assignment for confirmed and tentative tracks only
    assignment_->solveDeviceAsyncWithActive(
        d_cost_matrix_,
        config_.max_tracks,
        num_detections,
        d_row_assignments_,
        d_col_assignments_,
        d_track_active_,
        config_.match_threshold,
        stream_
    );

    // CRITICAL: Lock matched pairs to prevent reassignment in subsequent tiers
    kernelLockMatchedPairs<<<grid2d, block2d, 0, stream_>>>(
        d_cost_matrix_,
        d_gate_mask_,
        d_row_assignments_,
        d_col_assignments_,
        d_track_active_,
        config_.max_tracks,
        num_detections
    );
}

void GPUTracker::associateLowConfidence(int num_detections) {
    if (num_active_tracks_ == 0 || num_detections == 0) return;

    // SAVE prior tier assignments (high-conf tier results)
    CUDA_CHECK(cudaMemcpyAsync(d_row_assign_backup_, d_row_assignments_,
                               config_.max_tracks * sizeof(int), cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_col_assign_backup_, d_col_assignments_,
                               num_detections * sizeof(int), cudaMemcpyDeviceToDevice, stream_));

    // Use torso-only OKS for low confidence detections (more robust)
    // Note: cost_matrix already has locked pairs from high-conf tier (cost = inf)
    dim3 block2d(16, 16);
    dim3 grid2d((num_detections + 15) / 16, (config_.max_tracks + 15) / 16);

    kernelTorsoOKS<<<grid2d, block2d, 0, stream_>>>(
        d_predicted_poses_,
        d_det_poses_,
        d_gate_mask_,
        d_track_active_,
        d_cost_matrix_,
        config_.max_tracks,
        num_detections
    );

    // Re-run assignment for unmatched tracks/detections
    // Locked pairs already have cost = inf and gate_mask = 0
    assignment_->solveDeviceAsyncWithActive(
        d_cost_matrix_,
        config_.max_tracks,
        num_detections,
        d_row_assignments_,
        d_col_assignments_,
        d_track_active_,
        config_.match_threshold * 1.2f,  // Slightly relaxed for low-conf
        stream_
    );

    // MERGE: Restore prior tier assignments, only keep new matches for unassigned
    int block = 256;
    int grid_t = (config_.max_tracks + block - 1) / block;
    int grid_d = (num_detections + block - 1) / block;

    kernelMergeAssignments<<<grid_t, block, 0, stream_>>>(
        d_row_assign_backup_, d_row_assignments_, config_.max_tracks
    );
    kernelMergeAssignments<<<grid_d, block, 0, stream_>>>(
        d_col_assign_backup_, d_col_assignments_, num_detections
    );

    // Lock newly matched pairs for the lost track recovery tier
    kernelLockMatchedPairs<<<grid2d, block2d, 0, stream_>>>(
        d_cost_matrix_,
        d_gate_mask_,
        d_row_assignments_,
        d_col_assignments_,
        d_track_active_,
        config_.max_tracks,
        num_detections
    );
}

void GPUTracker::associateLostTracks(int num_detections) {
    if (num_detections == 0) return;

    // SAVE prior tier assignments (high-conf + low-conf results)
    CUDA_CHECK(cudaMemcpyAsync(d_row_assign_backup_, d_row_assignments_,
                               config_.max_tracks * sizeof(int), cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_col_assign_backup_, d_col_assignments_,
                               num_detections * sizeof(int), cudaMemcpyDeviceToDevice, stream_));

    dim3 block2d(16, 16);
    dim3 grid2d((num_detections + 15) / 16, (config_.max_tracks + 15) / 16);

    // IMPORTANT: Use separate d_lost_gate_mask_ buffer for lost tier
    // This preserves the locks in d_gate_mask_ from previous tiers
    //
    // Strategy:
    // 1. Compute fresh spatial gate into d_lost_gate_mask_ with wider threshold
    // 2. Mask out non-LOST tracks
    // 3. Lock previously matched detections in the lost gate buffer
    // 4. Use lost gate buffer for OKS computation

    // Compute spatial gate into SEPARATE buffer (preserves d_gate_mask_ locks)
    kernelSpatialGate<<<grid2d, block2d, 0, stream_>>>(
        d_track_centers_,
        d_det_centers_,
        d_track_velocities_,
        d_track_active_,
        d_track_states_,
        d_lost_gate_mask_,  // Use separate buffer!
        config_.max_tracks,
        num_detections,
        GATE_THRESHOLD * 1.3f  // Tighter gate for lost track recovery
    );

    // Mask out non-LOST tracks - only LOST tracks should participate
    kernelMaskTracksByState<<<grid2d, block2d, 0, stream_>>>(
        d_lost_gate_mask_,  // Use lost gate buffer
        d_track_states_,
        d_track_active_,
        config_.max_tracks,
        num_detections,
        TRACK_STATE_CONFIRMED  // Exclude confirmed
    );
    kernelMaskTracksByState<<<grid2d, block2d, 0, stream_>>>(
        d_lost_gate_mask_,  // Use lost gate buffer
        d_track_states_,
        d_track_active_,
        config_.max_tracks,
        num_detections,
        TRACK_STATE_TENTATIVE  // Exclude tentative
    );

    // Lock previously matched detections in the lost gate buffer
    // This ensures detections matched in high/low tiers aren't matched to lost tracks
    kernelLockMatchedPairs<<<grid2d, block2d, 0, stream_>>>(
        d_cost_matrix_,
        d_lost_gate_mask_,  // Lock in lost gate buffer
        d_row_assignments_,
        d_col_assignments_,
        d_track_active_,
        config_.max_tracks,
        num_detections
    );

    // Compute OKS using the lost gate buffer (not d_gate_mask_!)
    kernelOKSWithGating<<<grid2d, block2d, 0, stream_>>>(
        d_predicted_poses_,
        d_det_poses_,
        d_lost_gate_mask_,  // Use lost gate buffer
        d_track_active_,
        d_cost_matrix_,
        config_.max_tracks,
        num_detections,
        0.2f  // Visibility threshold for lost tracks (stricter)
    );

    // Solve assignment for lost tracks with relaxed threshold
    assignment_->solveDeviceAsyncWithActive(
        d_cost_matrix_,
        config_.max_tracks,
        num_detections,
        d_row_assignments_,
        d_col_assignments_,
        d_track_active_,
        config_.match_threshold,  // Same threshold for lost tracks (no relaxation)
        stream_
    );

    // MERGE: Restore prior tier assignments, only keep new matches for unassigned
    int block = 256;
    int grid_t = (config_.max_tracks + block - 1) / block;
    int grid_d = (num_detections + block - 1) / block;

    kernelMergeAssignments<<<grid_t, block, 0, stream_>>>(
        d_row_assign_backup_, d_row_assignments_, config_.max_tracks
    );
    kernelMergeAssignments<<<grid_d, block, 0, stream_>>>(
        d_col_assign_backup_, d_col_assignments_, num_detections
    );
}

void GPUTracker::updateMatchedTracks(int num_detections) {
    if (num_detections == 0) return;

    int block = 256;
    int grid = (config_.max_tracks + block - 1) / block;

    // Kalman update for matched tracks
    kernelKalmanUpdate<<<grid, block, 0, stream_>>>(
        d_track_poses_,
        d_track_velocities_,
        d_det_poses_,
        d_row_assignments_,
        d_track_active_,
        config_.max_tracks,
        0.1f,   // Process noise
        0.3f    // Measurement noise
    );

    // Update track metadata
    kernelUpdateMatchedTracks<<<grid, block, 0, stream_>>>(
        d_track_poses_,
        d_track_scores_,
        d_track_hits_,
        d_track_ages_,
        d_track_states_,
        d_track_last_frame_,
        d_track_active_,
        d_det_poses_,
        d_det_scores_,
        d_row_assignments_,
        config_.max_tracks,
        current_frame_,
        config_.min_hits
    );
}

void GPUTracker::handleUnmatchedTracks() {
    int block = 256;
    int grid = (config_.max_tracks + block - 1) / block;

    kernelAgeUnmatchedTracks<<<grid, block, 0, stream_>>>(
        d_track_ages_,
        d_track_states_,
        d_track_active_,
        d_row_assignments_,
        config_.max_tracks,
        config_.max_age,
        LOST_WINDOW
    );
}

void GPUTracker::createNewTracks(int num_detections) {
    if (num_detections == 0) return;

    CUDA_CHECK(cudaMemsetAsync(d_slot_for_det_, 0xFF, num_detections * sizeof(int), stream_));

    int block = 256;
    int grid = (num_detections + block - 1) / block;

    kernelAllocateNewTrackSlots<<<grid, block, 0, stream_>>>(
        d_track_active_,
        d_col_assignments_,
        d_det_scores_,
        d_slot_for_det_,
        d_next_slot_hint_,
        config_.max_tracks,
        num_detections,
        config_.new_track_thresh
    );

    kernelInitNewTracks<<<num_detections, 17, 0, stream_>>>(
        d_track_poses_,
        d_track_scores_,
        d_track_ids_,
        d_track_hits_,
        d_track_ages_,
        d_track_states_,
        d_track_last_frame_,
        d_track_velocities_,
        d_det_poses_,
        d_det_scores_,
        d_col_assignments_,
        d_slot_for_det_,
        d_next_track_id_,
        num_detections,
        current_frame_,
        config_.new_track_thresh
    );
}

void GPUTracker::removeDuplicateTracks() {
    // Compute IoU between all track pairs
    // Note: Only considers CONFIRMED tracks with sufficient hits
    dim3 block2d(16, 16);
    dim3 grid2d((config_.max_tracks + 15) / 16, (config_.max_tracks + 15) / 16);

    kernelTrackIoU<<<grid2d, block2d, 0, stream_>>>(
        d_track_centers_,
        d_track_active_,
        d_track_states_,
        d_track_hits_,
        d_track_iou_matrix_,
        config_.max_tracks,
        config_.min_hits  // Only dedup tracks with enough hits
    );

    // Remove duplicates (LOST tracks are preserved)
    int block = 256;
    int grid = (config_.max_tracks + block - 1) / block;

    kernelRemoveDuplicates<<<grid, block, 0, stream_>>>(
        d_track_iou_matrix_,
        d_track_active_,
        d_track_states_,
        d_track_hits_,
        d_track_ids_,
        config_.max_tracks,
        DEDUP_IOU_THRESHOLD
    );
}

std::vector<TrackOutput> GPUTracker::getActiveTracks() {
    std::vector<TrackOutput> tracks;

    // Copy data from device
    std::vector<int> h_col_assign(config_.max_detections);
    std::vector<int> h_track_ids(config_.max_tracks);
    std::vector<int> h_track_states(config_.max_tracks);
    std::vector<int> h_track_hits(config_.max_tracks);
    std::vector<float> h_track_poses(config_.max_tracks * 17 * 3);
    std::vector<float> h_track_scores(config_.max_tracks);
    std::vector<int> h_track_active(config_.max_tracks);

    CUDA_CHECK(cudaMemcpy(h_col_assign.data(), d_col_assignments_,
                          config_.max_detections * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_track_ids.data(), d_track_ids_,
                          config_.max_tracks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_track_states.data(), d_track_states_,
                          config_.max_tracks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_track_hits.data(), d_track_hits_,
                          config_.max_tracks * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_track_poses.data(), d_track_poses_,
                          config_.max_tracks * 17 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_track_scores.data(), d_track_scores_,
                          config_.max_tracks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_track_active.data(), d_track_active_,
                          config_.max_tracks * sizeof(int), cudaMemcpyDeviceToHost));

    // Copy current detection poses for output
    std::vector<float> h_det_poses(config_.max_detections * 17 * 3);
    std::vector<float> h_det_scores(config_.max_detections);
    CUDA_CHECK(cudaMemcpy(h_det_poses.data(), d_det_poses_,
                          config_.max_detections * 17 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_det_scores.data(), d_det_scores_,
                          config_.max_detections * sizeof(float), cudaMemcpyDeviceToHost));

    // Build output: for each detection, output its track info
    for (int det_idx = 0; det_idx < current_num_detections_; det_idx++) {
        int track_slot = h_col_assign[det_idx];
        if (track_slot < 0) continue;

        // Only output confirmed tracks with enough hits
        if (h_track_states[track_slot] == TRACK_STATE_TENTATIVE) {
            if (h_track_hits[track_slot] < config_.min_hits) continue;
        }
        if (h_track_states[track_slot] == TRACK_STATE_LOST) continue;

        TrackOutput output;
        output.track_id = h_track_ids[track_slot];
        output.score = h_det_scores[det_idx];

        // Use smoothed track poses (Kalman filtered) instead of raw detection
        for (int kp = 0; kp < 17; kp++) {
            int offset = track_slot * 17 * 3 + kp * 3;
            output.keypoints[kp].x = h_track_poses[offset + 0];
            output.keypoints[kp].y = h_track_poses[offset + 1];
            output.keypoints[kp].confidence = h_track_poses[offset + 2];
        }

        // Compute bbox from keypoints
        float min_x = 1e9f, min_y = 1e9f, max_x = -1e9f, max_y = -1e9f;
        for (int kp = 0; kp < 17; kp++) {
            if (output.keypoints[kp].confidence > 0.2f) {
                min_x = std::min(min_x, output.keypoints[kp].x);
                min_y = std::min(min_y, output.keypoints[kp].y);
                max_x = std::max(max_x, output.keypoints[kp].x);
                max_y = std::max(max_y, output.keypoints[kp].y);
            }
        }

        float pad_x = (max_x - min_x) * 0.1f;
        float pad_y = (max_y - min_y) * 0.1f;
        output.bbox[0] = min_x - pad_x;
        output.bbox[1] = min_y - pad_y;
        output.bbox[2] = max_x + pad_x;
        output.bbox[3] = max_y + pad_y;

        tracks.push_back(output);
    }

    return tracks;
}

void GPUTracker::printTimingStats() const {
    if (timing_.frame_count == 0) return;

    float n = timing_.frame_count;
    printf("\n=== GPU Tracker Timing Stats (%d frames) ===\n", timing_.frame_count);
    printf("  Predict:      %7.2f us/frame\n", timing_.predict_us / n);
    printf("  Spatial gate: %7.2f us/frame\n", timing_.gate_us / n);
    printf("  High assoc:   %7.2f us/frame\n", timing_.high_assoc_us / n);
    printf("  Low assoc:    %7.2f us/frame\n", timing_.low_assoc_us / n);
    printf("  Lost assoc:   %7.2f us/frame\n", timing_.lost_assoc_us / n);
    printf("  Update:       %7.2f us/frame\n", timing_.update_us / n);
    printf("  Age tracks:   %7.2f us/frame\n", timing_.age_us / n);
    printf("  New tracks:   %7.2f us/frame\n", timing_.new_track_us / n);
    printf("  Dedup:        %7.2f us/frame\n", timing_.dedup_us / n);
    printf("  -----------------------------\n");
    printf("  TOTAL:        %7.2f us/frame (%.1f FPS potential)\n",
           timing_.total_us / n, 1e6 * n / timing_.total_us);
}

void GPUTracker::captureGraph() {
    // Not implemented for complex multi-stage pipeline
    printf("[GPUTracker] Graph capture not available for production tracker\n");
}

void GPUTracker::executeGraph() {
    printf("[GPUTracker] Graph execution not available\n");
}

}  // namespace cuda
}  // namespace posebyte
