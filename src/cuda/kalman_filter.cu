#include "cuda/kalman_filter.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

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
// CUDA Kernels - All batched for maximum parallelism
// ============================================================================

// Kernel: Batch initialize multiple tracks
// Each block handles one track, threads handle keypoints
__global__ void kernelBatchInitiate(
    float* __restrict__ means,
    float* __restrict__ covariances,
    const float* __restrict__ detections,  // [num_new, 17*3]
    const int* __restrict__ track_slots,   // [num_new]
    int num_new,
    int state_dim,
    int num_keypoints
) {
    int new_idx = blockIdx.x;
    int kp_idx = threadIdx.x;

    if (new_idx >= num_new || kp_idx >= num_keypoints) return;

    int track_idx = track_slots[new_idx];
    int state_offset = track_idx * state_dim;
    int det_offset = new_idx * num_keypoints * 3;
    int kp_state_base = kp_idx * 8;
    int kp_det_base = kp_idx * 3;

    // Load detection data
    float x = detections[det_offset + kp_det_base];
    float y = detections[det_offset + kp_det_base + 1];
    float conf = detections[det_offset + kp_det_base + 2];

    // Initialize mean: position from detection, derivatives = 0
    means[state_offset + kp_state_base + 0] = x;
    means[state_offset + kp_state_base + 1] = y;
    means[state_offset + kp_state_base + 2] = 0.0f;  // vx
    means[state_offset + kp_state_base + 3] = 0.0f;  // vy
    means[state_offset + kp_state_base + 4] = 0.0f;  // ax
    means[state_offset + kp_state_base + 5] = 0.0f;  // ay
    means[state_offset + kp_state_base + 6] = 0.0f;  // jx
    means[state_offset + kp_state_base + 7] = 0.0f;  // jy

    // Initialize diagonal covariance
    float pos_var = (conf > 0.0f) ? 10.0f : 1000.0f;
    float vel_var = 100.0f;
    float acc_var = 100.0f;
    float jrk_var = 100.0f;

    int cov_offset = track_idx * state_dim * state_dim;

    // Clear and set diagonal (each thread handles its keypoint's 8 state dims)
    for (int i = 0; i < 8; i++) {
        int state_i = kp_state_base + i;
        // Clear row
        for (int j = 0; j < state_dim; j++) {
            covariances[cov_offset + state_i * state_dim + j] = 0.0f;
        }
        // Set diagonal
        float var;
        if (i < 2) var = pos_var;
        else if (i < 4) var = vel_var;
        else if (i < 6) var = acc_var;
        else var = jrk_var;
        covariances[cov_offset + state_i * state_dim + state_i] = var;
    }
}

// Kernel: Predict mean for all tracks (third-order motion model)
// Each block = one track, threads = keypoints
__global__ void kernelPredictMean(
    float* __restrict__ means,
    int num_tracks,
    int state_dim,
    int num_keypoints,
    float accel_memory,
    float jerk_memory
) {
    int track_idx = blockIdx.x;
    int kp_idx = threadIdx.x;

    if (track_idx >= num_tracks || kp_idx >= num_keypoints) return;

    int state_offset = track_idx * state_dim;
    int kp_base = kp_idx * 8;

    // Load current state
    float px = means[state_offset + kp_base + 0];
    float py = means[state_offset + kp_base + 1];
    float vx = means[state_offset + kp_base + 2];
    float vy = means[state_offset + kp_base + 3];
    float ax = means[state_offset + kp_base + 4];
    float ay = means[state_offset + kp_base + 5];
    float jx = means[state_offset + kp_base + 6];
    float jy = means[state_offset + kp_base + 7];

    // Third-order motion model (dt = 1):
    // p' = p + v + 0.5*a + (1/6)*j
    // v' = v + a + 0.5*j
    // a' = a * memory
    // j' = j * memory
    float new_px = px + vx + 0.5f * ax + (1.0f/6.0f) * jx;
    float new_py = py + vy + 0.5f * ay + (1.0f/6.0f) * jy;
    float new_vx = vx + ax + 0.5f * jx;
    float new_vy = vy + ay + 0.5f * jy;
    float new_ax = ax * accel_memory;
    float new_ay = ay * accel_memory;
    float new_jx = jx * jerk_memory;
    float new_jy = jy * jerk_memory;

    // Store predicted state
    means[state_offset + kp_base + 0] = new_px;
    means[state_offset + kp_base + 1] = new_py;
    means[state_offset + kp_base + 2] = new_vx;
    means[state_offset + kp_base + 3] = new_vy;
    means[state_offset + kp_base + 4] = new_ax;
    means[state_offset + kp_base + 5] = new_ay;
    means[state_offset + kp_base + 6] = new_jx;
    means[state_offset + kp_base + 7] = new_jy;
}

// Kernel: Predict covariance (diagonal approximation for speed)
__global__ void kernelPredictCovariance(
    float* __restrict__ covariances,
    int num_tracks,
    int state_dim
) {
    int track_idx = blockIdx.x;
    int state_i = threadIdx.x;

    if (track_idx >= num_tracks || state_i >= state_dim) return;

    int cov_offset = track_idx * state_dim * state_dim;
    int diag_idx = cov_offset + state_i * state_dim + state_i;

    // Determine noise based on state type
    int state_type = state_i % 8;
    float noise;

    if (state_type < 2) {
        noise = 1.0f;       // Position noise
    } else if (state_type < 4) {
        noise = 0.5f;       // Velocity noise
    } else if (state_type < 6) {
        noise = 0.1f;       // Acceleration noise
    } else {
        noise = 0.05f;      // Jerk noise
    }

    // Add process noise (diagonal approximation)
    covariances[diag_idx] += noise * noise;
}

// Kernel: Batch update - all matched tracks in parallel
// Each block handles one match, threads handle keypoints
__global__ void kernelBatchUpdate(
    float* __restrict__ means,
    float* __restrict__ covariances,
    const float* __restrict__ detections,  // [num_dets, 17*3]
    const int* __restrict__ matches,       // [num_matches, 2] (track_slot, det_idx)
    int num_matches,
    int state_dim,
    int num_keypoints
) {
    int match_idx = blockIdx.x;
    int kp_idx = threadIdx.x;

    if (match_idx >= num_matches || kp_idx >= num_keypoints) return;

    int track_idx = matches[match_idx * 2];
    int det_idx = matches[match_idx * 2 + 1];

    int state_offset = track_idx * state_dim;
    int cov_offset = track_idx * state_dim * state_dim;
    int det_offset = det_idx * num_keypoints * 3;
    int kp_base = kp_idx * 8;
    int kp_det_base = kp_idx * 3;

    // Get measurement
    float z_x = detections[det_offset + kp_det_base];
    float z_y = detections[det_offset + kp_det_base + 1];
    float conf = detections[det_offset + kp_det_base + 2];

    // Skip low confidence keypoints
    if (conf < 0.1f) return;

    // Get predicted position
    float pred_x = means[state_offset + kp_base + 0];
    float pred_y = means[state_offset + kp_base + 1];

    // Innovation (measurement residual)
    float y_x = z_x - pred_x;
    float y_y = z_y - pred_y;

    // Get position variance
    float P_xx = covariances[cov_offset + (kp_base + 0) * state_dim + (kp_base + 0)];
    float P_yy = covariances[cov_offset + (kp_base + 1) * state_dim + (kp_base + 1)];

    // Measurement noise (scaled by inverse confidence)
    float R = 5.0f / (conf + 0.1f);

    // Innovation covariance
    float S_xx = P_xx + R;
    float S_yy = P_yy + R;

    // Kalman gain
    float K_x = P_xx / S_xx;
    float K_y = P_yy / S_yy;

    // Update state mean
    means[state_offset + kp_base + 0] += K_x * y_x;
    means[state_offset + kp_base + 1] += K_y * y_y;

    // Velocity update (coupled)
    float K_v = 0.5f * K_x;
    means[state_offset + kp_base + 2] += K_v * y_x;
    means[state_offset + kp_base + 3] += K_v * y_y;

    // Update covariance
    covariances[cov_offset + (kp_base + 0) * state_dim + (kp_base + 0)] = (1.0f - K_x) * P_xx;
    covariances[cov_offset + (kp_base + 1) * state_dim + (kp_base + 1)] = (1.0f - K_y) * P_yy;
}

// Kernel: Extract poses from Kalman state to device buffer
// For direct GPU-to-GPU pipeline without D2H
__global__ void kernelExtractPosesToDevice(
    const float* __restrict__ means,
    float* __restrict__ out_poses,         // [num_tracks, 17*3]
    const int* __restrict__ track_slots,   // [num_tracks] - which slots to extract
    int num_tracks,
    int state_dim,
    int num_keypoints
) {
    int track_output_idx = blockIdx.x;
    int kp_idx = threadIdx.x;

    if (track_output_idx >= num_tracks || kp_idx >= num_keypoints) return;

    int track_slot = track_slots[track_output_idx];
    int state_offset = track_slot * state_dim;
    int out_offset = track_output_idx * num_keypoints * 3;
    int kp_base = kp_idx * 8;
    int kp_out_base = kp_idx * 3;

    // Extract x, y, conf=1.0 from state
    out_poses[out_offset + kp_out_base + 0] = means[state_offset + kp_base + 0];
    out_poses[out_offset + kp_out_base + 1] = means[state_offset + kp_base + 1];
    out_poses[out_offset + kp_out_base + 2] = 1.0f;  // Confidence
}

// Kernel: Extract single pose (for legacy compatibility)
__global__ void kernelExtractSinglePose(
    const float* __restrict__ means,
    float* __restrict__ out_pose,          // [17*3]
    int track_idx,
    int state_dim,
    int num_keypoints
) {
    int kp_idx = threadIdx.x;
    if (kp_idx >= num_keypoints) return;

    int state_offset = track_idx * state_dim;
    int kp_base = kp_idx * 8;

    out_pose[kp_idx * 3 + 0] = means[state_offset + kp_base + 0];
    out_pose[kp_idx * 3 + 1] = means[state_offset + kp_base + 1];
    out_pose[kp_idx * 3 + 2] = 1.0f;
}

// ============================================================================
// KalmanFilterCUDA Implementation
// ============================================================================

KalmanFilterCUDA::KalmanFilterCUDA(int max_tracks) : max_tracks_(max_tracks) {
    // ========================================================================
    // Allocate Device Memory (persistent)
    // ========================================================================
    CUDA_CHECK(cudaMalloc(&d_means_, max_tracks * TOTAL_STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covariances_, max_tracks * TOTAL_STATE_DIM * TOTAL_STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_transition_, TOTAL_STATE_DIM * TOTAL_STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_process_noise_, TOTAL_STATE_DIM * TOTAL_STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_measurement_noise_, NUM_KEYPOINTS * 2 * NUM_KEYPOINTS * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_workspace_, max_tracks * TOTAL_STATE_DIM * sizeof(float)));

    // Pre-allocated buffers for batch operations
    CUDA_CHECK(cudaMalloc(&d_det_buffer_, max_tracks * NUM_KEYPOINTS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_match_buffer_, max_tracks * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_slot_buffer_, max_tracks * sizeof(int)));

    // ========================================================================
    // Allocate Pinned Host Memory (for fast transfers)
    // ========================================================================
    CUDA_CHECK(cudaMallocHost(&h_det_pinned_, max_tracks * NUM_KEYPOINTS * 3 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_match_pinned_, max_tracks * 2 * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_slot_pinned_, max_tracks * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_pose_pinned_, max_tracks * NUM_KEYPOINTS * 3 * sizeof(float)));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_means_, 0, max_tracks * TOTAL_STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_covariances_, 0, max_tracks * TOTAL_STATE_DIM * TOTAL_STATE_DIM * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream_));

    // Initialize matrices
    initTransitionMatrix(0.9f, 0.9f);
    initNoiseCovariances();
}

KalmanFilterCUDA::~KalmanFilterCUDA() {
    // Device memory
    cudaFree(d_means_);
    cudaFree(d_covariances_);
    cudaFree(d_transition_);
    cudaFree(d_process_noise_);
    cudaFree(d_measurement_noise_);
    cudaFree(d_workspace_);
    cudaFree(d_det_buffer_);
    cudaFree(d_match_buffer_);
    cudaFree(d_slot_buffer_);

    // Pinned host memory
    cudaFreeHost(h_det_pinned_);
    cudaFreeHost(h_match_pinned_);
    cudaFreeHost(h_slot_pinned_);
    cudaFreeHost(h_pose_pinned_);

    cudaStreamDestroy(stream_);
}

void KalmanFilterCUDA::initTransitionMatrix(float accel_memory, float jerk_memory) {
    std::vector<float> h_transition(TOTAL_STATE_DIM * TOTAL_STATE_DIM, 0.0f);

    for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
        int base = kp * 8;

        // Position row (x)
        h_transition[(base + 0) * TOTAL_STATE_DIM + (base + 0)] = 1.0f;
        h_transition[(base + 0) * TOTAL_STATE_DIM + (base + 2)] = 1.0f;
        h_transition[(base + 0) * TOTAL_STATE_DIM + (base + 4)] = 0.5f;
        h_transition[(base + 0) * TOTAL_STATE_DIM + (base + 6)] = 1.0f / 6.0f;

        // Position row (y)
        h_transition[(base + 1) * TOTAL_STATE_DIM + (base + 1)] = 1.0f;
        h_transition[(base + 1) * TOTAL_STATE_DIM + (base + 3)] = 1.0f;
        h_transition[(base + 1) * TOTAL_STATE_DIM + (base + 5)] = 0.5f;
        h_transition[(base + 1) * TOTAL_STATE_DIM + (base + 7)] = 1.0f / 6.0f;

        // Velocity row (x)
        h_transition[(base + 2) * TOTAL_STATE_DIM + (base + 2)] = 1.0f;
        h_transition[(base + 2) * TOTAL_STATE_DIM + (base + 4)] = 1.0f;
        h_transition[(base + 2) * TOTAL_STATE_DIM + (base + 6)] = 0.5f;

        // Velocity row (y)
        h_transition[(base + 3) * TOTAL_STATE_DIM + (base + 3)] = 1.0f;
        h_transition[(base + 3) * TOTAL_STATE_DIM + (base + 5)] = 1.0f;
        h_transition[(base + 3) * TOTAL_STATE_DIM + (base + 7)] = 0.5f;

        // Acceleration (with memory)
        h_transition[(base + 4) * TOTAL_STATE_DIM + (base + 4)] = accel_memory;
        h_transition[(base + 5) * TOTAL_STATE_DIM + (base + 5)] = accel_memory;

        // Jerk (with memory)
        h_transition[(base + 6) * TOTAL_STATE_DIM + (base + 6)] = jerk_memory;
        h_transition[(base + 7) * TOTAL_STATE_DIM + (base + 7)] = jerk_memory;
    }

    CUDA_CHECK(cudaMemcpy(d_transition_, h_transition.data(),
                          TOTAL_STATE_DIM * TOTAL_STATE_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void KalmanFilterCUDA::initNoiseCovariances() {
    std::vector<float> h_process(TOTAL_STATE_DIM * TOTAL_STATE_DIM, 0.0f);

    for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
        int base = kp * 8;
        h_process[(base + 0) * TOTAL_STATE_DIM + (base + 0)] = 1.0f;
        h_process[(base + 1) * TOTAL_STATE_DIM + (base + 1)] = 1.0f;
        h_process[(base + 2) * TOTAL_STATE_DIM + (base + 2)] = 0.25f;
        h_process[(base + 3) * TOTAL_STATE_DIM + (base + 3)] = 0.25f;
        h_process[(base + 4) * TOTAL_STATE_DIM + (base + 4)] = 0.01f;
        h_process[(base + 5) * TOTAL_STATE_DIM + (base + 5)] = 0.01f;
        h_process[(base + 6) * TOTAL_STATE_DIM + (base + 6)] = 0.001f;
        h_process[(base + 7) * TOTAL_STATE_DIM + (base + 7)] = 0.001f;
    }

    CUDA_CHECK(cudaMemcpy(d_process_noise_, h_process.data(),
                          TOTAL_STATE_DIM * TOTAL_STATE_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));

    int meas_dim = NUM_KEYPOINTS * 2;
    std::vector<float> h_meas(meas_dim * meas_dim, 0.0f);
    for (int i = 0; i < meas_dim; i++) {
        h_meas[i * meas_dim + i] = 5.0f;
    }

    CUDA_CHECK(cudaMemcpy(d_measurement_noise_, h_meas.data(),
                          meas_dim * meas_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// ============================================================================
// GPU-Native Batch Operations (async)
// ============================================================================

void KalmanFilterCUDA::predictAsync(int num_active_tracks, cudaStream_t stream) {
    if (num_active_tracks == 0) return;

    cudaStream_t s = stream ? stream : stream_;

    // Predict mean
    kernelPredictMean<<<num_active_tracks, NUM_KEYPOINTS, 0, s>>>(
        d_means_, num_active_tracks, TOTAL_STATE_DIM, NUM_KEYPOINTS,
        accel_memory_, jerk_memory_
    );

    // Predict covariance
    kernelPredictCovariance<<<num_active_tracks, TOTAL_STATE_DIM, 0, s>>>(
        d_covariances_, num_active_tracks, TOTAL_STATE_DIM
    );
    // No sync - caller decides when to sync
}

void KalmanFilterCUDA::updateBatchAsync(
    const float* d_detections,
    const int* d_matches,
    int num_matches,
    cudaStream_t stream
) {
    if (num_matches == 0) return;

    cudaStream_t s = stream ? stream : stream_;

    kernelBatchUpdate<<<num_matches, NUM_KEYPOINTS, 0, s>>>(
        d_means_, d_covariances_,
        d_detections, d_matches,
        num_matches, TOTAL_STATE_DIM, NUM_KEYPOINTS
    );
    // No sync
}

void KalmanFilterCUDA::initiateBatchAsync(
    const float* d_detections,
    const int* d_track_slots,
    int num_new,
    cudaStream_t stream
) {
    if (num_new == 0) return;

    cudaStream_t s = stream ? stream : stream_;

    kernelBatchInitiate<<<num_new, NUM_KEYPOINTS, 0, s>>>(
        d_means_, d_covariances_,
        d_detections, d_track_slots,
        num_new, TOTAL_STATE_DIM, NUM_KEYPOINTS
    );
    // No sync
}

void KalmanFilterCUDA::extractPosesToDeviceAsync(
    float* d_out_poses,
    const int* d_track_slots,
    int num_tracks,
    cudaStream_t stream
) {
    if (num_tracks == 0) return;

    cudaStream_t s = stream ? stream : stream_;

    kernelExtractPosesToDevice<<<num_tracks, NUM_KEYPOINTS, 0, s>>>(
        d_means_, d_out_poses, d_track_slots,
        num_tracks, TOTAL_STATE_DIM, NUM_KEYPOINTS
    );
    // No sync
}

void KalmanFilterCUDA::sync(cudaStream_t stream) {
    cudaStream_t s = stream ? stream : stream_;
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ============================================================================
// Legacy Host-Side Operations (for compatibility)
// ============================================================================

void KalmanFilterCUDA::initiate(int track_idx, const PoseDetection& detection) {
    if (track_idx < 0 || track_idx >= max_tracks_) {
        fprintf(stderr, "KalmanFilter::initiate: track_idx %d out of bounds\n", track_idx);
        return;
    }

    // Prepare detection in pinned memory
    for (int i = 0; i < NUM_KEYPOINTS; i++) {
        h_det_pinned_[i * 3 + 0] = detection.keypoints[i].x;
        h_det_pinned_[i * 3 + 1] = detection.keypoints[i].y;
        h_det_pinned_[i * 3 + 2] = detection.keypoints[i].confidence;
    }

    h_slot_pinned_[0] = track_idx;

    // Copy to device
    CUDA_CHECK(cudaMemcpyAsync(d_det_buffer_, h_det_pinned_,
                               NUM_KEYPOINTS * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_slot_buffer_, h_slot_pinned_,
                               sizeof(int), cudaMemcpyHostToDevice, stream_));

    // Launch batch init with single track
    initiateBatchAsync(d_det_buffer_, d_slot_buffer_, 1, stream_);

    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void KalmanFilterCUDA::predict(int num_tracks, float accel_memory, float jerk_memory) {
    if (num_tracks == 0) return;

    accel_memory_ = accel_memory;
    jerk_memory_ = jerk_memory;

    predictAsync(num_tracks, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void KalmanFilterCUDA::update(const PoseDetection* detections, const int* matches, int num_matches) {
    if (num_matches == 0) return;

    // Prepare data in pinned memory
    for (int i = 0; i < num_matches; i++) {
        int det_idx = matches[i * 2 + 1];
        for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
            h_det_pinned_[i * NUM_KEYPOINTS * 3 + kp * 3 + 0] = detections[det_idx].keypoints[kp].x;
            h_det_pinned_[i * NUM_KEYPOINTS * 3 + kp * 3 + 1] = detections[det_idx].keypoints[kp].y;
            h_det_pinned_[i * NUM_KEYPOINTS * 3 + kp * 3 + 2] = detections[det_idx].keypoints[kp].confidence;
        }
        h_match_pinned_[i * 2 + 0] = matches[i * 2 + 0];  // track_idx
        h_match_pinned_[i * 2 + 1] = i;  // Remap det_idx to local buffer index
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpyAsync(d_det_buffer_, h_det_pinned_,
                               num_matches * NUM_KEYPOINTS * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_match_buffer_, h_match_pinned_,
                               num_matches * 2 * sizeof(int),
                               cudaMemcpyHostToDevice, stream_));

    // Batch update
    updateBatchAsync(d_det_buffer_, d_match_buffer_, num_matches, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void KalmanFilterCUDA::getPredictedPose(int track_idx, PoseDetection& out_pose) const {
    // Extract single pose to pinned memory
    kernelExtractSinglePose<<<1, NUM_KEYPOINTS, 0, stream_>>>(
        d_means_, const_cast<float*>(h_pose_pinned_),  // Write directly to pinned
        track_idx, TOTAL_STATE_DIM, NUM_KEYPOINTS
    );

    // Actually need device buffer then copy
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, NUM_KEYPOINTS * 3 * sizeof(float)));

    kernelExtractSinglePose<<<1, NUM_KEYPOINTS, 0, stream_>>>(
        d_means_, d_temp, track_idx, TOTAL_STATE_DIM, NUM_KEYPOINTS
    );

    CUDA_CHECK(cudaMemcpyAsync(const_cast<float*>(h_pose_pinned_), d_temp,
                               NUM_KEYPOINTS * 3 * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    cudaFree(d_temp);

    // Convert to PoseDetection
    for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
        out_pose.keypoints[kp].x = h_pose_pinned_[kp * 3 + 0];
        out_pose.keypoints[kp].y = h_pose_pinned_[kp * 3 + 1];
        out_pose.keypoints[kp].confidence = h_pose_pinned_[kp * 3 + 2];
    }
}

void KalmanFilterCUDA::getAllPredictedPoses(PoseDetection* out_poses, int num_tracks) const {
    if (num_tracks == 0) return;

    // Prepare slot indices (0, 1, 2, ...)
    for (int i = 0; i < num_tracks; i++) {
        const_cast<int*>(h_slot_pinned_)[i] = i;
    }

    CUDA_CHECK(cudaMemcpyAsync(const_cast<int*>(d_slot_buffer_), h_slot_pinned_,
                               num_tracks * sizeof(int),
                               cudaMemcpyHostToDevice, stream_));

    // Extract all poses
    kernelExtractPosesToDevice<<<num_tracks, NUM_KEYPOINTS, 0, stream_>>>(
        d_means_, const_cast<float*>(d_det_buffer_),  // Reuse det buffer for output
        d_slot_buffer_, num_tracks, TOTAL_STATE_DIM, NUM_KEYPOINTS
    );

    // Copy back
    CUDA_CHECK(cudaMemcpyAsync(const_cast<float*>(h_pose_pinned_), d_det_buffer_,
                               num_tracks * NUM_KEYPOINTS * 3 * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Convert to PoseDetections
    for (int t = 0; t < num_tracks; t++) {
        for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
            out_poses[t].keypoints[kp].x = h_pose_pinned_[t * NUM_KEYPOINTS * 3 + kp * 3 + 0];
            out_poses[t].keypoints[kp].y = h_pose_pinned_[t * NUM_KEYPOINTS * 3 + kp * 3 + 1];
            out_poses[t].keypoints[kp].confidence = h_pose_pinned_[t * NUM_KEYPOINTS * 3 + kp * 3 + 2];
        }
    }
}

void KalmanFilterCUDA::getState(int track_idx, float* mean, float* covariance) const {
    CUDA_CHECK(cudaMemcpy(mean, d_means_ + track_idx * TOTAL_STATE_DIM,
                          TOTAL_STATE_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    if (covariance) {
        CUDA_CHECK(cudaMemcpy(covariance, d_covariances_ + track_idx * TOTAL_STATE_DIM * TOTAL_STATE_DIM,
                              TOTAL_STATE_DIM * TOTAL_STATE_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void KalmanFilterCUDA::resetTrack(int track_idx) {
    CUDA_CHECK(cudaMemsetAsync(d_means_ + track_idx * TOTAL_STATE_DIM, 0,
                               TOTAL_STATE_DIM * sizeof(float), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_covariances_ + track_idx * TOTAL_STATE_DIM * TOTAL_STATE_DIM, 0,
                               TOTAL_STATE_DIM * TOTAL_STATE_DIM * sizeof(float), stream_));
}

}  // namespace cuda
}  // namespace posebyte
