#include "cuda/hungarian.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <limits>

namespace posebyte {
namespace cuda {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CUDA Kernels for Auction Algorithm
// ============================================================================

// Kernel: Find best and second-best columns for each unassigned row
// NOTE: row_active can be null for standard assignment, non-null for tracking
__global__ void kernelAuctionBidding(
    const float* cost_matrix,   // [num_rows, num_cols]
    const float* prices,        // [num_cols]
    const int* row_assignments, // [num_rows] -1 if unassigned
    const int* row_active,      // [num_rows] optional: 1 if active, 0 if inactive (can be null)
    float* row_best_value,      // [num_rows]
    int* row_best_col,          // [num_rows]
    float* row_second_value,    // [num_rows]
    int num_rows,
    int num_cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    // Skip if already assigned
    if (row_assignments[row] >= 0) {
        row_best_col[row] = -1;
        return;
    }

    // Skip if inactive (when row_active is provided)
    if (row_active != nullptr && row_active[row] == 0) {
        row_best_col[row] = -1;
        row_best_value[row] = -1e9f;
        row_second_value[row] = -1e9f;
        return;
    }

    float best_value = -1e9f;
    float second_value = -1e9f;
    int best_col = -1;

    for (int col = 0; col < num_cols; col++) {
        // Value = -cost - price (we want to minimize cost)
        float value = -cost_matrix[row * num_cols + col] - prices[col];

        if (value > best_value) {
            second_value = best_value;
            best_value = value;
            best_col = col;
        } else if (value > second_value) {
            second_value = value;
        }
    }

    row_best_value[row] = best_value;
    row_best_col[row] = best_col;
    row_second_value[row] = second_value;
}

// Kernel: Process bids and update assignments
__global__ void kernelAuctionAssignment(
    const float* row_best_value,
    const int* row_best_col,
    const float* row_second_value,
    float* prices,
    int* row_assignments,
    int* col_assignments,
    int* changed,
    float epsilon,
    int num_rows,
    int num_cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= num_cols) return;

    // Find highest bidder for this column
    float highest_bid = -1e9f;
    int highest_bidder = -1;

    for (int row = 0; row < num_rows; row++) {
        if (row_best_col[row] == col) {
            float bid = row_best_value[row] - row_second_value[row] + epsilon;
            if (bid > highest_bid) {
                highest_bid = bid;
                highest_bidder = row;
            }
        }
    }

    if (highest_bidder >= 0) {
        // Unassign previous owner
        int prev_owner = col_assignments[col];
        if (prev_owner >= 0) {
            row_assignments[prev_owner] = -1;
        }

        // Assign to new bidder
        col_assignments[col] = highest_bidder;
        row_assignments[highest_bidder] = col;

        // Update price
        prices[col] += highest_bid;

        atomicExch(changed, 1);
    }
}

// Kernel: Greedy row-wise assignment
__global__ void kernelGreedyMatch(
    const float* cost_matrix,
    int* row_matched,           // [num_rows] output col index, -1 if none
    int* col_matched,           // [num_cols] 1 if matched, 0 otherwise
    int num_rows,
    int num_cols,
    float threshold
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float best_cost = threshold;
    int best_col = -1;

    for (int col = 0; col < num_cols; col++) {
        float cost = cost_matrix[row * num_cols + col];
        if (cost < best_cost) {
            // Check if column is not yet taken (atomic check)
            if (atomicCAS(&col_matched[col], 0, 1) == 0) {
                // We got this column
                if (best_col >= 0) {
                    // Release previous best
                    atomicExch(&col_matched[best_col], 0);
                }
                best_col = col;
                best_cost = cost;
            }
        }
    }

    row_matched[row] = best_col;
}

// ============================================================================
// LinearAssignmentCUDA Implementation
// ============================================================================

LinearAssignmentCUDA::LinearAssignmentCUDA(int max_size) : max_size_(max_size) {
    CUDA_CHECK(cudaMalloc(&d_cost_matrix_, max_size * max_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prices_, max_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_assignments_, max_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_assignments_, max_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_best_, max_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_best_value_, max_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_second_value_, max_size * sizeof(float)));  // Pre-allocate
    CUDA_CHECK(cudaMalloc(&d_changed_, sizeof(int)));

    h_cost_matrix_ = new float[max_size * max_size];
    h_row_assignments_ = new int[max_size];
    h_col_assignments_ = new int[max_size];

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

LinearAssignmentCUDA::~LinearAssignmentCUDA() {
    cudaFree(d_cost_matrix_);
    cudaFree(d_prices_);
    cudaFree(d_row_assignments_);
    cudaFree(d_col_assignments_);
    cudaFree(d_row_best_);
    cudaFree(d_row_best_value_);
    cudaFree(d_row_second_value_);
    cudaFree(d_changed_);

    delete[] h_cost_matrix_;
    delete[] h_row_assignments_;
    delete[] h_col_assignments_;

    cudaStreamDestroy(stream_);
}

void LinearAssignmentCUDA::greedyAssign(
    const float* cost_matrix,
    int num_rows,
    int num_cols,
    int* row_assignments,
    int* col_assignments,
    float threshold
) {
    // Simple CPU greedy assignment (used for small matrices)
    std::vector<bool> col_used(num_cols, false);

    // Initialize
    for (int i = 0; i < num_rows; i++) row_assignments[i] = -1;
    for (int i = 0; i < num_cols; i++) col_assignments[i] = -1;

    // Greedy: for each row, find best available column
    for (int row = 0; row < num_rows; row++) {
        float best_cost = threshold;
        int best_col = -1;

        for (int col = 0; col < num_cols; col++) {
            if (!col_used[col]) {
                float cost = cost_matrix[row * num_cols + col];
                if (cost < best_cost) {
                    best_cost = cost;
                    best_col = col;
                }
            }
        }

        if (best_col >= 0) {
            row_assignments[row] = best_col;
            col_assignments[best_col] = row;
            col_used[best_col] = true;
        }
    }
}

int LinearAssignmentCUDA::solve(
    const float* cost_matrix,
    int num_rows,
    int num_cols,
    int* row_assignments,
    int* col_assignments,
    float threshold
) {
    if (num_rows == 0 || num_cols == 0) return 0;

    // For small matrices, use CPU greedy (faster due to overhead)
    if (num_rows * num_cols < 100) {
        greedyAssign(cost_matrix, num_rows, num_cols,
                     row_assignments, col_assignments, threshold);

        int count = 0;
        for (int i = 0; i < num_rows; i++) {
            if (row_assignments[i] >= 0) count++;
        }
        return count;
    }

    // Copy cost matrix to device
    CUDA_CHECK(cudaMemcpyAsync(d_cost_matrix_, cost_matrix,
                               num_rows * num_cols * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));

    // Initialize
    CUDA_CHECK(cudaMemsetAsync(d_prices_, 0, num_cols * sizeof(float), stream_));

    std::vector<int> h_row_init(num_rows, -1);
    std::vector<int> h_col_init(num_cols, -1);
    CUDA_CHECK(cudaMemcpyAsync(d_row_assignments_, h_row_init.data(),
                               num_rows * sizeof(int),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_col_assignments_, h_col_init.data(),
                               num_cols * sizeof(int),
                               cudaMemcpyHostToDevice, stream_));

    // Use pre-allocated d_row_second_value_ (no cudaMalloc per frame!)

    // Auction algorithm parameters
    float epsilon = 1.0f / (num_rows + 1);
    int max_iterations = num_rows * 3;

    int block_size = 256;
    int grid_rows = (num_rows + block_size - 1) / block_size;
    int grid_cols = (num_cols + block_size - 1) / block_size;

    // Auction iterations
    for (int iter = 0; iter < max_iterations; iter++) {
        // Reset changed flag
        int h_changed = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_changed_, &h_changed, sizeof(int),
                                   cudaMemcpyHostToDevice, stream_));

        // Bidding phase
        kernelAuctionBidding<<<grid_rows, block_size, 0, stream_>>>(
            d_cost_matrix_, d_prices_, d_row_assignments_,
            nullptr,  // no row_active filter for standard solve
            d_row_best_value_, d_row_best_, d_row_second_value_,
            num_rows, num_cols
        );

        // Assignment phase
        kernelAuctionAssignment<<<grid_cols, block_size, 0, stream_>>>(
            d_row_best_value_, d_row_best_, d_row_second_value_,
            d_prices_, d_row_assignments_, d_col_assignments_,
            d_changed_, epsilon, num_rows, num_cols
        );

        // Check convergence
        CUDA_CHECK(cudaMemcpyAsync(&h_changed, d_changed_, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        if (h_changed == 0) break;

        // Decrease epsilon for better convergence
        epsilon *= 0.9f;
    }

    // Copy results back
    CUDA_CHECK(cudaMemcpy(row_assignments, d_row_assignments_,
                          num_rows * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(col_assignments, d_col_assignments_,
                          num_cols * sizeof(int), cudaMemcpyDeviceToHost));

    // Apply threshold filter and count assignments
    int count = 0;
    for (int row = 0; row < num_rows; row++) {
        int col = row_assignments[row];
        if (col >= 0) {
            if (cost_matrix[row * num_cols + col] <= threshold) {
                count++;
            } else {
                // Invalidate assignment above threshold
                col_assignments[col] = -1;
                row_assignments[row] = -1;
            }
        }
    }

    return count;
}

// GPU-Native async version
void LinearAssignmentCUDA::solveDeviceAsync(
    const float* d_cost_matrix,
    int num_rows,
    int num_cols,
    int* d_row_assignments,
    int* d_col_assignments,
    float threshold,
    cudaStream_t stream
) {
    // Call extended version with no row_active filter
    solveDeviceAsyncWithActive(d_cost_matrix, num_rows, num_cols,
                                d_row_assignments, d_col_assignments,
                                nullptr, threshold, stream);
}

// GPU-Native async version with row activity filter (for tracking)
void LinearAssignmentCUDA::solveDeviceAsyncWithActive(
    const float* d_cost_matrix,
    int num_rows,
    int num_cols,
    int* d_row_assignments,
    int* d_col_assignments,
    const int* d_row_active,    // Optional: skip inactive rows
    float threshold,
    cudaStream_t stream
) {
    if (num_rows == 0 || num_cols == 0) return;

    cudaStream_t s = stream ? stream : stream_;

    // Initialize assignments to -1
    CUDA_CHECK(cudaMemsetAsync(d_row_assignments, 0xFF, num_rows * sizeof(int), s));
    CUDA_CHECK(cudaMemsetAsync(d_col_assignments, 0xFF, num_cols * sizeof(int), s));
    CUDA_CHECK(cudaMemsetAsync(d_prices_, 0, num_cols * sizeof(float), s));

    // Auction algorithm
    float epsilon = 1.0f / (num_rows + 1);
    int max_iterations = std::min(num_rows * 3, 50);  // Cap iterations

    int block_size = 256;
    int grid_rows = (num_rows + block_size - 1) / block_size;
    int grid_cols = (num_cols + block_size - 1) / block_size;

    // Run fixed number of iterations without intermediate syncs
    for (int iter = 0; iter < max_iterations; iter++) {
        CUDA_CHECK(cudaMemsetAsync(d_changed_, 0, sizeof(int), s));

        kernelAuctionBidding<<<grid_rows, block_size, 0, s>>>(
            d_cost_matrix, d_prices_, d_row_assignments,
            d_row_active,  // Pass row_active filter
            d_row_best_value_, d_row_best_, d_row_second_value_,
            num_rows, num_cols
        );

        kernelAuctionAssignment<<<grid_cols, block_size, 0, s>>>(
            d_row_best_value_, d_row_best_, d_row_second_value_,
            d_prices_, d_row_assignments, d_col_assignments,
            d_changed_, epsilon, num_rows, num_cols
        );

        epsilon *= 0.9f;
    }
    // No sync - caller decides when to sync
}

void LinearAssignmentCUDA::sync(cudaStream_t stream) {
    cudaStream_t s = stream ? stream : stream_;
    CUDA_CHECK(cudaStreamSynchronize(s));
}

int LinearAssignmentCUDA::solveDevice(
    float* d_cost_matrix,
    int num_rows,
    int num_cols,
    int* d_row_assignments,
    int* d_col_assignments,
    float threshold
) {
    solveDeviceAsync(d_cost_matrix, num_rows, num_cols,
                     d_row_assignments, d_col_assignments, threshold, stream_);
    sync(stream_);

    // Copy back and count (need host copy for this)
    std::vector<int> h_row(num_rows);
    CUDA_CHECK(cudaMemcpy(h_row.data(), d_row_assignments, num_rows * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int count = 0;
    for (int i = 0; i < num_rows; i++) {
        if (h_row[i] >= 0) count++;
    }
    return count;
}

// ============================================================================
// GreedyMatcherCUDA Implementation
// ============================================================================

GreedyMatcherCUDA::GreedyMatcherCUDA(int max_size) : max_size_(max_size) {
    CUDA_CHECK(cudaMalloc(&d_costs_, max_size * max_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_matched_, max_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_matched_, max_size * sizeof(int)));
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

GreedyMatcherCUDA::~GreedyMatcherCUDA() {
    cudaFree(d_costs_);
    cudaFree(d_row_matched_);
    cudaFree(d_col_matched_);
    cudaStreamDestroy(stream_);
}

std::vector<std::pair<int, int>> GreedyMatcherCUDA::match(
    const float* cost_matrix,
    int num_rows,
    int num_cols,
    float threshold
) {
    std::vector<std::pair<int, int>> matches;
    if (num_rows == 0 || num_cols == 0) return matches;

    // For small matrices, use CPU implementation
    if (num_rows * num_cols < 200) {
        std::vector<bool> col_used(num_cols, false);

        // Create sorted list of all costs
        std::vector<std::tuple<float, int, int>> costs;
        for (int r = 0; r < num_rows; r++) {
            for (int c = 0; c < num_cols; c++) {
                float cost = cost_matrix[r * num_cols + c];
                if (cost < threshold) {
                    costs.push_back({cost, r, c});
                }
            }
        }

        std::sort(costs.begin(), costs.end());

        std::vector<bool> row_used(num_rows, false);

        for (const auto& [cost, row, col] : costs) {
            if (!row_used[row] && !col_used[col]) {
                matches.push_back({row, col});
                row_used[row] = true;
                col_used[col] = true;
            }
        }

        return matches;
    }

    // GPU implementation for larger matrices
    CUDA_CHECK(cudaMemcpyAsync(d_costs_, cost_matrix,
                               num_rows * num_cols * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemsetAsync(d_col_matched_, 0, num_cols * sizeof(int), stream_));

    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;

    kernelGreedyMatch<<<grid_size, block_size, 0, stream_>>>(
        d_costs_, d_row_matched_, d_col_matched_,
        num_rows, num_cols, threshold
    );

    std::vector<int> h_row_matched(num_rows);
    CUDA_CHECK(cudaMemcpy(h_row_matched.data(), d_row_matched_,
                          num_rows * sizeof(int), cudaMemcpyDeviceToHost));

    for (int row = 0; row < num_rows; row++) {
        if (h_row_matched[row] >= 0) {
            matches.push_back({row, h_row_matched[row]});
        }
    }

    return matches;
}

// GPU-Native async version
void GreedyMatcherCUDA::matchDeviceAsync(
    const float* d_cost_matrix,
    int num_rows,
    int num_cols,
    int* d_row_matched,
    float threshold,
    cudaStream_t stream
) {
    if (num_rows == 0 || num_cols == 0) return;

    cudaStream_t s = stream ? stream : stream_;

    CUDA_CHECK(cudaMemsetAsync(d_col_matched_, 0, num_cols * sizeof(int), s));

    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;

    kernelGreedyMatch<<<grid_size, block_size, 0, s>>>(
        d_cost_matrix, d_row_matched, d_col_matched_,
        num_rows, num_cols, threshold
    );
    // No sync
}

}  // namespace cuda
}  // namespace posebyte
