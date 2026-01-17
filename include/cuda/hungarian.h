#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace posebyte {
namespace cuda {

// Linear Assignment Problem Solver using Auction Algorithm
// GPU-friendly alternative to Hungarian algorithm
class LinearAssignmentCUDA {
public:
    LinearAssignmentCUDA(int max_size = 256);
    ~LinearAssignmentCUDA();

    // ========================================================================
    // GPU-Native Operations (async, for pipeline integration)
    // ========================================================================

    // Solve from device cost matrix - async (call sync() after)
    // d_cost_matrix: [num_rows, num_cols] on device
    // d_row_assignments: [num_rows] output on device
    // d_col_assignments: [num_cols] output on device
    void solveDeviceAsync(
        const float* d_cost_matrix,
        int num_rows,
        int num_cols,
        int* d_row_assignments,
        int* d_col_assignments,
        float threshold,
        cudaStream_t stream = 0
    );

    // Solve with row activity filter - for tracking use
    // d_row_active: optional filter, 1=active, 0=skip (can be nullptr)
    void solveDeviceAsyncWithActive(
        const float* d_cost_matrix,
        int num_rows,
        int num_cols,
        int* d_row_assignments,
        int* d_col_assignments,
        const int* d_row_active,
        float threshold,
        cudaStream_t stream = 0
    );

    // Sync stream
    void sync(cudaStream_t stream = 0);

    // ========================================================================
    // Legacy Operations (for compatibility)
    // ========================================================================

    // Solve assignment problem
    int solve(
        const float* cost_matrix,
        int num_rows,
        int num_cols,
        int* row_assignments,
        int* col_assignments,
        float threshold = 1.0f
    );

    // Solve with device memory (sync)
    int solveDevice(
        float* d_cost_matrix,
        int num_rows,
        int num_cols,
        int* d_row_assignments,
        int* d_col_assignments,
        float threshold = 1.0f
    );

    // ========================================================================
    // Device Memory Access
    // ========================================================================

    float* getCostMatrixDevice() { return d_cost_matrix_; }
    int* getRowAssignmentsDevice() { return d_row_assignments_; }
    int* getColAssignmentsDevice() { return d_col_assignments_; }
    float* getPricesDevice() { return d_prices_; }
    int getMaxSize() const { return max_size_; }

private:
    int max_size_;

    // Device memory
    float* d_cost_matrix_;
    float* d_prices_;
    int* d_row_assignments_;
    int* d_col_assignments_;
    int* d_row_best_;
    float* d_row_best_value_;
    float* d_row_second_value_;
    int* d_changed_;

    // Host buffers (pinned memory)
    float* h_cost_matrix_;
    int* h_row_assignments_;
    int* h_col_assignments_;

    cudaStream_t stream_;

    // Greedy assignment (CPU fallback)
    void greedyAssign(
        const float* cost_matrix,
        int num_rows,
        int num_cols,
        int* row_assignments,
        int* col_assignments,
        float threshold
    );
};

// Simpler greedy matcher for speed
class GreedyMatcherCUDA {
public:
    GreedyMatcherCUDA(int max_size = 256);
    ~GreedyMatcherCUDA();

    // ========================================================================
    // GPU-Native Operations
    // ========================================================================

    // Match from device cost matrix - async
    void matchDeviceAsync(
        const float* d_cost_matrix,
        int num_rows,
        int num_cols,
        int* d_row_matched,
        float threshold,
        cudaStream_t stream = 0
    );

    // ========================================================================
    // Legacy Operations
    // ========================================================================

    std::vector<std::pair<int, int>> match(
        const float* cost_matrix,
        int num_rows,
        int num_cols,
        float threshold = 0.8f
    );

    // ========================================================================
    // Device Memory Access
    // ========================================================================

    float* getCostsDevice() { return d_costs_; }
    int* getRowMatchedDevice() { return d_row_matched_; }
    int* getColMatchedDevice() { return d_col_matched_; }

private:
    int max_size_;
    float* d_costs_;
    int* d_row_matched_;
    int* d_col_matched_;
    cudaStream_t stream_;
};

}  // namespace cuda
}  // namespace posebyte
