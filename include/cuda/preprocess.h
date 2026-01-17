#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace posebyte {
namespace cuda {

class PreprocessorCUDA {
public:
    PreprocessorCUDA(int max_input_width, int max_input_height, int target_width, int target_height);
    ~PreprocessorCUDA();

    // Preprocess frame: resize with letterbox, BGR->RGB, normalize, HWC->CHW
    // Input: BGR image data (uint8_t* on host or device)
    // Output: CHW float tensor [3, target_height, target_width] on device
    void preprocess(
        const uint8_t* input_bgr,    // Input BGR image (host memory)
        int input_width,
        int input_height,
        float* output_tensor,         // Output tensor (device memory)
        float& scale_x,
        float& scale_y,
        int& pad_x,
        int& pad_y
    );

    // Get device output buffer
    float* getDeviceOutput() { return d_output_; }

private:
    int max_input_width_;
    int max_input_height_;
    int target_width_;
    int target_height_;

    // Device buffers
    uint8_t* d_input_;
    float* d_output_;

    cudaStream_t stream_;
};

}  // namespace cuda
}  // namespace posebyte
