#include "cuda/preprocess.h"
#include <cstdio>
#include <algorithm>

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

// Combined kernel: resize with letterbox, BGR->RGB, normalize, HWC->CHW
// Uses bilinear interpolation
__global__ void kernelPreprocess(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int input_width,
    int input_height,
    int target_width,
    int target_height,
    int new_width,
    int new_height,
    int pad_x,
    int pad_y,
    float scale
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= target_width || ty >= target_height) return;

    const int target_size = target_width * target_height;

    // Check if in padding area
    if (tx < pad_x || tx >= pad_x + new_width ||
        ty < pad_y || ty >= pad_y + new_height) {
        // Gray padding (114/255 = 0.447)
        float gray = 114.0f / 255.0f;
        output[0 * target_size + ty * target_width + tx] = gray;  // R
        output[1 * target_size + ty * target_width + tx] = gray;  // G
        output[2 * target_size + ty * target_width + tx] = gray;  // B
        return;
    }

    // Map to source coordinates (bilinear interpolation)
    float src_x = (tx - pad_x) / scale;
    float src_y = (ty - pad_y) / scale;

    // Clamp to valid range
    src_x = fminf(fmaxf(src_x, 0.0f), input_width - 1.001f);
    src_y = fminf(fmaxf(src_y, 0.0f), input_height - 1.001f);

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, input_width - 1);
    int y1 = min(y0 + 1, input_height - 1);

    float wx = src_x - x0;
    float wy = src_y - y0;

    // Bilinear interpolation for each channel
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        float v00 = input[(y0 * input_width + x0) * 3 + c];
        float v01 = input[(y0 * input_width + x1) * 3 + c];
        float v10 = input[(y1 * input_width + x0) * 3 + c];
        float v11 = input[(y1 * input_width + x1) * 3 + c];

        float v = (1 - wx) * (1 - wy) * v00 +
                  wx * (1 - wy) * v01 +
                  (1 - wx) * wy * v10 +
                  wx * wy * v11;

        // Normalize to [0,1] and convert BGR->RGB (swap channel 0 and 2)
        int out_c = (c == 0) ? 2 : (c == 2) ? 0 : c;
        output[out_c * target_size + ty * target_width + tx] = v / 255.0f;
    }
}

PreprocessorCUDA::PreprocessorCUDA(int max_input_width, int max_input_height,
                                     int target_width, int target_height)
    : max_input_width_(max_input_width)
    , max_input_height_(max_input_height)
    , target_width_(target_width)
    , target_height_(target_height)
{
    // Allocate device memory
    size_t input_size = max_input_width * max_input_height * 3 * sizeof(uint8_t);
    size_t output_size = 3 * target_width * target_height * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input_, input_size));
    CUDA_CHECK(cudaMalloc(&d_output_, output_size));
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

PreprocessorCUDA::~PreprocessorCUDA() {
    cudaFree(d_input_);
    cudaFree(d_output_);
    cudaStreamDestroy(stream_);
}

void PreprocessorCUDA::preprocess(
    const uint8_t* input_bgr,
    int input_width,
    int input_height,
    float* output_tensor,
    float& scale_x,
    float& scale_y,
    int& pad_x,
    int& pad_y
) {
    // Calculate scale to maintain aspect ratio
    float scale = std::min(
        static_cast<float>(target_width_) / input_width,
        static_cast<float>(target_height_) / input_height
    );

    int new_width = static_cast<int>(input_width * scale);
    int new_height = static_cast<int>(input_height * scale);

    pad_x = (target_width_ - new_width) / 2;
    pad_y = (target_height_ - new_height) / 2;

    // Store inverse scale for detection scaling
    scale_x = 1.0f / scale;
    scale_y = 1.0f / scale;

    // Copy input to device
    size_t input_size = input_width * input_height * 3 * sizeof(uint8_t);
    CUDA_CHECK(cudaMemcpyAsync(d_input_, input_bgr, input_size,
                                cudaMemcpyHostToDevice, stream_));

    // Launch preprocessing kernel
    dim3 block(16, 16);
    dim3 grid((target_width_ + block.x - 1) / block.x,
              (target_height_ + block.y - 1) / block.y);

    kernelPreprocess<<<grid, block, 0, stream_>>>(
        d_input_, output_tensor,
        input_width, input_height,
        target_width_, target_height_,
        new_width, new_height,
        pad_x, pad_y,
        scale
    );

    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace cuda
}  // namespace posebyte
