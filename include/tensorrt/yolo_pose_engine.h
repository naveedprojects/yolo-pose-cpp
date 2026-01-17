#pragma once

#include "types.h"
#include "cuda/gpu_postprocess.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

namespace posebyte {
namespace tensorrt {

// Precision modes for TensorRT engine
enum class Precision {
    FP32,
    FP16,
    INT8
};

// TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    void setLogLevel(Severity level) { log_level_ = level; }
private:
    Severity log_level_ = Severity::kWARNING;
};

// INT8 calibrator for quantization
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(
        const std::vector<std::string>& image_paths,
        int batch_size,
        int input_width,
        int input_height,
        const std::string& cache_file
    );
    ~Int8EntropyCalibrator();

    int getBatchSize() const noexcept override { return batch_size_; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    int batch_size_;
    int input_width_;
    int input_height_;
    int current_batch_;
    std::vector<std::string> image_paths_;
    std::string cache_file_;
    std::vector<char> calibration_cache_;
    float* d_input_;
};

// YOLO-Pose TensorRT Engine
class YoloPoseEngine {
public:
    YoloPoseEngine();
    ~YoloPoseEngine();

    // Build engine from ONNX model
    bool buildFromONNX(
        const std::string& onnx_path,
        Precision precision = Precision::FP16,
        int max_batch_size = 1,
        const std::string& calibration_cache = ""
    );

    // Load pre-built engine from file
    bool loadEngine(const std::string& engine_path);

    // Save engine to file
    bool saveEngine(const std::string& engine_path);

    // Run inference
    // Returns detections for a single image
    std::vector<PoseDetection> detect(
        const float* input_data,  // Preprocessed input [3, H, W]
        float conf_threshold = 0.25f,
        float nms_threshold = 0.65f
    );

    // Batch inference
    std::vector<std::vector<PoseDetection>> detectBatch(
        const float* input_data,  // [batch, 3, H, W]
        int batch_size,
        float conf_threshold = 0.25f,
        float nms_threshold = 0.65f
    );

    // Run inference from device memory (zero-copy)
    std::vector<PoseDetection> detectFromDevice(
        float* d_input_data,  // Device pointer [3, H, W]
        float conf_threshold = 0.25f,
        float nms_threshold = 0.65f
    );

    // Get device input buffer for direct GPU preprocessing
    float* getDeviceInputBuffer() { return d_input_; }

    // GPU-native detection: entire pipeline on device
    // Returns number of detections, data stays on GPU
    int detectGPUNative(
        float* d_input_data,          // Device pointer [3, H, W]
        float conf_threshold = 0.25f,
        float nms_threshold = 0.65f
    );

    // Get GPU postprocess outputs (for chaining to tracker)
    cuda::GPUPostprocess* getGPUPostprocess() { return gpu_postprocess_.get(); }
    float* getDeviceOutputBuffer() { return d_output_; }

    // Get input dimensions
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }
    int getMaxBatchSize() const { return max_batch_size_; }

    // Preprocess image (resize, normalize, HWC->CHW)
    void preprocess(
        const unsigned char* image_data,  // BGR HWC format
        int image_width,
        int image_height,
        float* output_data  // CHW normalized output
    );

    // Get inference time (ms)
    float getLastInferenceTime() const { return last_inference_time_; }

private:
    TRTLogger logger_;

    // TensorRT objects
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Input/output dimensions
    int input_width_ = 640;
    int input_height_ = 640;
    int max_batch_size_ = 1;
    int num_detections_ = 8400;  // YOLO default for 640x640
    int detection_size_ = 56;     // 4 (bbox) + 1 (conf) + 51 (17 keypoints * 3)

    // Device memory
    float* d_input_ = nullptr;
    float* d_output_ = nullptr;

    // TensorRT 10: Tensor names instead of binding indices
    std::string input_name_;
    std::string output_name_;

    // CUDA stream
    cudaStream_t stream_;

    // Timing
    float last_inference_time_ = 0.0f;
    cudaEvent_t start_event_, end_event_;

    // GPU-native postprocessing
    std::unique_ptr<cuda::GPUPostprocess> gpu_postprocess_;

    // Post-process raw output
    std::vector<PoseDetection> postprocess(
        const float* output_data,
        int batch_idx,
        float conf_threshold,
        float nms_threshold,
        float scale_x,
        float scale_y
    );

    // Allocate device memory
    bool allocateBuffers();
    void freeBuffers();
};

}  // namespace tensorrt
}  // namespace posebyte
