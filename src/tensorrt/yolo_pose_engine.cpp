#include "tensorrt/yolo_pose_engine.h"
#include "cuda/nms.h"
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <filesystem>

namespace posebyte {
namespace tensorrt {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

// ============================================================================
// TRTLogger Implementation
// ============================================================================

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity > log_level_) return;

    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "[TRT INTERNAL ERROR] " << msg << std::endl;
            break;
        case Severity::kERROR:
            std::cerr << "[TRT ERROR] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cerr << "[TRT WARNING] " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "[TRT INFO] " << msg << std::endl;
            break;
        case Severity::kVERBOSE:
            std::cout << "[TRT VERBOSE] " << msg << std::endl;
            break;
    }
}

// ============================================================================
// Int8EntropyCalibrator Implementation
// ============================================================================

Int8EntropyCalibrator::Int8EntropyCalibrator(
    const std::vector<std::string>& image_paths,
    int batch_size,
    int input_width,
    int input_height,
    const std::string& cache_file
) : batch_size_(batch_size),
    input_width_(input_width),
    input_height_(input_height),
    current_batch_(0),
    image_paths_(image_paths),
    cache_file_(cache_file),
    d_input_(nullptr) {

    size_t input_size = batch_size * 3 * input_height * input_width * sizeof(float);
    cudaMalloc(&d_input_, input_size);
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    if (d_input_) cudaFree(d_input_);
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (current_batch_ * batch_size_ >= static_cast<int>(image_paths_.size())) {
        return false;
    }

    std::vector<float> batch_data(batch_size_ * 3 * input_height_ * input_width_, 114.0f / 255.0f);

    for (int i = 0; i < batch_size_; i++) {
        int img_idx = current_batch_ * batch_size_ + i;
        if (img_idx >= static_cast<int>(image_paths_.size())) break;

        // Load image using OpenCV
        cv::Mat img = cv::imread(image_paths_[img_idx]);
        if (img.empty()) {
            std::cerr << "Failed to load calibration image: " << image_paths_[img_idx] << std::endl;
            continue;
        }

        // Calculate scale to maintain aspect ratio
        float scale = std::min(
            static_cast<float>(input_width_) / img.cols,
            static_cast<float>(input_height_) / img.rows
        );

        int new_width = static_cast<int>(img.cols * scale);
        int new_height = static_cast<int>(img.rows * scale);
        int pad_x = (input_width_ - new_width) / 2;
        int pad_y = (input_height_ - new_height) / 2;

        // Resize image
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_width, new_height));

        // Create padded image
        cv::Mat padded(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_width, new_height)));

        // Convert BGR to RGB and normalize
        cv::Mat rgb;
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

        // Copy to batch buffer (CHW format)
        float* batch_ptr = batch_data.data() + i * 3 * input_height_ * input_width_;
        for (int y = 0; y < input_height_; y++) {
            for (int x = 0; x < input_width_; x++) {
                cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
                batch_ptr[0 * input_height_ * input_width_ + y * input_width_ + x] = pixel[0] / 255.0f;
                batch_ptr[1 * input_height_ * input_width_ + y * input_width_ + x] = pixel[1] / 255.0f;
                batch_ptr[2 * input_height_ * input_width_ + y * input_width_ + x] = pixel[2] / 255.0f;
            }
        }
    }

    cudaMemcpy(d_input_, batch_data.data(),
               batch_data.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    bindings[0] = d_input_;
    current_batch_++;

    std::cout << "Calibration batch " << current_batch_ << "/"
              << (image_paths_.size() + batch_size_ - 1) / batch_size_ << std::endl;

    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
    calibration_cache_.clear();
    std::ifstream input(cache_file_, std::ios::binary);

    if (input.good()) {
        input.seekg(0, std::ios::end);
        length = input.tellg();
        input.seekg(0, std::ios::beg);
        calibration_cache_.resize(length);
        input.read(calibration_cache_.data(), length);
        return calibration_cache_.data();
    }

    length = 0;
    return nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::ofstream output(cache_file_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

// ============================================================================
// YoloPoseEngine Implementation - TensorRT 10 API
// ============================================================================

YoloPoseEngine::YoloPoseEngine() {
    cudaStreamCreate(&stream_);
    cudaEventCreate(&start_event_);
    cudaEventCreate(&end_event_);
}

YoloPoseEngine::~YoloPoseEngine() {
    freeBuffers();
    cudaStreamDestroy(stream_);
    cudaEventDestroy(start_event_);
    cudaEventDestroy(end_event_);
}

bool YoloPoseEngine::buildFromONNX(
    const std::string& onnx_path,
    Precision precision,
    int max_batch_size,
    const std::string& calibration_cache
) {
    max_batch_size_ = max_batch_size;

    // Create builder - TensorRT 10 API
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return false;
    }

    // Create network with explicit batch - TensorRT 10: flag is default now
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0));  // TRT 10: 0 means explicit batch (default)
    if (!network) {
        std::cerr << "Failed to create network" << std::endl;
        return false;
    }

    // Create ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return false;
    }

    // Parse ONNX model
    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model: " << onnx_path << std::endl;
        for (int i = 0; i < parser->getNbErrors(); i++) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    std::cout << "Successfully parsed ONNX model" << std::endl;

    // Create builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return false;
    }

    // Set precision
    if (precision == Precision::FP16) {
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout << "Using FP16 precision" << std::endl;
        } else {
            std::cerr << "FP16 not supported, using FP32" << std::endl;
        }
    } else if (precision == Precision::INT8) {
        if (builder->platformHasFastInt8()) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            // Also enable FP16 as fallback for layers that don't support INT8
            if (builder->platformHasFastFp16()) {
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }

            if (!calibration_cache.empty()) {
                // Check if calibration_cache is a directory (for images) or file (for cache)
                std::vector<std::string> calib_images;
                std::string cache_file = calibration_cache;

                if (std::filesystem::is_directory(calibration_cache)) {
                    // Load all images from directory
                    std::cout << "Loading calibration images from: " << calibration_cache << std::endl;
                    for (const auto& entry : std::filesystem::directory_iterator(calibration_cache)) {
                        std::string ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                            calib_images.push_back(entry.path().string());
                        }
                    }
                    std::sort(calib_images.begin(), calib_images.end());
                    cache_file = calibration_cache + "/calibration.cache";
                    std::cout << "Found " << calib_images.size() << " calibration images" << std::endl;
                }

                if (!calib_images.empty()) {
                    auto calibrator = new Int8EntropyCalibrator(
                        calib_images, 1, input_width_, input_height_, cache_file);
                    config->setInt8Calibrator(calibrator);
                } else {
                    std::cerr << "Warning: No calibration images found, INT8 may fail" << std::endl;
                }
            }

            std::cout << "Using INT8 precision with partial quantization" << std::endl;

            // Apply partial quantization: mark entire blocks (all layer types) as FP16
            // This handles fused operations (Conv+SiLU) that lack INT8 implementations
            int fp16_layers = 0;
            int skipped_layers = 0;
            for (int i = 0; i < network->getNbLayers(); i++) {
                auto layer = network->getLayer(i);
                std::string layer_name = layer->getName();
                auto layer_type = layer->getType();

                // Skip Constant layers - they can't have precision set
                if (layer_type == nvinfer1::LayerType::kCONSTANT) {
                    continue;
                }

                bool force_fp16 = false;

                // Mark all layers in the early backbone as FP16 (not just convolutions)
                // This ensures fused operations (Conv+Activation) are all in the same precision
                // Model structure: 0=Conv, 1=Conv, 2=C2f, 3=Conv, 4=C2f, 5=Conv, 6=C2f, 7=Conv, 8=C2f
                // The C2f blocks (2, 4, 6, 8) often have INT8 implementation issues
                if (layer_name.find("/model.0/") != std::string::npos ||
                    layer_name.find("/model.1/") != std::string::npos ||
                    layer_name.find("/model.2/") != std::string::npos ||
                    layer_name.find("/model.3/") != std::string::npos ||
                    layer_name.find("/model.4/") != std::string::npos) {
                    force_fp16 = true;
                }

                if (force_fp16) {
                    // Check if this layer type supports precision setting
                    // Avoid setting precision on layers that use integer-typed weights
                    if (layer_type == nvinfer1::LayerType::kGATHER ||
                        layer_type == nvinfer1::LayerType::kSHAPE ||
                        layer_type == nvinfer1::LayerType::kSLICE ||
                        layer_type == nvinfer1::LayerType::kSHUFFLE) {
                        skipped_layers++;
                        continue;
                    }
                    layer->setPrecision(nvinfer1::DataType::kHALF);
                    // Also set output type to ensure precision consistency in fused ops
                    for (int j = 0; j < layer->getNbOutputs(); j++) {
                        layer->setOutputType(j, nvinfer1::DataType::kHALF);
                    }
                    fp16_layers++;
                }
            }

            if (fp16_layers > 0) {
                std::cout << "Partial quantization: " << fp16_layers
                          << " layers forced to FP16, " << skipped_layers << " skipped" << std::endl;
            }
        } else {
            std::cerr << "INT8 not supported, using FP32" << std::endl;
        }
    }

    // Set memory pool limit - TensorRT 10 API
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // Set optimization profile for dynamic batch
    auto profile = builder->createOptimizationProfile();

    // Get input tensor
    auto input = network->getInput(0);
    auto input_dims = input->getDimensions();
    input_name_ = input->getName();

    // Store dimensions (handle dynamic shapes which are -1)
    if (input_dims.nbDims == 4) {
        input_height_ = input_dims.d[2] > 0 ? input_dims.d[2] : 640;
        input_width_ = input_dims.d[3] > 0 ? input_dims.d[3] : 640;
    }
    std::cout << "Input dimensions: " << input_width_ << "x" << input_height_ << std::endl;

    // Get output tensor name
    auto output = network->getOutput(0);
    output_name_ = output->getName();

    // Set min/opt/max shapes
    nvinfer1::Dims4 min_dims(1, 3, input_height_, input_width_);
    nvinfer1::Dims4 opt_dims(max_batch_size, 3, input_height_, input_width_);
    nvinfer1::Dims4 max_dims(max_batch_size, 3, input_height_, input_width_);

    profile->setDimensions(input_name_.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input_name_.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input_name_.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);

    config->addOptimizationProfile(profile);

    // Build engine - TensorRT 10: buildSerializedNetwork
    std::cout << "Building TensorRT engine (this may take a while)..." << std::endl;

    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        std::cerr << "Failed to build serialized engine" << std::endl;
        return false;
    }

    // Create runtime and deserialize
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(
        serialized_engine->data(), serialized_engine->size()));

    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // TensorRT 10: Get tensor info using new API
    // YOLO-Pose output is [batch, features, detections] = [1, 56, 8400]
    auto output_dims = engine_->getTensorShape(output_name_.c_str());
    if (output_dims.nbDims >= 3) {
        detection_size_ = output_dims.d[1];  // 56 (features per detection)
        num_detections_ = output_dims.d[2];  // 8400 (number of anchor boxes)
    }

    std::cout << "Engine built successfully!" << std::endl;
    std::cout << "  Input: " << input_name_ << " [" << input_width_ << "x" << input_height_ << "]" << std::endl;
    std::cout << "  Output: " << output_name_ << " [" << detection_size_ << " x " << num_detections_ << "]" << std::endl;

    return allocateBuffers();
}

bool YoloPoseEngine::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));

    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // TensorRT 10: Use getNbIOTensors() instead of getNbBindings()
    int num_io_tensors = engine_->getNbIOTensors();

    for (int i = 0; i < num_io_tensors; i++) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            auto dims = engine_->getTensorShape(name);
            if (dims.nbDims == 4) {
                max_batch_size_ = dims.d[0] > 0 ? dims.d[0] : 1;
                input_height_ = dims.d[2];
                input_width_ = dims.d[3];
            }
        } else {
            output_name_ = name;
            auto dims = engine_->getTensorShape(name);
            // YOLO-Pose output is [batch, features, detections] = [1, 56, 8400]
            if (dims.nbDims >= 3) {
                detection_size_ = dims.d[1];  // 56 (features per detection)
                num_detections_ = dims.d[2];  // 8400 (number of anchor boxes)
            }
        }
    }

    std::cout << "Engine loaded successfully! (TensorRT 10)" << std::endl;
    std::cout << "  Input: " << input_name_ << " [" << input_width_ << "x" << input_height_ << "]" << std::endl;
    std::cout << "  Max batch: " << max_batch_size_ << std::endl;
    std::cout << "  Output: " << output_name_ << " [" << detection_size_ << " x " << num_detections_ << "]" << std::endl;

    return allocateBuffers();
}

bool YoloPoseEngine::saveEngine(const std::string& engine_path) {
    if (!engine_) {
        std::cerr << "No engine to save" << std::endl;
        return false;
    }

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        engine_->serialize());
    if (!serialized) {
        std::cerr << "Failed to serialize engine" << std::endl;
        return false;
    }

    std::ofstream file(engine_path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    file.close();

    std::cout << "Engine saved to: " << engine_path << std::endl;
    return true;
}

bool YoloPoseEngine::allocateBuffers() {
    size_t input_size = max_batch_size_ * 3 * input_height_ * input_width_ * sizeof(float);
    size_t output_size = max_batch_size_ * num_detections_ * detection_size_ * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input_, input_size));
    CUDA_CHECK(cudaMalloc(&d_output_, output_size));

    // Initialize GPU-native postprocessing
    gpu_postprocess_ = std::make_unique<cuda::GPUPostprocess>(1024, num_detections_);

    return true;
}

void YoloPoseEngine::freeBuffers() {
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
}

void YoloPoseEngine::preprocess(
    const unsigned char* image_data,
    int image_width,
    int image_height,
    float* output_data
) {
    float scale = std::min(
        static_cast<float>(input_width_) / image_width,
        static_cast<float>(input_height_) / image_height
    );

    int new_width = static_cast<int>(image_width * scale);
    int new_height = static_cast<int>(image_height * scale);

    int pad_x = (input_width_ - new_width) / 2;
    int pad_y = (input_height_ - new_height) / 2;

    float pad_value = 114.0f / 255.0f;
    for (int i = 0; i < 3 * input_height_ * input_width_; i++) {
        output_data[i] = pad_value;
    }

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            float src_x = x / scale;
            float src_y = y / scale;

            int x0 = std::min(static_cast<int>(src_x), image_width - 1);
            int y0 = std::min(static_cast<int>(src_y), image_height - 1);

            int src_idx = (y0 * image_width + x0) * 3;
            int out_x = pad_x + x;
            int out_y = pad_y + y;

            output_data[0 * input_height_ * input_width_ + out_y * input_width_ + out_x] =
                image_data[src_idx + 2] / 255.0f;
            output_data[1 * input_height_ * input_width_ + out_y * input_width_ + out_x] =
                image_data[src_idx + 1] / 255.0f;
            output_data[2 * input_height_ * input_width_ + out_y * input_width_ + out_x] =
                image_data[src_idx + 0] / 255.0f;
        }
    }
}

std::vector<PoseDetection> YoloPoseEngine::detect(
    const float* input_data,
    float conf_threshold,
    float nms_threshold
) {
    auto results = detectBatch(input_data, 1, conf_threshold, nms_threshold);
    return results.empty() ? std::vector<PoseDetection>() : results[0];
}

std::vector<PoseDetection> YoloPoseEngine::detectFromDevice(
    float* d_input_data,
    float conf_threshold,
    float nms_threshold
) {
    std::vector<PoseDetection> results;

    if (!context_) {
        std::cerr << "Engine not initialized" << std::endl;
        return results;
    }

    // TensorRT 10: Use setInputShape
    nvinfer1::Dims4 input_dims(1, 3, input_height_, input_width_);
    context_->setInputShape(input_name_.c_str(), input_dims);

    // TensorRT 10: Use setTensorAddress with device pointer directly
    context_->setTensorAddress(input_name_.c_str(), d_input_data);
    context_->setTensorAddress(output_name_.c_str(), d_output_);

    // Run inference with timing
    cudaEventRecord(start_event_, stream_);
    bool success = context_->enqueueV3(stream_);
    cudaEventRecord(end_event_, stream_);
    cudaStreamSynchronize(stream_);

    cudaEventElapsedTime(&last_inference_time_, start_event_, end_event_);

    if (!success) {
        std::cerr << "Inference failed" << std::endl;
        return results;
    }

    // Copy output to host
    size_t output_size = num_detections_ * detection_size_ * sizeof(float);
    std::vector<float> output_data(num_detections_ * detection_size_);
    cudaMemcpy(output_data.data(), d_output_, output_size, cudaMemcpyDeviceToHost);

    // Post-process
    return postprocess(output_data.data(), 0, conf_threshold, nms_threshold, 1.0f, 1.0f);
}

int YoloPoseEngine::detectGPUNative(
    float* d_input_data,
    float conf_threshold,
    float nms_threshold
) {
    if (!context_ || !gpu_postprocess_) {
        std::cerr << "Engine not initialized" << std::endl;
        return 0;
    }

    // TensorRT 10: Use setInputShape
    nvinfer1::Dims4 input_dims(1, 3, input_height_, input_width_);
    context_->setInputShape(input_name_.c_str(), input_dims);

    // Set tensor addresses - output stays on device
    context_->setTensorAddress(input_name_.c_str(), d_input_data);
    context_->setTensorAddress(output_name_.c_str(), d_output_);

    // Run inference with timing
    cudaEventRecord(start_event_, stream_);
    bool success = context_->enqueueV3(stream_);
    cudaEventRecord(end_event_, stream_);

    if (!success) {
        std::cerr << "Inference failed" << std::endl;
        return 0;
    }

    // GPU-native postprocess: decode + NMS entirely on device
    // No D2H copy! Output goes directly to GPU postprocess buffers
    int num_dets = gpu_postprocess_->process(d_output_, conf_threshold, nms_threshold, stream_);

    // Get inference time (this sync is for timing only)
    cudaEventElapsedTime(&last_inference_time_, start_event_, end_event_);

    return num_dets;
}

std::vector<std::vector<PoseDetection>> YoloPoseEngine::detectBatch(
    const float* input_data,
    int batch_size,
    float conf_threshold,
    float nms_threshold
) {
    std::vector<std::vector<PoseDetection>> results(batch_size);

    if (!context_) {
        std::cerr << "Engine not initialized" << std::endl;
        return results;
    }

    batch_size = std::min(batch_size, max_batch_size_);

    // TensorRT 10: Use setInputShape instead of setBindingDimensions
    nvinfer1::Dims4 input_dims(batch_size, 3, input_height_, input_width_);
    context_->setInputShape(input_name_.c_str(), input_dims);

    // Copy input to device
    size_t input_size = batch_size * 3 * input_height_ * input_width_ * sizeof(float);
    cudaMemcpyAsync(d_input_, input_data, input_size, cudaMemcpyHostToDevice, stream_);

    // TensorRT 10: Use setTensorAddress instead of bindings array
    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);

    // Run inference with timing
    cudaEventRecord(start_event_, stream_);

    // TensorRT 10: Use enqueueV3 instead of enqueueV2
    bool success = context_->enqueueV3(stream_);

    cudaEventRecord(end_event_, stream_);
    cudaStreamSynchronize(stream_);

    cudaEventElapsedTime(&last_inference_time_, start_event_, end_event_);

    if (!success) {
        std::cerr << "Inference failed" << std::endl;
        return results;
    }

    // Copy output to host
    // Output shape is [batch, detection_size_, num_detections_] = [1, 56, 8400]
    size_t output_size = batch_size * num_detections_ * detection_size_ * sizeof(float);
    std::vector<float> output_data(batch_size * num_detections_ * detection_size_);
    cudaMemcpy(output_data.data(), d_output_, output_size, cudaMemcpyDeviceToHost);

    // Post-process each batch
    for (int b = 0; b < batch_size; b++) {
        results[b] = postprocess(output_data.data(), b, conf_threshold, nms_threshold, 1.0f, 1.0f);
    }

    return results;
}

std::vector<PoseDetection> YoloPoseEngine::postprocess(
    const float* output_data,
    int batch_idx,
    float conf_threshold,
    float nms_threshold,
    float scale_x,
    float scale_y
) {
    std::vector<PoseDetection> detections;

    const float* batch_output = output_data + batch_idx * num_detections_ * detection_size_;

    // YOLOv8-pose output is [batch, 56, 8400] - always transposed format
    // Channel layout: [cx, cy, w, h, conf, kp0_x, kp0_y, kp0_vis, ...]
    bool transposed = true;

    for (int i = 0; i < num_detections_; i++) {
        float conf, cx, cy, w, h;

        if (transposed) {
            cx = batch_output[0 * num_detections_ + i];
            cy = batch_output[1 * num_detections_ + i];
            w = batch_output[2 * num_detections_ + i];
            h = batch_output[3 * num_detections_ + i];
            conf = batch_output[4 * num_detections_ + i];
        } else {
            const float* det = batch_output + i * detection_size_;
            cx = det[0];
            cy = det[1];
            w = det[2];
            h = det[3];
            conf = det[4];
        }

        if (conf < conf_threshold) continue;

        PoseDetection det;
        det.score = conf;

        det.bbox[0] = (cx - w / 2) * scale_x;
        det.bbox[1] = (cy - h / 2) * scale_y;
        det.bbox[2] = (cx + w / 2) * scale_x;
        det.bbox[3] = (cy + h / 2) * scale_y;

        for (int kp = 0; kp < NUM_KEYPOINTS; kp++) {
            if (transposed) {
                det.keypoints[kp].x = batch_output[(5 + kp * 3 + 0) * num_detections_ + i] * scale_x;
                det.keypoints[kp].y = batch_output[(5 + kp * 3 + 1) * num_detections_ + i] * scale_y;
                det.keypoints[kp].confidence = batch_output[(5 + kp * 3 + 2) * num_detections_ + i];
            } else {
                const float* det_data = batch_output + i * detection_size_;
                det.keypoints[kp].x = det_data[5 + kp * 3 + 0] * scale_x;
                det.keypoints[kp].y = det_data[5 + kp * 3 + 1] * scale_y;
                det.keypoints[kp].confidence = det_data[5 + kp * 3 + 2];
            }
        }

        detections.push_back(det);
    }

    if (!detections.empty()) {
        cuda::NMSCuda nms(static_cast<int>(detections.size()));
        auto keep_indices = nms.apply(detections.data(), static_cast<int>(detections.size()),
                                      nms_threshold, conf_threshold);

        std::vector<PoseDetection> nms_detections;
        nms_detections.reserve(keep_indices.size());
        for (int idx : keep_indices) {
            nms_detections.push_back(detections[idx]);
        }
        return nms_detections;
    }

    return detections;
}

}  // namespace tensorrt
}  // namespace posebyte
