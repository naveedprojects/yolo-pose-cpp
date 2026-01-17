#include "tensorrt/yolo_pose_engine.h"
#include <iostream>
#include <string>
#include <getopt.h>

using namespace posebyte;

void printUsage(const char* program) {
    std::cout << "YOLO-Pose ONNX to TensorRT Engine Exporter\n";
    std::cout << "Usage: " << program << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  -m, --model PATH     ONNX model file (required)\n";
    std::cout << "  -o, --output PATH    Output engine file (required)\n";
    std::cout << "  -p, --precision STR  Precision: fp32, fp16, int8 (default: fp16)\n";
    std::cout << "  -b, --batch INT      Max batch size (default: 1)\n";
    std::cout << "  -c, --calib PATH     INT8 calibration cache file\n";
    std::cout << "  -h, --help           Show this help message\n";
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string output_path;
    std::string precision_str = "fp16";
    int max_batch = 1;
    std::string calib_cache;

    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"output", required_argument, 0, 'o'},
        {"precision", required_argument, 0, 'p'},
        {"batch", required_argument, 0, 'b'},
        {"calib", required_argument, 0, 'c'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:o:p:b:c:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 'p': precision_str = optarg; break;
            case 'b': max_batch = std::stoi(optarg); break;
            case 'c': calib_cache = optarg; break;
            case 'h':
            default:
                printUsage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }

    if (model_path.empty() || output_path.empty()) {
        std::cerr << "Error: Model and output paths are required\n";
        printUsage(argv[0]);
        return 1;
    }

    // Parse precision
    tensorrt::Precision precision;
    if (precision_str == "fp32" || precision_str == "FP32") {
        precision = tensorrt::Precision::FP32;
    } else if (precision_str == "fp16" || precision_str == "FP16") {
        precision = tensorrt::Precision::FP16;
    } else if (precision_str == "int8" || precision_str == "INT8") {
        precision = tensorrt::Precision::INT8;
        if (calib_cache.empty()) {
            std::cerr << "Warning: INT8 precision requires calibration cache for best results\n";
        }
    } else {
        std::cerr << "Unknown precision: " << precision_str << std::endl;
        return 1;
    }

    std::cout << "=== YOLO-Pose Engine Export ===" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Output: " << output_path << std::endl;
    std::cout << "Precision: " << precision_str << std::endl;
    std::cout << "Max batch: " << max_batch << std::endl;

    tensorrt::YoloPoseEngine engine;

    std::cout << "\nBuilding engine..." << std::endl;
    if (!engine.buildFromONNX(model_path, precision, max_batch, calib_cache)) {
        std::cerr << "Failed to build engine from ONNX" << std::endl;
        return 1;
    }

    std::cout << "Saving engine..." << std::endl;
    if (!engine.saveEngine(output_path)) {
        std::cerr << "Failed to save engine" << std::endl;
        return 1;
    }

    std::cout << "\n=== Export Complete ===" << std::endl;
    std::cout << "Engine saved to: " << output_path << std::endl;

    return 0;
}
