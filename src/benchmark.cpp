#include "tensorrt/yolo_pose_engine.h"
#include "cuda/gpu_tracker.h"
#include "cuda/kalman_filter.h"
#include "cuda/oks_distance.h"
#include "cuda/hungarian.h"
#include "cuda/nms.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <string>
#include <getopt.h>

using namespace posebyte;

// Generate random pose detection for testing
PoseDetection generateRandomPose(std::mt19937& rng, int width, int height) {
    std::uniform_real_distribution<float> pos_x(0.0f, static_cast<float>(width));
    std::uniform_real_distribution<float> pos_y(0.0f, static_cast<float>(height));
    std::uniform_real_distribution<float> conf(0.3f, 1.0f);

    PoseDetection pose;

    // Generate a rough skeleton shape
    float cx = pos_x(rng);
    float cy = pos_y(rng);
    float scale = 50.0f + pos_y(rng) / height * 100.0f;

    // Keypoint offsets from center (rough human shape)
    float offsets[NUM_KEYPOINTS][2] = {
        {0, -1.5f},   // nose
        {-0.1f, -1.6f}, {0.1f, -1.6f},  // eyes
        {-0.2f, -1.5f}, {0.2f, -1.5f},  // ears
        {-0.5f, -1.0f}, {0.5f, -1.0f},  // shoulders
        {-0.8f, -0.3f}, {0.8f, -0.3f},  // elbows
        {-1.0f, 0.3f}, {1.0f, 0.3f},    // wrists
        {-0.3f, 0.0f}, {0.3f, 0.0f},    // hips
        {-0.3f, 0.8f}, {0.3f, 0.8f},    // knees
        {-0.3f, 1.5f}, {0.3f, 1.5f}     // ankles
    };

    for (int i = 0; i < NUM_KEYPOINTS; i++) {
        pose.keypoints[i].x = cx + offsets[i][0] * scale;
        pose.keypoints[i].y = cy + offsets[i][1] * scale;
        pose.keypoints[i].confidence = conf(rng);
    }

    pose.score = conf(rng);

    // Compute bounding box
    float min_x = 1e9f, min_y = 1e9f, max_x = -1e9f, max_y = -1e9f;
    for (int i = 0; i < NUM_KEYPOINTS; i++) {
        min_x = std::min(min_x, pose.keypoints[i].x);
        min_y = std::min(min_y, pose.keypoints[i].y);
        max_x = std::max(max_x, pose.keypoints[i].x);
        max_y = std::max(max_y, pose.keypoints[i].y);
    }
    pose.bbox[0] = min_x - 10;
    pose.bbox[1] = min_y - 10;
    pose.bbox[2] = max_x + 10;
    pose.bbox[3] = max_y + 10;

    return pose;
}

void benchmarkKalmanFilter(int num_tracks, int num_iterations) {
    std::cout << "\n=== Kalman Filter Benchmark ===" << std::endl;
    std::cout << "Tracks: " << num_tracks << ", Iterations: " << num_iterations << std::endl;

    cuda::KalmanFilterCUDA kalman(num_tracks);

    // Generate random poses
    std::mt19937 rng(42);
    std::vector<PoseDetection> poses(num_tracks);
    for (int i = 0; i < num_tracks; i++) {
        poses[i] = generateRandomPose(rng, 1920, 1080);
        kalman.initiate(i, poses[i]);
    }

    // Warm up
    for (int i = 0; i < 10; i++) {
        kalman.predict(num_tracks);
    }
    cudaDeviceSynchronize();

    // Benchmark predict
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        kalman.predict(num_tracks);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    float predict_time = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "Predict time: " << predict_time / num_iterations << " ms/iteration" << std::endl;
}

void benchmarkOKSDistance(int num_tracks, int num_detections, int num_iterations) {
    std::cout << "\n=== OKS Distance Benchmark ===" << std::endl;
    std::cout << "Tracks: " << num_tracks << ", Detections: " << num_detections
              << ", Iterations: " << num_iterations << std::endl;

    cuda::OKSDistanceCUDA oks_distance(num_tracks, num_detections);

    // Generate random poses
    std::mt19937 rng(42);
    std::vector<PoseDetection> tracks(num_tracks);
    std::vector<PoseDetection> dets(num_detections);

    for (int i = 0; i < num_tracks; i++) {
        tracks[i] = generateRandomPose(rng, 1920, 1080);
    }
    for (int i = 0; i < num_detections; i++) {
        dets[i] = generateRandomPose(rng, 1920, 1080);
    }

    std::vector<float> costs(num_tracks * num_detections);

    // Warm up
    for (int i = 0; i < 10; i++) {
        oks_distance.computeOKSDistance(tracks.data(), dets.data(),
                                         costs.data(), num_tracks, num_detections);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        oks_distance.computeOKSDistance(tracks.data(), dets.data(),
                                         costs.data(), num_tracks, num_detections);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    float time = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "OKS distance time: " << time / num_iterations << " ms/iteration" << std::endl;
}

void benchmarkHungarian(int matrix_size, int num_iterations) {
    std::cout << "\n=== Linear Assignment Benchmark ===" << std::endl;
    std::cout << "Matrix size: " << matrix_size << "x" << matrix_size
              << ", Iterations: " << num_iterations << std::endl;

    cuda::LinearAssignmentCUDA solver(matrix_size);

    // Generate random cost matrix
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> costs(matrix_size * matrix_size);
    for (auto& c : costs) {
        c = dist(rng);
    }

    std::vector<int> row_assign(matrix_size);
    std::vector<int> col_assign(matrix_size);

    // Warm up
    for (int i = 0; i < 10; i++) {
        solver.solve(costs.data(), matrix_size, matrix_size,
                    row_assign.data(), col_assign.data(), 0.5f);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        solver.solve(costs.data(), matrix_size, matrix_size,
                    row_assign.data(), col_assign.data(), 0.5f);
    }
    auto end = std::chrono::high_resolution_clock::now();

    float time = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "Linear assignment time: " << time / num_iterations << " ms/iteration" << std::endl;
}

void benchmarkNMS(int num_detections, int num_iterations) {
    std::cout << "\n=== NMS Benchmark ===" << std::endl;
    std::cout << "Detections: " << num_detections
              << ", Iterations: " << num_iterations << std::endl;

    cuda::NMSCuda nms(num_detections);

    // Generate random poses
    std::mt19937 rng(42);
    std::vector<PoseDetection> dets(num_detections);
    for (int i = 0; i < num_detections; i++) {
        dets[i] = generateRandomPose(rng, 1920, 1080);
    }

    // Warm up
    for (int i = 0; i < 10; i++) {
        nms.apply(dets.data(), num_detections, 0.65f, 0.25f);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        nms.apply(dets.data(), num_detections, 0.65f, 0.25f);
    }
    auto end = std::chrono::high_resolution_clock::now();

    float time = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "NMS time: " << time / num_iterations << " ms/iteration" << std::endl;
}

void benchmarkGPUTracker(int num_iterations) {
    std::cout << "\n=== GPU-Native Tracker Benchmark ===" << std::endl;
    std::cout << "Iterations: " << num_iterations << std::endl;

    cuda::GPUTrackerConfig config;
    config.max_tracks = 128;
    config.max_detections = 64;
    cuda::GPUTracker tracker(config);

    std::mt19937 rng(42);
    int num_persons = 5;
    int width = 1920, height = 1080;

    // Allocate device memory for detections
    float* d_det_poses;
    float* d_det_scores;
    cudaMalloc(&d_det_poses, config.max_detections * 17 * 3 * sizeof(float));
    cudaMalloc(&d_det_scores, config.max_detections * sizeof(float));

    // Generate random detection data
    std::vector<float> h_poses(num_persons * 17 * 3);
    std::vector<float> h_scores(num_persons);

    auto generateFrame = [&]() {
        std::uniform_real_distribution<float> pos_x(0.0f, static_cast<float>(width));
        std::uniform_real_distribution<float> pos_y(0.0f, static_cast<float>(height));
        std::uniform_real_distribution<float> conf(0.3f, 1.0f);

        for (int p = 0; p < num_persons; p++) {
            float cx = pos_x(rng);
            float cy = pos_y(rng);
            float scale = 50.0f + cy / height * 100.0f;

            float offsets[17][2] = {
                {0, -1.5f}, {-0.1f, -1.6f}, {0.1f, -1.6f}, {-0.2f, -1.5f}, {0.2f, -1.5f},
                {-0.5f, -1.0f}, {0.5f, -1.0f}, {-0.8f, -0.3f}, {0.8f, -0.3f},
                {-1.0f, 0.3f}, {1.0f, 0.3f}, {-0.3f, 0.0f}, {0.3f, 0.0f},
                {-0.3f, 0.8f}, {0.3f, 0.8f}, {-0.3f, 1.5f}, {0.3f, 1.5f}
            };

            for (int k = 0; k < 17; k++) {
                int idx = p * 17 * 3 + k * 3;
                h_poses[idx + 0] = cx + offsets[k][0] * scale;
                h_poses[idx + 1] = cy + offsets[k][1] * scale;
                h_poses[idx + 2] = conf(rng);
            }
            h_scores[p] = conf(rng);
        }

        cudaMemcpy(d_det_poses, h_poses.data(), num_persons * 17 * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_det_scores, h_scores.data(), num_persons * sizeof(float), cudaMemcpyHostToDevice);
    };

    // Warm up
    for (int i = 0; i < 100; i++) {
        generateFrame();
        tracker.update(d_det_poses, d_det_scores, num_persons, i);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        generateFrame();
        tracker.update(d_det_poses, d_det_scores, num_persons, i);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    float time = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "GPU Tracker update time: " << time / num_iterations << " ms/frame" << std::endl;
    std::cout << "Throughput: " << (1000.0f * num_iterations / time) << " fps" << std::endl;

    tracker.printTimingStats();

    cudaFree(d_det_poses);
    cudaFree(d_det_scores);
}

void benchmarkEngine(const std::string& engine_path, int num_iterations) {
    std::cout << "\n=== TensorRT Engine Benchmark ===" << std::endl;
    std::cout << "Engine: " << engine_path << std::endl;
    std::cout << "Iterations: " << num_iterations << std::endl;

    tensorrt::YoloPoseEngine engine;
    if (!engine.loadEngine(engine_path)) {
        std::cerr << "Failed to load engine" << std::endl;
        return;
    }

    int input_size = 3 * engine.getInputHeight() * engine.getInputWidth();
    std::vector<float> input(input_size);

    // Fill with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : input) {
        v = dist(rng);
    }

    // Warm up
    for (int i = 0; i < 10; i++) {
        engine.detect(input.data());
    }

    // Benchmark
    float total_time = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        engine.detect(input.data());
        total_time += engine.getLastInferenceTime();
    }

    std::cout << "Inference time: " << total_time / num_iterations << " ms/frame" << std::endl;
    std::cout << "Throughput: " << (1000.0f * num_iterations / total_time) << " fps" << std::endl;
}

int main(int argc, char** argv) {
    std::string engine_path;
    int num_iterations = 1000;

    static struct option long_options[] = {
        {"engine", required_argument, 0, 'e'},
        {"iterations", required_argument, 0, 'n'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "e:n:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'e': engine_path = optarg; break;
            case 'n': num_iterations = std::stoi(optarg); break;
            case 'h':
            default:
                std::cout << "Usage: " << argv[0] << " [-e engine_path] [-n iterations]\n";
                return (opt == 'h') ? 0 : 1;
        }
    }

    std::cout << "============================================" << std::endl;
    std::cout << "       PoseBYTE CUDA Benchmark Suite        " << std::endl;
    std::cout << "============================================" << std::endl;

    // Run benchmarks
    benchmarkKalmanFilter(50, num_iterations);
    benchmarkOKSDistance(50, 100, num_iterations);
    benchmarkHungarian(50, num_iterations);
    benchmarkNMS(100, num_iterations);
    benchmarkGPUTracker(num_iterations);

    if (!engine_path.empty()) {
        benchmarkEngine(engine_path, num_iterations);
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "            Benchmark Complete              " << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
