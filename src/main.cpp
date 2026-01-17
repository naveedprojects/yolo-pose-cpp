#include "tensorrt/yolo_pose_engine.h"
#include "cuda/gpu_tracker.h"
#include "utils/video_utils.h"
#include "cuda/preprocess.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <getopt.h>

using namespace posebyte;

void printUsage(const char* program) {
    std::cout << "PoseBYTE GPU-Native Tracker Demo\n";
    std::cout << "Usage: " << program << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  -e, --engine PATH    TensorRT engine file (required)\n";
    std::cout << "  -i, --input PATH     Input video file (required)\n";
    std::cout << "  -o, --output PATH    Output video file (optional)\n";
    std::cout << "  -c, --conf FLOAT     Confidence threshold (default: 0.30)\n";
    std::cout << "  -n, --nms FLOAT      NMS threshold (default: 0.65)\n";
    std::cout << "  -t, --track FLOAT    Track match threshold (default: 0.3)\n";
    std::cout << "  -a, --max-age INT    Max frames a track can be undetected before deletion (default: 10)\n";
    std::cout << "  -d, --display        Display output in window\n";
    std::cout << "  -v, --verbose        Show per-frame detection/tracking details\n";
    std::cout << "  -h, --help           Show this help message\n";
}

void printProgressBar(int current, int total, float fps) {
    int barWidth = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1)
              << (progress * 100.0f) << "% | "
              << current << "/" << total << " frames | "
              << std::setprecision(0) << fps << " FPS" << std::flush;
}

// Scale TrackOutput keypoints from model coordinates to original frame coordinates
void scaleTrackOutputs(
    std::vector<TrackOutput>& tracks,
    float scale_x,
    float scale_y,
    int pad_x,
    int pad_y
) {
    for (auto& track : tracks) {
        // Scale bounding box
        track.bbox[0] = (track.bbox[0] - pad_x) * scale_x;
        track.bbox[1] = (track.bbox[1] - pad_y) * scale_y;
        track.bbox[2] = (track.bbox[2] - pad_x) * scale_x;
        track.bbox[3] = (track.bbox[3] - pad_y) * scale_y;

        // Scale keypoints
        for (int i = 0; i < NUM_KEYPOINTS; i++) {
            track.keypoints[i].x = (track.keypoints[i].x - pad_x) * scale_x;
            track.keypoints[i].y = (track.keypoints[i].y - pad_y) * scale_y;
        }
    }
}

int main(int argc, char** argv) {
    // Default parameters
    std::string engine_path;
    std::string input_path;
    std::string output_path;
    float conf_threshold = 0.30f;
    float nms_threshold = 0.65f;
    float track_threshold = 0.3f;  // Cost threshold (1 - OKS)
    int max_age = 10;              // Frames before track is lost
    bool display = false;
    bool verbose = false;

    // Parse command line arguments
    static struct option long_options[] = {
        {"engine", required_argument, 0, 'e'},
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"conf", required_argument, 0, 'c'},
        {"nms", required_argument, 0, 'n'},
        {"track", required_argument, 0, 't'},
        {"max-age", required_argument, 0, 'a'},
        {"display", no_argument, 0, 'd'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "e:i:o:c:n:t:a:dvh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'e': engine_path = optarg; break;
            case 'i': input_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 'c': conf_threshold = std::stof(optarg); break;
            case 'n': nms_threshold = std::stof(optarg); break;
            case 't': track_threshold = std::stof(optarg); break;
            case 'a': max_age = std::stoi(optarg); break;
            case 'v': verbose = true; break;
            case 'd': display = true; break;
            case 'h':
            default:
                printUsage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }

    // Validate arguments
    if (engine_path.empty() || input_path.empty()) {
        std::cerr << "Error: Engine and input paths are required\n";
        printUsage(argv[0]);
        return 1;
    }

    // Initialize TensorRT engine
    std::cout << "Loading TensorRT engine: " << engine_path << std::endl;
    tensorrt::YoloPoseEngine engine;
    if (!engine.loadEngine(engine_path)) {
        std::cerr << "Failed to load engine: " << engine_path << std::endl;
        return 1;
    }

    // Initialize GPU-native tracker
    cuda::GPUTrackerConfig tracker_config;
    tracker_config.match_threshold = 0.5f;  // OKS > 0.5 for match (cost < 0.5)
    tracker_config.high_thresh = conf_threshold;
    tracker_config.low_thresh = conf_threshold * 0.5f;  // 0.15 with conf=0.30
    tracker_config.new_track_thresh = conf_threshold;
    tracker_config.min_hits = 3;      // Require 3 hits before track is confirmed
    tracker_config.max_age = max_age; // User-configurable track persistence
    tracker_config.max_tracks = 128;
    tracker_config.max_detections = 64;

    cuda::GPUTracker tracker(tracker_config);
    std::cout << "GPU-native tracker initialized (max " << tracker_config.max_tracks
              << " tracks, " << tracker_config.max_detections << " detections)" << std::endl;

    // Open video
    std::cout << "Opening video: " << input_path << std::endl;
    utils::VideoReader video(input_path);
    if (!video.isOpened()) {
        std::cerr << "Failed to open video: " << input_path << std::endl;
        return 1;
    }

    std::cout << "Video info: " << video.getWidth() << "x" << video.getHeight()
              << " @ " << video.getFPS() << " fps, "
              << video.getFrameCount() << " frames" << std::endl;

    // Initialize video writer if output specified
    std::unique_ptr<utils::VideoWriter> writer;
    if (!output_path.empty()) {
        writer = std::make_unique<utils::VideoWriter>(
            output_path, video.getWidth(), video.getHeight(), video.getFPS());
        if (!writer->isOpened()) {
            std::cerr << "Failed to create output video: " << output_path << std::endl;
            return 1;
        }
        std::cout << "Writing output to: " << output_path << std::endl;
    }

    // Initialize GPU preprocessor
    cuda::PreprocessorCUDA preprocessor(
        video.getWidth(), video.getHeight(),
        engine.getInputWidth(), engine.getInputHeight()
    );

    // Allocate device memory for preprocessed input
    float* d_input;
    int input_size = 3 * engine.getInputHeight() * engine.getInputWidth();
    cudaMalloc(&d_input, input_size * sizeof(float));

    // Processing variables
    cv::Mat frame;
    int frame_id = 0;
    float total_fps = 0.0f;
    int fps_count = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "\nProcessing video with GPU-native tracker..." << std::endl;

    // Timing accumulators
    float total_preprocess_ms = 0, total_detect_ms = 0, total_track_ms = 0;

    while (video.read(frame)) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        // GPU Preprocess
        auto t1 = std::chrono::high_resolution_clock::now();
        float scale_x, scale_y;
        int pad_x, pad_y;
        preprocessor.preprocess(frame.data, frame.cols, frame.rows,
                                d_input, scale_x, scale_y, pad_x, pad_y);
        auto t2 = std::chrono::high_resolution_clock::now();
        total_preprocess_ms += std::chrono::duration<float, std::milli>(t2 - t1).count();

        // GPU-native detection (stays on device)
        int num_detections = engine.detectGPUNative(d_input, conf_threshold, nms_threshold);
        auto t3 = std::chrono::high_resolution_clock::now();
        total_detect_ms += std::chrono::duration<float, std::milli>(t3 - t2).count();

        auto* gpu_postprocess = engine.getGPUPostprocess();

        // GPU-native tracking (data stays on device)
        tracker.update(
            gpu_postprocess->getDetectionPoses(),
            gpu_postprocess->getDetectionScores(),
            num_detections,
            frame_id
        );
        auto t4 = std::chrono::high_resolution_clock::now();
        total_track_ms += std::chrono::duration<float, std::milli>(t4 - t3).count();

        // Get tracks for visualization (only D2H copy in the pipeline)
        auto tracks = tracker.getActiveTracks();

        // Scale tracks back to original coordinates
        scaleTrackOutputs(tracks, scale_x, scale_y, pad_x, pad_y);

        // Visualize
        utils::drawAllTracks(frame, tracks);

        auto frame_end = std::chrono::high_resolution_clock::now();
        float frame_time = std::chrono::duration<float, std::milli>(frame_end - frame_start).count();
        float fps = 1000.0f / frame_time;

        total_fps += fps;
        fps_count++;

        // Display count of tracks actually being drawn (confirmed tracks)
        utils::drawStats(frame, fps, static_cast<int>(tracks.size()), engine.getLastInferenceTime());

        // Write output
        if (writer) {
            writer->write(frame);
        }

        // Display
        if (display) {
            cv::imshow("PoseBYTE GPU-Native Tracker", frame);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q') {  // ESC or Q to quit
                break;
            }
        }

        // Progress update
        if (verbose) {
            // Verbose: show per-frame details every 30 frames
            if (frame_id % 30 == 0) {
                std::cout << "Frame " << frame_id << " - Dets: " << num_detections
                          << " - Tracks: " << tracks.size();
                if (!tracks.empty()) {
                    std::cout << " - IDs: [";
                    for (size_t i = 0; i < tracks.size() && i < 5; i++) {
                        if (i > 0) std::cout << ",";
                        std::cout << tracks[i].track_id;
                    }
                    if (tracks.size() > 5) std::cout << "...";
                    std::cout << "]";
                }
                std::cout << std::endl;
            }
        } else {
            // Normal: show progress bar
            if (frame_id % 10 == 0) {
                printProgressBar(frame_id, video.getFrameCount(), fps);
            }
        }

        frame_id++;
    }

    // Clear progress bar line
    if (!verbose) {
        std::cout << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float>(end_time - start_time).count();

    // Print statistics
    std::cout << "\n=== Processing Complete ===" << std::endl;
    std::cout << "Total frames: " << frame_id << std::endl;
    std::cout << "Total time: " << total_time << " seconds" << std::endl;
    std::cout << "Average FPS: " << (fps_count > 0 ? total_fps / fps_count : 0.0f) << std::endl;

    // Timing breakdown
    std::cout << "\n=== Timing Breakdown (GPU-Native) ===" << std::endl;
    std::cout << "Avg preprocess: " << total_preprocess_ms / frame_id << " ms" << std::endl;
    std::cout << "Avg detect (TensorRT + GPU postprocess): " << total_detect_ms / frame_id << " ms" << std::endl;
    std::cout << "Avg track (GPU-native): " << total_track_ms / frame_id << " ms" << std::endl;
    std::cout << "Total per frame: " << (total_preprocess_ms + total_detect_ms + total_track_ms) / frame_id << " ms" << std::endl;
    std::cout << "Theoretical max FPS: " << 1000.0f / ((total_preprocess_ms + total_detect_ms + total_track_ms) / frame_id) << std::endl;

    if (display) {
        cv::destroyAllWindows();
    }

    // Cleanup
    cudaFree(d_input);

    return 0;
}
