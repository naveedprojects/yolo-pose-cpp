#pragma once

#include "types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace posebyte {
namespace utils {

// Color palette for visualization
extern const cv::Scalar TRACK_COLORS[20];

// Skeleton connections for visualization
extern const std::pair<int, int> SKELETON_CONNECTIONS[19];

// Draw pose on image
void drawPose(
    cv::Mat& image,
    const PoseDetection& pose,
    const cv::Scalar& color = cv::Scalar(0, 255, 0),
    int thickness = 2,
    float conf_threshold = 0.3f
);

// Draw tracked pose with ID
void drawTrackedPose(
    cv::Mat& image,
    const Track& track,
    int thickness = 2,
    float conf_threshold = 0.3f
);

// Draw all tracks
void drawAllTracks(
    cv::Mat& image,
    const std::vector<Track>& tracks,
    int thickness = 2,
    float conf_threshold = 0.3f
);

// Draw all tracks from GPU tracker output
void drawAllTracks(
    cv::Mat& image,
    const std::vector<TrackOutput>& tracks,
    int thickness = 2,
    float conf_threshold = 0.3f
);

// Draw FPS and stats
void drawStats(
    cv::Mat& image,
    float fps,
    int num_tracks,
    float inference_time_ms
);

// Video reader wrapper with preprocessing
class VideoReader {
public:
    VideoReader(const std::string& path);
    ~VideoReader();

    bool isOpened() const;
    bool read(cv::Mat& frame);

    int getWidth() const;
    int getHeight() const;
    double getFPS() const;
    int getFrameCount() const;

private:
    cv::VideoCapture cap_;
    int width_;
    int height_;
    double fps_;
    int frame_count_;
};

// Video writer wrapper
class VideoWriter {
public:
    VideoWriter(
        const std::string& path,
        int width,
        int height,
        double fps,
        const std::string& codec = "mp4v"
    );
    ~VideoWriter();

    bool isOpened() const;
    void write(const cv::Mat& frame);

private:
    cv::VideoWriter writer_;
};

// Preprocess frame for YOLO-Pose
// Returns preprocessed tensor and scale factors
void preprocessFrame(
    const cv::Mat& frame,
    float* output_tensor,
    int target_width,
    int target_height,
    float& scale_x,
    float& scale_y,
    int& pad_x,
    int& pad_y
);

// Scale detections back to original frame coordinates
void scaleDetections(
    std::vector<PoseDetection>& detections,
    float scale_x,
    float scale_y,
    int pad_x,
    int pad_y
);

}  // namespace utils
}  // namespace posebyte
