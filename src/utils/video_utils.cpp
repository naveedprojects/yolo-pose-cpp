#include "utils/video_utils.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace posebyte {
namespace utils {

// Color palette for tracks (20 distinct colors)
const cv::Scalar TRACK_COLORS[20] = {
    cv::Scalar(255, 0, 0),     // Blue
    cv::Scalar(0, 255, 0),     // Green
    cv::Scalar(0, 0, 255),     // Red
    cv::Scalar(255, 255, 0),   // Cyan
    cv::Scalar(255, 0, 255),   // Magenta
    cv::Scalar(0, 255, 255),   // Yellow
    cv::Scalar(128, 0, 0),     // Dark Blue
    cv::Scalar(0, 128, 0),     // Dark Green
    cv::Scalar(0, 0, 128),     // Dark Red
    cv::Scalar(128, 128, 0),   // Dark Cyan
    cv::Scalar(128, 0, 128),   // Dark Magenta
    cv::Scalar(0, 128, 128),   // Dark Yellow
    cv::Scalar(255, 128, 0),   // Orange-Blue
    cv::Scalar(255, 0, 128),   // Pink-Blue
    cv::Scalar(128, 255, 0),   // Lime
    cv::Scalar(0, 255, 128),   // Spring Green
    cv::Scalar(128, 0, 255),   // Purple
    cv::Scalar(0, 128, 255),   // Orange
    cv::Scalar(255, 128, 128), // Light Blue
    cv::Scalar(128, 255, 128)  // Light Green
};

// COCO skeleton connections
const std::pair<int, int> SKELETON_CONNECTIONS[19] = {
    // Head
    {0, 1}, {0, 2}, {1, 3}, {2, 4},
    // Body
    {5, 6},   // shoulders
    {5, 7}, {7, 9},    // left arm
    {6, 8}, {8, 10},   // right arm
    {5, 11}, {6, 12},  // torso
    {11, 12},          // hips
    // Legs
    {11, 13}, {13, 15}, // left leg
    {12, 14}, {14, 16}, // right leg
    // Additional connections
    {0, 5}, {0, 6}      // nose to shoulders
};

void drawPose(
    cv::Mat& image,
    const PoseDetection& pose,
    const cv::Scalar& color,
    int thickness,
    float conf_threshold
) {
    // Draw skeleton lines
    for (const auto& [i, j] : SKELETON_CONNECTIONS) {
        if (pose.keypoints[i].confidence > conf_threshold &&
            pose.keypoints[j].confidence > conf_threshold) {
            cv::Point p1(static_cast<int>(pose.keypoints[i].x),
                        static_cast<int>(pose.keypoints[i].y));
            cv::Point p2(static_cast<int>(pose.keypoints[j].x),
                        static_cast<int>(pose.keypoints[j].y));
            cv::line(image, p1, p2, color, thickness);
        }
    }

    // Draw keypoints
    for (int i = 0; i < NUM_KEYPOINTS; i++) {
        if (pose.keypoints[i].confidence > conf_threshold) {
            cv::Point center(static_cast<int>(pose.keypoints[i].x),
                           static_cast<int>(pose.keypoints[i].y));
            cv::circle(image, center, thickness + 2, color, -1);
        }
    }
}

void drawTrackedPose(
    cv::Mat& image,
    const Track& track,
    int thickness,
    float conf_threshold
) {
    const cv::Scalar& color = TRACK_COLORS[track.id % 20];

    // Draw pose
    drawPose(image, track.pose, color, thickness, conf_threshold);

    // Draw bounding box
    cv::rectangle(image,
                  cv::Point(static_cast<int>(track.pose.bbox[0]),
                           static_cast<int>(track.pose.bbox[1])),
                  cv::Point(static_cast<int>(track.pose.bbox[2]),
                           static_cast<int>(track.pose.bbox[3])),
                  color, 1);

    // Draw ID label
    std::ostringstream oss;
    oss << "ID:" << track.id << " (" << std::fixed << std::setprecision(2)
        << track.score << ")";

    int baseline;
    cv::Size text_size = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX,
                                          0.5, 1, &baseline);

    cv::Point text_org(static_cast<int>(track.pose.bbox[0]),
                       static_cast<int>(track.pose.bbox[1]) - 5);

    // Background rectangle for text
    cv::rectangle(image,
                  cv::Point(text_org.x, text_org.y - text_size.height - 2),
                  cv::Point(text_org.x + text_size.width, text_org.y + 2),
                  color, -1);

    cv::putText(image, oss.str(), text_org, cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(255, 255, 255), 1);
}

void drawAllTracks(
    cv::Mat& image,
    const std::vector<Track>& tracks,
    int thickness,
    float conf_threshold
) {
    for (const auto& track : tracks) {
        if (track.state == TrackState::Tracked) {
            drawTrackedPose(image, track, thickness, conf_threshold);
        }
    }
}

// Overload for GPU tracker output (TrackOutput)
void drawAllTracks(
    cv::Mat& image,
    const std::vector<TrackOutput>& tracks,
    int thickness,
    float conf_threshold
) {
    for (const auto& track : tracks) {
        const cv::Scalar& color = TRACK_COLORS[track.track_id % 20];

        // Draw skeleton lines
        for (const auto& [i, j] : SKELETON_CONNECTIONS) {
            if (track.keypoints[i].confidence > conf_threshold &&
                track.keypoints[j].confidence > conf_threshold) {
                cv::Point p1(static_cast<int>(track.keypoints[i].x),
                            static_cast<int>(track.keypoints[i].y));
                cv::Point p2(static_cast<int>(track.keypoints[j].x),
                            static_cast<int>(track.keypoints[j].y));
                cv::line(image, p1, p2, color, thickness);
            }
        }

        // Draw keypoints
        for (int i = 0; i < NUM_KEYPOINTS; i++) {
            if (track.keypoints[i].confidence > conf_threshold) {
                cv::Point center(static_cast<int>(track.keypoints[i].x),
                               static_cast<int>(track.keypoints[i].y));
                cv::circle(image, center, thickness + 2, color, -1);
            }
        }

        // Draw bounding box
        cv::rectangle(image,
                      cv::Point(static_cast<int>(track.bbox[0]),
                               static_cast<int>(track.bbox[1])),
                      cv::Point(static_cast<int>(track.bbox[2]),
                               static_cast<int>(track.bbox[3])),
                      color, 1);

        // Draw ID label
        std::ostringstream oss;
        oss << "ID:" << track.track_id << " (" << std::fixed << std::setprecision(2)
            << track.score << ")";

        int baseline;
        cv::Size text_size = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX,
                                              0.5, 1, &baseline);

        cv::Point text_org(static_cast<int>(track.bbox[0]),
                           static_cast<int>(track.bbox[1]) - 5);

        // Background rectangle for text
        cv::rectangle(image,
                      cv::Point(text_org.x, text_org.y - text_size.height - 2),
                      cv::Point(text_org.x + text_size.width, text_org.y + 2),
                      color, -1);

        cv::putText(image, oss.str(), text_org, cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(255, 255, 255), 1);
    }
}

void drawStats(
    cv::Mat& image,
    float fps,
    int num_tracks,
    float inference_time_ms
) {
    std::vector<std::string> lines;

    std::ostringstream oss;
    oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    lines.push_back(oss.str());

    oss.str("");
    oss << "Tracks: " << num_tracks;
    lines.push_back(oss.str());

    oss.str("");
    oss << "Inference: " << std::fixed << std::setprecision(1) << inference_time_ms << " ms";
    lines.push_back(oss.str());

    int y = 25;
    for (const auto& line : lines) {
        cv::putText(image, line, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 0, 0), 3);  // Shadow
        cv::putText(image, line, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 0), 2);
        y += 25;
    }
}

// ============================================================================
// VideoReader Implementation
// ============================================================================

VideoReader::VideoReader(const std::string& path) {
    cap_.open(path);
    if (cap_.isOpened()) {
        width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps_ = cap_.get(cv::CAP_PROP_FPS);
        frame_count_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    } else {
        width_ = height_ = frame_count_ = 0;
        fps_ = 0.0;
    }
}

VideoReader::~VideoReader() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool VideoReader::isOpened() const { return cap_.isOpened(); }
bool VideoReader::read(cv::Mat& frame) { return cap_.read(frame); }
int VideoReader::getWidth() const { return width_; }
int VideoReader::getHeight() const { return height_; }
double VideoReader::getFPS() const { return fps_; }
int VideoReader::getFrameCount() const { return frame_count_; }

// ============================================================================
// VideoWriter Implementation
// ============================================================================

VideoWriter::VideoWriter(
    const std::string& path,
    int width,
    int height,
    double fps,
    const std::string& codec
) {
    int fourcc = cv::VideoWriter::fourcc(
        codec[0], codec[1], codec[2], codec[3]);
    writer_.open(path, fourcc, fps, cv::Size(width, height));
}

VideoWriter::~VideoWriter() {
    if (writer_.isOpened()) {
        writer_.release();
    }
}

bool VideoWriter::isOpened() const { return writer_.isOpened(); }
void VideoWriter::write(const cv::Mat& frame) { writer_.write(frame); }

// ============================================================================
// Preprocessing Functions
// ============================================================================

void preprocessFrame(
    const cv::Mat& frame,
    float* output_tensor,
    int target_width,
    int target_height,
    float& scale_x,
    float& scale_y,
    int& pad_x,
    int& pad_y
) {
    // Calculate scale to maintain aspect ratio
    float scale = std::min(
        static_cast<float>(target_width) / frame.cols,
        static_cast<float>(target_height) / frame.rows
    );

    int new_width = static_cast<int>(frame.cols * scale);
    int new_height = static_cast<int>(frame.rows * scale);

    pad_x = (target_width - new_width) / 2;
    pad_y = (target_height - new_height) / 2;

    // Store inverse scale for detection scaling
    scale_x = 1.0f / scale;
    scale_y = 1.0f / scale;

    // Resize image
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_width, new_height));

    // Create padded image with gray background (114)
    cv::Mat padded(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_width, new_height)));

    // Convert BGR to RGB and normalize to [0, 1]
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // Convert to float and normalize
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // Convert HWC to CHW format
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    for (int c = 0; c < 3; c++) {
        std::memcpy(
            output_tensor + c * target_height * target_width,
            channels[c].data,
            target_height * target_width * sizeof(float)
        );
    }
}

void scaleDetections(
    std::vector<PoseDetection>& detections,
    float scale_x,
    float scale_y,
    int pad_x,
    int pad_y
) {
    for (auto& det : detections) {
        // Scale bounding box
        det.bbox[0] = (det.bbox[0] - pad_x) * scale_x;
        det.bbox[1] = (det.bbox[1] - pad_y) * scale_y;
        det.bbox[2] = (det.bbox[2] - pad_x) * scale_x;
        det.bbox[3] = (det.bbox[3] - pad_y) * scale_y;

        // Scale keypoints
        for (int i = 0; i < NUM_KEYPOINTS; i++) {
            det.keypoints[i].x = (det.keypoints[i].x - pad_x) * scale_x;
            det.keypoints[i].y = (det.keypoints[i].y - pad_y) * scale_y;
        }
    }
}

}  // namespace utils
}  // namespace posebyte
