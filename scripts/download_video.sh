#!/bin/bash
# Download a dance video from YouTube for testing

set -e

# Create data directory if it doesn't exist
mkdir -p data

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "yt-dlp not found. Installing..."
    pip install yt-dlp
fi

# Download a dance video
# Using a popular dance video with multiple people
VIDEO_URL="https://www.youtube.com/watch?v=iLnmTe5Q2Qw"  # BTS Dynamite dance practice

echo "Downloading dance video for testing..."
yt-dlp -f 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]' \
    --merge-output-format mp4 \
    -o "data/dance_video.mp4" \
    "$VIDEO_URL"

echo ""
echo "Download complete: data/dance_video.mp4"
echo ""
echo "Alternative videos you can try:"
echo "  - K-pop dance: https://www.youtube.com/watch?v=iLnmTe5Q2Qw"
echo "  - Ballet: https://www.youtube.com/watch?v=4aeETEoNfOg"
echo "  - Street dance: https://www.youtube.com/watch?v=GtL1huin9EE"
echo ""
echo "To download a different video, run:"
echo "  yt-dlp -f 'bestvideo[height<=1080][ext=mp4]' -o 'data/my_video.mp4' <URL>"
