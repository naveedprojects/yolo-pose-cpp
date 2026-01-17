#!/usr/bin/env python3
"""
Download and export YOLO-Pose model to ONNX format.
Then use the C++ export_engine tool to convert to TensorRT.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Setup YOLO-Pose model')
    parser.add_argument('--model', type=str, default='yolov8n-pose',
                        choices=['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose',
                                 'yolov8l-pose', 'yolov8x-pose', 'yolo11n-pose',
                                 'yolo11s-pose', 'yolo11m-pose', 'yolo11l-pose'],
                        help='Model variant to download')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    print(f"Loading {args.model}...")
    model = YOLO(f"{args.model}.pt")

    # Export to ONNX
    print(f"Exporting to ONNX (imgsz={args.imgsz}, opset={args.opset})...")
    onnx_path = model.export(
        format='onnx',
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False  # Static shapes for TensorRT
    )

    # Move to output directory
    final_path = os.path.join(args.output_dir, os.path.basename(onnx_path))
    if onnx_path != final_path:
        import shutil
        shutil.move(onnx_path, final_path)

    print(f"\nONNX model saved to: {final_path}")
    print("\nTo convert to TensorRT, use the export_engine tool:")
    print(f"  ./build/export_engine -m {final_path} -o models/{args.model}.engine -p fp16")
    print(f"  ./build/export_engine -m {final_path} -o models/{args.model}_int8.engine -p int8")

    # Also try to create TensorRT engine directly using trtexec if available
    print("\nOr use NVIDIA's trtexec directly:")
    print(f"  trtexec --onnx={final_path} --saveEngine=models/{args.model}_fp16.engine --fp16")
    print(f"  trtexec --onnx={final_path} --saveEngine=models/{args.model}_int8.engine --int8")

if __name__ == '__main__':
    main()
