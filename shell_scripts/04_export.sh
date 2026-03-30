#!/bin/bash

print_usage() {
    echo "Usage: $0 [non-horizon]"
    echo "  non-horizon: export non-horizon onnx model, default is export horizon onnx model"
}

if [ $# -eq 0 ]; then
    export_type="horizon"
elif [ "$1" == "non-horizon" ]; then
    export_type="non-horizon"
else
    print_usage
    exit 1
fi

function check_package() {
    package_name=$1
    if ! pip list | grep -q "$package_name"; then
        echo "$package_name is not installed. Install now..."
        pip install $package_name
        echo "$package_name installed successfully"
    else
        echo "$package_name is already installed"
    fi
}

if [ "$export_type" == "horizon" ]; then
    check_package "onnxsim"
    check_package "onnx_tool"
    python3 scripts/export_onnx_with_horizon.py \
        --checkpoint checkpoints/fastbev_2025-12-29_14-21-51/checkpoint_step_6000.pth \
        --camera_intrinsic 514.644 0 654.782 0 514.644 524.035 0 0 1
else
    python3 scripts/export_onnx.py \
        --checkpoint checkpoints/fastbev_2025-12-29_14-21-51/checkpoint_step_6000.pth \
        --camera_intrinsic 514.644 0 654.782 0 514.644 524.035 0 0 1
fi