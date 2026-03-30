#!/bin/bash

print_usage() {
    echo "Usage: $0 [non-horizon]"
    echo "  non-horizon: test non-horizon onnx model, default is test horizon onnx model"
}

if [ $# -eq 0 ]; then
    test_type="horizon"
elif [ "$1" == "non-horizon" ]; then
    test_type="non-horizon"
else
    print_usage
    exit 1
fi

if [ "$test_type" == "horizon" ]; then
    python3 scripts/test_onnx_horizon.py \
        --onnx_model_path onnx/fastbev_model_simplified.onnx \
        --camera_intrinsic 514.644 0 654.782 0 514.644 524.035 0 0 1
else
    # 测试脚本会自动创建带时间戳的输出目录
    python3 scripts/test_onnx.py \
        --onnx_model_path onnx/fastbev_model_simplified.onnx \
        --camera_intrinsic 514.644 0 654.782 0 514.644 524.035 0 0 1
fi