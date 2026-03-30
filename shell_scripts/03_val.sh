#!/bin/bash

# 验证脚本会自动创建带时间戳的输出目录，无需手动删除
python3 scripts/validation.py \
    --annotation_file infer_image_list.txt \
    --checkpoint ./checkpoints/fastbev_2026-01-21_12-38-32/checkpoint_step_48000.pth