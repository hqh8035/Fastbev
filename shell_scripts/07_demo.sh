#!/bin/bash

# 执行命令前，先激活 conda 环境 nova_fastbev

# 1. 首先按照README.md中的步骤，拉取数据，生成图像列表

# caijiche
## f: 5.2431359863281250e+02
## cx: 6.5763726806640625e+02
## cy: 5.1953070068359375e+02
## base_line: 7.9946304321289062e+01

# 2. 推理：
python scripts/inference.py \
    --model_path ./checkpoints/fastbev_2026-01-21_12-38-32/checkpoint_step_48000.pth \
    --camera_intrinsic 524.31359863281250 0 657.63726806640625 0 524.31359863281250 519.53070068359375 0 0 1 \
    --frame_size 696 320 \
    --image_list_file demo_image_list.txt \
    --disable_seg \
    --make_video