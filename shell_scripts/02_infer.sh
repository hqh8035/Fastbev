#!/bin/bash

# caijiche
## f: 5.2431359863281250e+02
## cx: 6.5763726806640625e+02
## cy: 5.1953070068359375e+02
## base_line: 7.9946304321289062e+01

export CUDA_VISIBLE_DEVICES="0"
python3 scripts/inference.py --model_path ./checkpoints/fastbev_2026-01-14_16-31-11/checkpoint_step_48000.pth \
    --camera_intrinsic 524.31359863281250 0 657.63726806640625 0 524.31359863281250 519.53070068359375 0 0 1 \
    --frame_size 696 320 \
    --disable_seg \
    --make_video