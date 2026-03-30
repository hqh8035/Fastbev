#!/bin/bash

# 不限制GPU，让PyTorch使用所有可用的GPU
# 如果需要限制GPU，请使用连续的ID，如: export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,7"
python3 scripts/train.py --config fastbev_v6 --distributed