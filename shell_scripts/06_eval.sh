#!/bin/bash

# 评估脚本会自动保存结果到 outputs/eval/eval_TIMESTAMP/ 目录
# 使用 nuScenes mAP 进行评估（更适合小目标）
python3 scripts/eval.py \
    --gt_dir /perception/users/chenfuxuan/WorkSpace_2025/Data/liufen/accepted_data/batch13/labels/caijiche \
    --pd_dir outputs/infer/infer_2026-03-02_12-03-16 \
    --use_nuscenes_map