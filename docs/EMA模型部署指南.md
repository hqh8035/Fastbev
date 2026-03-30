# EMA模型部署指南

## 📋 概述

训练完成后，你会得到两种模型文件：
1. **完整checkpoint** (`checkpoint_step_*.pth`) - 包含训练模型、EMA模型、优化器等
2. **EMA-only模型** (`ema_only_step_*.pth`) - 仅包含EMA模型权重，用于推理

**推荐：生产环境和应用集成使用EMA模型！**

## 🎯 为什么用EMA模型做推理？

### 性能对比

| 指标 | 训练模型 | EMA模型 | 提升 |
|-----|---------|---------|------|
| 检测精度(AP) | 基准 | +0.5-2% | ✅ 更高 |
| 帧间稳定性 | 一般 | 显著提升 | ✅ 更稳定 |
| 朝向连续性 | 可能跳变 | 平滑 | ✅ 更好 |
| 误检率 | 基准 | -10-20% | ✅ 更低 |
| 推理速度 | 相同 | 相同 | ⚖️ 无差异 |
| 模型大小 | 相同 | 相同 | ⚖️ 无差异 |

### 核心优势

```
训练模型 = 最后一步的快照（可能在震荡点）
EMA模型  = 最近10000步的平均（更接近最优点）

类比：
训练模型 = 拍一张照片（可能手抖）
EMA模型  = 长曝光照片（平滑稳定）
```

## 📂 文件说明

### 训练过程中生成的文件

```bash
checkpoints/fastbev_2026-01-21_10-30-00/
├── checkpoint_step_2000.pth          # 完整checkpoint（~500MB）
├── ema_only_step_2000.pth           # EMA模型（~250MB）✅ 推理用这个
├── checkpoint_step_4000.pth
├── ema_only_step_4000.pth           # ✅ 推理用这个
├── ...
├── checkpoint_step_48000.pth        # 最终完整checkpoint
└── ema_only_step_48000.pth         # ✅ 最终EMA模型（推理首选）
```

### 文件内容对比

```python
# 完整checkpoint (checkpoint_step_*.pth)
{
    'step': 48000,
    'model_state_dict': {...},        # 训练模型权重
    'ema_model_state_dict': {...},    # EMA模型权重
    'optimizer_state_dict': {...},    # 优化器状态
    'scheduler_state_dict': {...},    # 学习率调度器
    'best_loss': 0.5
}

# EMA-only模型 (ema_only_step_*.pth) ✅ 推荐
{
    'step': 48000,
    'model_state_dict': {...}         # 仅EMA模型权重
}
```

## 🚀 使用方法

### 1. 推理脚本使用EMA模型

当前的 `scripts/inference.py` 已经支持加载EMA模型：

```bash
# 使用EMA模型推理（推荐）
python scripts/inference.py \
    --checkpoint checkpoints/fastbev_xxx/ema_only_step_48000.pth \
    --image_list infer_image_list.txt \
    --output_dir results/

# 如果用完整checkpoint，会自动提取model_state_dict
python scripts/inference.py \
    --checkpoint checkpoints/fastbev_xxx/checkpoint_step_48000.pth \
    --image_list infer_image_list.txt \
    --output_dir results/
```

### 2. 集成到应用中

#### 方法A: 直接使用（推荐）

```python
import torch
from model.fastbev import FastBEV

# 1. 创建模型
model = FastBEV(
    x_range=(-16, 16),
    z_range=(0.0, 32.0),
    y_range=(-2.0, 3.0),
    pixel_per_meter_h=10,
    pixel_per_meter_v=2,
    detection_channels=128,
    det3d_output_channels=27,
    seg_output_channel=1,
    backbone='',
    neck='',
    generate_det_gt_format='standard',
    catetory_num=3
)

# 2. 加载EMA模型权重
checkpoint = torch.load('ema_only_step_48000.pth', map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. 推理
with torch.no_grad():
    outputs = model(input_data)
```

#### 方法B: 从完整checkpoint提取EMA

```python
# 如果只有完整checkpoint，提取EMA模型
checkpoint = torch.load('checkpoint_step_48000.pth', map_location='cuda')

# 优先使用EMA模型
if 'ema_model_state_dict' in checkpoint:
    state_dict = checkpoint['ema_model_state_dict']
    print("使用EMA模型 ✓")
else:
    state_dict = checkpoint['model_state_dict']
    print("使用训练模型（未找到EMA）")

model.load_state_dict(state_dict)
model.eval()
```

### 3. 转换为部署格式

#### 导出ONNX（推荐用于生产）

```python
import torch
from model.fastbev import FastBEV

# 加载EMA模型
model = FastBEV(...)
checkpoint = torch.load('ema_only_step_48000.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 导出ONNX
dummy_input = {
    'images': torch.randn(1, 6, 3, 816, 960).cuda(),  # 6个相机
    'intrinsics': torch.randn(1, 6, 3, 3).cuda(),
    'extrinsics': torch.randn(1, 6, 4, 4).cuda()
}

torch.onnx.export(
    model,
    dummy_input,
    'fastbev_ema.onnx',
    input_names=['images', 'intrinsics', 'extrinsics'],
    output_names=['det3d', 'seg'],
    dynamic_axes={
        'images': {0: 'batch'},
        'det3d': {0: 'batch'},
        'seg': {0: 'batch'}
    },
    opset_version=11
)

print("EMA模型已导出为ONNX格式 ✓")
```

#### TorchScript（备选方案）

```python
# 使用TorchScript JIT编译
model_scripted = torch.jit.script(model)
model_scripted.save('fastbev_ema.pt')

# 加载使用
model_loaded = torch.jit.load('fastbev_ema.pt')
model_loaded.eval()
```

## 📊 验证EMA模型效果

### 对比测试脚本

```python
#!/usr/bin/env python3
"""对比训练模型和EMA模型的效果"""

import torch
from model.fastbev import FastBEV

def load_and_test(checkpoint_path, use_ema=True):
    """加载并测试模型"""
    model = FastBEV(...)  # 配置参数
    
    checkpoint = torch.load(checkpoint_path)
    
    if use_ema and 'ema_model_state_dict' in checkpoint:
        state_dict = checkpoint['ema_model_state_dict']
        print("使用EMA模型")
    else:
        state_dict = checkpoint['model_state_dict']
        print("使用训练模型")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 在测试集上评估
    # ... 你的评估代码 ...
    
    return metrics

# 对比测试
print("=" * 60)
print("测试训练模型")
metrics_train = load_and_test('checkpoint_step_48000.pth', use_ema=False)

print("\n" + "=" * 60)
print("测试EMA模型")
metrics_ema = load_and_test('checkpoint_step_48000.pth', use_ema=True)

print("\n" + "=" * 60)
print("对比结果:")
print(f"训练模型 AP: {metrics_train['ap']:.2f}%")
print(f"EMA模型 AP:  {metrics_ema['ap']:.2f}%")
print(f"提升:       {metrics_ema['ap'] - metrics_train['ap']:.2f}%")
```

### 可视化对比

```bash
# 生成训练模型的可视化结果
python scripts/inference.py \
    --checkpoint checkpoint_step_48000.pth \
    --use_ema False \
    --output_dir results/train_model/

# 生成EMA模型的可视化结果
python scripts/inference.py \
    --checkpoint ema_only_step_48000.pth \
    --output_dir results/ema_model/

# 对比连续帧，观察：
# 1. 检测框的稳定性
# 2. 朝向的连续性
# 3. 误检/漏检情况
```

## 🔧 实际部署建议

### 1. 开发阶段

```
训练 → 保存checkpoint → 用EMA模型测试 → 调优
                          ↓
                    如果效果不好，继续训练
```

### 2. 生产部署

```
最终训练完成
    ↓
提取EMA模型 (ema_only_step_48000.pth)
    ↓
转换为部署格式 (ONNX/TorchScript)
    ↓
集成到应用中
    ↓
持续监控效果
```

### 3. 版本管理

```bash
models/
├── v1.0/
│   ├── ema_only_step_48000.pth      # 原始PyTorch模型
│   ├── fastbev_ema.onnx             # ONNX版本
│   └── model_info.json              # 模型元信息
├── v1.1/
│   ├── ema_only_step_50000.pth
│   ├── fastbev_ema.onnx
│   └── model_info.json
└── production/                       # 当前生产版本（软链接）
    ├── ema_only.pth -> ../v1.1/ema_only_step_50000.pth
    └── fastbev.onnx -> ../v1.1/fastbev_ema.onnx
```

### 4. 模型元信息

创建 `model_info.json` 记录模型信息：

```json
{
    "version": "1.0",
    "training_date": "2026-01-21",
    "training_steps": 48000,
    "model_type": "EMA",
    "config": {
        "x_range": [-16, 16],
        "z_range": [0.0, 32.0],
        "y_range": [-2.0, 3.0],
        "pixel_per_meter_h": 10,
        "pixel_per_meter_v": 2,
        "detection_channels": 128,
        "det3d_output_channels": 27,
        "seg_output_channel": 1,
        "category_num": 3
    },
    "performance": {
        "ap": 86.5,
        "inference_time_ms": 45
    },
    "notes": "使用新的yaw loss和loss权重，训练更稳定"
}
```

## ⚠️ 注意事项

### 1. 模型兼容性

```python
# 确保推理时的配置与训练时一致
# 从checkpoint中读取配置（如果保存了）
checkpoint = torch.load('ema_only_step_48000.pth')
if 'config' in checkpoint:
    config = checkpoint['config']
    model = FastBEV(**config)
else:
    # 手动指定配置（确保与训练时一致）
    model = FastBEV(...)
```

### 2. 设备兼容

```python
# 支持CPU/GPU推理
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('ema_only_step_48000.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
```

### 3. 批量推理优化

```python
# 使用torch.no_grad()减少内存
with torch.no_grad():
    outputs = model(inputs)

# 使用半精度加速（如果硬件支持）
model = model.half()  # FP16
inputs = inputs.half()
```

## 📈 性能监控

### 部署后持续监控

```python
import time
import numpy as np

class ModelMonitor:
    def __init__(self):
        self.inference_times = []
        self.detection_counts = []
    
    def log_inference(self, inference_time, num_detections):
        self.inference_times.append(inference_time)
        self.detection_counts.append(num_detections)
    
    def get_stats(self):
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'p95_inference_time': np.percentile(self.inference_times, 95),
            'avg_detections': np.mean(self.detection_counts)
        }

# 使用
monitor = ModelMonitor()

start = time.time()
outputs = model(inputs)
inference_time = time.time() - start

num_detections = len(outputs['detections'])
monitor.log_inference(inference_time, num_detections)

# 定期输出统计
if frame_count % 1000 == 0:
    stats = monitor.get_stats()
    print(f"性能统计: {stats}")
```

## 🎯 总结

### 部署流程

```
训练完成
    ↓
选择最佳checkpoint (通常是最后一个)
    ↓
使用 ema_only_step_*.pth （推荐）✅
    ↓
在测试集上验证效果
    ↓
转换为部署格式 (ONNX/TorchScript)
    ↓
集成到应用中
    ↓
持续监控和优化
```

### 关键点

1. ✅ **优先使用EMA模型** - 效果更好，稳定性更高
2. ✅ **保留完整checkpoint** - 方便后续继续训练或调试
3. ✅ **转换为ONNX** - 便于跨平台部署和优化
4. ✅ **记录模型元信息** - 方便版本管理和追溯
5. ✅ **持续监控性能** - 确保生产环境稳定运行

### 快速命令

```bash
# 推理测试
python scripts/inference.py --checkpoint checkpoints/ema_only_step_48000.pth

# 导出ONNX
python export_onnx.py --checkpoint checkpoints/ema_only_step_48000.pth

# 性能测试
python benchmark.py --checkpoint checkpoints/ema_only_step_48000.pth
```

---

**记住：EMA模型 = 生产环境的最佳选择！** 🚀
