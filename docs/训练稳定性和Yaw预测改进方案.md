# 训练稳定性和Yaw预测改进方案

## 问题分析

### 问题1: 训练震荡不稳定
- **原因**: 学习率过高(5e-4)、梯度裁剪阈值过大(5.0)、缺乏warmup
- **表现**: Loss曲线震荡剧烈，难以收敛

### 问题2: Yaw预测不准确
- **原因**: 使用L1 Loss无法保证sin/cos的单位圆约束
- **表现**: 帧间朝向跳变，视觉效果差

## 改进方案

### 1. Loss权重优化

基于实际训练数据调整权重：

```python
['center_loss', 'offset_loss', 'height_loss', 'lwh_loss', 'yaw_loss']
[15.0,          2.0,           1.5,           0.3,        2.0]
```

**调整后的Loss平衡**:
- center_loss: 0.015 × 15 = 0.225 (主导)
- offset_loss: 0.17 × 2 = 0.34 (主导)
- height_loss: 0.06 × 1.5 = 0.09 (辅助)
- lwh_loss: 0.06 × 0.3 = 0.018 (辅助)
- yaw_loss: 0.09 × 2 = 0.18 (重要)

### 2. Yaw Loss改进

**从L1 Loss改为Cosine Similarity Loss**:

```python
# 旧方案: L1 Loss
yaw_loss = F.l1_loss(pred, gt)

# 新方案: Cosine Similarity Loss
yaw_pred_norm = F.normalize(pred, p=2, dim=2)
yaw_gt_norm = F.normalize(gt, p=2, dim=2)
cos_sim = (yaw_pred_norm * yaw_gt_norm).sum(dim=2)
yaw_loss = 1.0 - cos_sim
```

**优势**:
- ✅ 自动保证预测值在单位圆上
- ✅ 角度差异的直接度量
- ✅ 梯度更稳定，收敛更快
- ✅ 帧间朝向更平滑

### 3. 训练稳定性改进

#### 3.1 降低学习率
```python
LR: 5e-4 → 3e-4
```

#### 3.2 增加正则化
```python
WEIGHT_DECAY: 5e-5 → 1e-4
```

#### 3.3 降低梯度裁剪
```python
CLIP_GRAD_NORM: 5.0 → 2.0
```

#### 3.4 添加Warmup
```python
WARMUP_ITERS: 1000  # 前1000步线性增加学习率
```

#### 3.5 改用Cosine Annealing调度器
```python
# 旧: OneCycleLR (震荡较大)
# 新: CosineAnnealingLR with Warmup (更平滑)
```

#### 3.6 添加EMA (Exponential Moving Average)
```python
EMA_DECAY: 0.9999
```

**EMA优势**:
- ✅ 平滑模型参数，减少震荡
- ✅ 提升模型泛化能力
- ✅ 推理时使用EMA模型效果更好
- ✅ 自动保存`ema_only_step_*.pth`用于推理

## 预期效果

### 训练稳定性
- Loss曲线更平滑，震荡幅度减小50%+
- 收敛速度提升，更容易找到最优解
- 梯度爆炸/消失问题显著减少

### Yaw预测精度
- 角度预测误差降低30-50%
- 帧间朝向连续性显著提升
- 视觉效果更自然流畅

### 整体性能
- 检测精度提升2-5%
- 训练时间可能略微增加(+5%)，但收敛更稳定
- 推理使用EMA模型，效果优于普通模型

## 使用建议

### 训练
```bash
# 使用新配置训练
python scripts/train.py --config fastbev_v6 --distributed
```

### 推理
```bash
# 优先使用EMA模型
python scripts/inference.py --checkpoint checkpoints/ema_only_step_48000.pth
```

### 监控
重点关注TensorBoard中的：
- `Loss/yaw_loss`: 应该更平滑，收敛到0.05-0.15
- `Loss/train_step`: 震荡幅度应该明显减小
- `LR`: 观察warmup和cosine annealing曲线

### 微调
如果仍有问题，可以尝试：
- 进一步降低学习率: 3e-4 → 2e-4
- 增加warmup: 1000 → 2000
- 调整EMA decay: 0.9999 → 0.999 (更激进的平滑)

## 技术细节

### Cosine Similarity Loss推导
对于角度θ，表示为(sin(θ), cos(θ)):
- 预测: (s_p, c_p)
- 真值: (s_g, c_g)

Cosine similarity:
```
cos(θ_diff) = s_p*s_g + c_p*c_g
```

Loss:
```
L = 1 - cos(θ_diff) ∈ [0, 2]
```

当预测完全正确时，cos(θ_diff)=1，Loss=0
当预测相反时，cos(θ_diff)=-1，Loss=2

### EMA更新公式
```python
θ_ema = decay * θ_ema + (1 - decay) * θ_model
```

decay=0.9999意味着：
- 99.99%保留历史
- 0.01%采用新值
- 相当于对最近10000步取平均

## 文件修改清单

- ✅ `model/fastbev.py`: 更新loss权重，改进yaw loss
- ✅ `configs/config_fastbev_v6.py`: 优化训练超参数
- ✅ `scripts/train.py`: 添加warmup、EMA、cosine scheduler

## 版本历史

- v1.0 (2026-01-21): 初始版本，包含所有改进

---

# 改进方案对比总结

## 📊 改进前 vs 改进后

### Loss权重对比

| Loss类型 | 旧权重 | 新权重 | 原始值 | 旧加权值 | 新加权值 | 变化 |
|---------|-------|-------|--------|---------|---------|------|
| center_loss | 1.0 | **15.0** | ~0.015 | 0.015 | **0.225** | ⬆️ 15x |
| offset_loss | 1.0 | **2.0** | ~0.17 | 0.17 | **0.34** | ⬆️ 2x |
| height_loss | 1.0 | **1.5** | ~0.06 | 0.06 | **0.09** | ⬆️ 1.5x |
| lwh_loss | 1.0 | **0.3** | ~0.06 | 0.06 | **0.018** | ⬇️ 0.3x |
| yaw_loss | 1.0 | **2.0** | ~0.09 | 0.09 | **0.18** | ⬆️ 2x |

### Yaw Loss计算方式对比

#### ❌ 旧方案: L1 Loss
```python
yaw_loss = F.l1_loss(pred, gt)
```
**问题**:
- 无法保证单位圆约束
- 梯度不稳定
- 帧间跳变严重

#### ✅ 新方案: Cosine Similarity Loss
```python
yaw_pred_norm = F.normalize(pred, p=2, dim=2)
yaw_gt_norm = F.normalize(gt, p=2, dim=2)
cos_sim = (yaw_pred_norm * yaw_gt_norm).sum(dim=2)
yaw_loss = 1.0 - cos_sim
```
**优势**:
- ✅ 自动单位圆约束
- ✅ 梯度稳定
- ✅ 帧间平滑

### 训练配置对比

| 参数 | 旧值 | 新值 | 说明 |
|-----|------|------|------|
| 学习率 | 5e-4 | **3e-4** | 降低40%，更稳定 |
| 权重衰减 | 5e-5 | **1e-4** | 增加正则化 |
| 梯度裁剪 | 5.0 | **2.0** | 防止梯度爆炸 |
| Warmup | ❌ 无 | **✅ 1000步** | 平滑启动 |
| 调度器 | OneCycleLR | **CosineAnnealingLR** | 更平滑 |
| EMA | ❌ 无 | **✅ 0.9999** | 参数平滑 |

## 🎯 预期改进效果

### 1. 训练稳定性
- **Loss震荡**: 减少 **50-70%**
- **收敛速度**: 提升 **20-30%**
- **梯度稳定**: 显著改善

### 2. Yaw预测精度
- **角度误差**: 降低 **30-50%**
- **帧间连续性**: 显著提升
- **视觉效果**: 更流畅自然

### 3. 整体检测性能
- **AP提升**: 预计 **+2-5%**
- **误检率**: 降低 **10-20%**
- **推理稳定性**: 显著提升

## 📈 训练曲线预期变化

### 旧训练曲线特征
```
Loss/train_step:  震荡剧烈 ▲▼▲▼▲▼
Loss/center_loss: 几乎为0，被忽略
Loss/yaw_loss:    不稳定，跳变大
```

### 新训练曲线预期
```
Loss/train_step:  平滑下降 ＼＼＼
Loss/center_loss: 明显贡献，稳定下降
Loss/yaw_loss:    平滑收敛，稳定
```

## 🔧 使用方式

### 训练
```bash
# 单卡训练
python scripts/train.py --config fastbev_v6

# 多卡训练
python scripts/train.py --config fastbev_v6 --distributed
```

### 推理（使用EMA模型）
```bash
# 优先使用EMA模型，效果更好
python scripts/inference.py --checkpoint checkpoints/ema_only_step_48000.pth
```

### 监控
```bash
# 启动TensorBoard
tensorboard --logdir checkpoints_logs

# 重点观察:
# 1. Loss/train_step - 应该更平滑
# 2. Loss/center_loss - 应该有明显贡献(0.1-0.3)
# 3. Loss/yaw_loss - 应该平滑收敛(0.05-0.15)
# 4. LR - 观察warmup和cosine曲线
```

## ⚠️ 注意事项

### 1. 学习率调整
如果训练仍不稳定，可以进一步降低学习率：
```python
_C.TRAIN.LR = 2e-4  # 从3e-4降到2e-4
```

### 2. Warmup调整
如果初期仍震荡，可以增加warmup步数：
```python
_C.TRAIN.WARMUP_ITERS = 2000  # 从1000增到2000
```

### 3. EMA调整
如果需要更激进的平滑：
```python
_C.TRAIN.EMA_DECAY = 0.999  # 从0.9999降到0.999
```

### 4. 权重微调
根据实际训练效果，可以微调loss权重：
- center_loss太小: 15.0 → 20.0
- yaw_loss仍不稳定: 2.0 → 3.0
- lwh_loss过小: 0.3 → 0.5

## 📝 技术亮点

### 1. Cosine Similarity Loss
- 理论基础扎实，符合角度表示的几何约束
- 梯度性质好，不会出现L1的不可导点
- 自然处理周期性，避免180°跳变

### 2. EMA机制
- 相当于对最近10000步模型取平均
- 推理时使用EMA模型，效果通常优于最后一步模型
- 几乎无额外计算开销

### 3. Warmup + Cosine Annealing
- Warmup避免初期梯度爆炸
- Cosine平滑衰减，避免突变
- 比OneCycleLR更稳定

## 🚀 下一步优化方向

1. **数据增强**: 可以尝试更强的增强提升泛化
2. **多尺度训练**: 提升不同距离的检测效果
3. **Focal Loss调参**: 微调alpha和gamma
4. **后处理优化**: NMS阈值、置信度阈值等

## 📚 参考资料

- Focal Loss: https://arxiv.org/abs/1708.02002
- EMA: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
- Cosine Annealing: https://arxiv.org/abs/1608.03983
