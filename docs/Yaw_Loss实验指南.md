# Yaw Loss 实验对比指南

## 📋 概述

本文档提供三个yaw loss改进方案的实验对比指南。这些方案旨在解决当前Cosine Similarity Loss训练震荡严重的问题。

## 🌿 实验分支

已创建三个实验分支，每个分支实现不同的yaw loss方案：

| 分支名 | Loss类型 | 主要特点 | 预期效果 |
|--------|---------|---------|---------|
| `hrx/feat/yaw-loss-smooth-l1` | Smooth L1 | 零点附近平滑梯度 | ⭐⭐⭐ 稳定性好 |
| `hrx/feat/yaw-loss-periodic-l1` | Smooth L1 + 周期约束 | 处理2π边界 | ⭐⭐⭐⭐ 最完备 |
| `hrx/feat/yaw-loss-cosine-optimized` | 优化Cosine | 数值稳定+降权重 | ⭐⭐ 探索优化 |

## 🔬 方案详解

### 方案1: Smooth L1 Loss

**分支**: `hrx/feat/yaw-loss-smooth-l1`

**核心思想**:
```python
           { 0.5 * x^2 / beta,  if |x| < beta
loss(x) = {
           { |x| - 0.5 * beta,  otherwise
```

**优势**:
- ✅ 零点附近使用L2平滑梯度，精细调整
- ✅ 远离零点使用L1稳定梯度，快速收敛
- ✅ 全局可导，无L1的不连续点
- ✅ 实现简单，PyTorch内置支持

**参数**:
- `beta=1.0`: L1/L2切换点（默认）

**预期Loss值**: 0.15-0.20（比当前L1的0.25更低）

---

### 方案2: PeriodicL1 Loss（推荐）

**分支**: `hrx/feat/yaw-loss-periodic-l1`

**核心思想**:
```
Loss = SmoothL1(ŷ, y_gt) + λ · PeriodicL1(θ̂, θ_gt)
      └─主项：sin/cos回归    └─辅助项：角度周期约束
```

**双项设计**:

1. **主项** - Smooth L1回归sin/cos:
   ```python
   regression_loss = F.smooth_l1_loss(pred, target)
   ```

2. **辅助项** - 处理角度2π周期性:
   ```python
   # 恢复角度
   pred_angle = torch.atan2(sin_pred, cos_pred)
   target_angle = torch.atan2(sin_gt, cos_gt)
   
   # 归一化角度差到[-π, π]
   angle_diff = pred_angle - target_angle
   angle_diff_norm = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
   
   # 计算周期性loss
   periodic_loss = F.smooth_l1_loss(angle_diff_norm, 0)
   ```

**为什么需要辅助项？**

问题场景：
- 预测角度: +3.1 rad (接近+π)
- 真实角度: -3.1 rad (接近-π)
- 直接差值: 6.2 rad (看起来差很大！)
- 实际差异: ~0.08 rad (角度上很接近)

辅助项通过角度归一化，将6.2 rad正确映射为~0.08 rad，避免跨边界的错误惩罚。

**参数**:
- `beta=1.0`: Smooth L1转换点
- `lambda_periodic=0.5`: 辅助项权重（可在config中调整）

**监控指标**:
- `Loss/yaw_loss`: 总loss
- `Loss/yaw_regression_loss`: 主项（sin/cos回归）
- `Loss/yaw_periodic_loss`: 辅助项（周期约束）

**预期Loss值**: 主项0.15-0.20 + 辅助项0.05-0.10

---

### 方案3: 优化Cosine Similarity

**分支**: `hrx/feat/yaw-loss-cosine-optimized`

**优化措施**:

1. **数值稳定性**:
   ```python
   eps = 1e-8
   yaw_pred_stable = yaw_pred + eps
   yaw_pred_norm = F.normalize(yaw_pred_stable, p=2, dim=2)
   ```

2. **避免重复归一化**:
   ```python
   gt_norm = torch.sqrt((yaw_gt ** 2).sum(dim=2, keepdim=True))
   if torch.allclose(gt_norm, 1.0, atol=1e-2):
       yaw_gt_norm = yaw_gt  # GT已归一化，直接使用
   ```

3. **Cosine范围裁剪**:
   ```python
   cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)
   ```

4. **降低权重**:
   - yaw_loss权重: 2.0 → 1.0

**预期效果**: 如果震荡是数值问题导致的，应该会显著改善

---

## 🚀 训练实验步骤

### 1. 准备环境

确保在正确的工作目录：
```bash
cd /perception/projects_data/sixents/train/nova_fastbev
```

### 2. 并行训练三个方案

**方案1 - Smooth L1**:
```bash
git checkout hrx/feat/yaw-loss-smooth-l1
bash shell_scripts/01_train.sh
```

**方案2 - PeriodicL1**:
```bash
git checkout hrx/feat/yaw-loss-periodic-l1
bash shell_scripts/01_train.sh
```

**方案3 - 优化Cosine**:
```bash
git checkout hrx/feat/yaw-loss-cosine-optimized
bash shell_scripts/01_train.sh
```

### 3. 监控训练

启动TensorBoard观察三个实验：
```bash
tensorboard --logdir checkpoints_logs --port 6006
```

**关键曲线**:
- `Loss/yaw_loss`: 主要对比指标
  - 观察平滑度（震荡幅度）
  - 观察收敛值（最终loss大小）
  
- `Loss/train_step`: 整体训练稳定性

- `Loss/yaw_regression_loss` (仅PeriodicL1): 主项loss
- `Loss/yaw_periodic_loss` (仅PeriodicL1): 辅助项loss

### 4. 评估标准

| 指标 | 说明 | 目标 |
|------|------|------|
| **震荡幅度** | loss曲线的标准差 | 越小越好 |
| **收敛值** | 48000步时的loss值 | 越低越好 |
| **平滑度** | 曲线视觉平滑程度 | 越平滑越好 |
| **实际效果** | 可视化yaw预测准确度 | 帧间连续性好 |

### 5. 预期结果对比

基于理论分析的预期排名：

**稳定性排名**:
1. 🥇 PeriodicL1 (双项约束，最稳定)
2. 🥈 Smooth L1 (平滑梯度)
3. 🥉 优化Cosine (取决于是否是数值问题)

**收敛效果排名**:
1. 🥇 PeriodicL1 (理论最完备)
2. 🥈 Smooth L1 (简单有效)
3. 🥉 优化Cosine (改进版)

**实现复杂度**:
1. 🥇 Smooth L1 (最简单)
2. 🥈 优化Cosine (中等)
3. 🥉 PeriodicL1 (稍复杂，但值得)

## 📊 结果对比表格模板

训练完成后，填写以下表格：

| 方案 | 最终Loss | 震荡幅度(std) | 收敛速度 | 视觉效果 | 推荐度 |
|------|---------|-------------|---------|---------|--------|
| **原始L1** | 0.2468 | 高 | 中 | 中 | ⭐⭐ |
| **原始Cosine** | 0.2577 | 很高⚠️ | 慢 | 中 | ⭐ |
| **Smooth L1** | _待填_ | _待填_ | _待填_ | _待填_ | _待评_ |
| **PeriodicL1** | _待填_ | _待填_ | _待填_ | _待填_ | _待评_ |
| **优化Cosine** | _待填_ | _待填_ | _待填_ | _待填_ | _待评_ |

## 🎯 决策建议

根据实验结果选择最佳方案：

### 场景1: PeriodicL1表现最好（预期）
✅ **推荐**: 将PeriodicL1合并到主分支
- 理论最完备
- 同时优化回归和角度连续性
- 适合长期使用

### 场景2: Smooth L1已经足够好
✅ **推荐**: 使用Smooth L1
- 实现简单
- 稳定性好
- 如果效果已达标，无需更复杂方案

### 场景3: 优化Cosine意外表现很好
✅ **推荐**: 使用优化Cosine
- 保持原有架构
- 仅需微调
- 证明原问题确实是数值稳定性

## 🔧 参数调优

如果某个方案效果不理想，可以调整：

### Smooth L1 调优
```python
# utils/loss.py - smooth_l1_loss_for_yaw()
beta = 1.0  # 尝试 0.5 (更平滑) 或 2.0 (更激进)
```

### PeriodicL1 调优
```python
# configs/config_fastbev_v6.py
_C.MODEL.FASTBEV.YAW_LOSS.BETA = 1.0  # Smooth L1参数
_C.MODEL.FASTBEV.YAW_LOSS.LAMBDA_PERIODIC = 0.5  # 辅助项权重

# 调优建议：
# - lambda=0.3: 轻微约束
# - lambda=0.5: 平衡约束（默认）
# - lambda=1.0: 强约束
```

### Cosine 调优
```python
# model/fastbev.py - 权重调整
yaw_loss_weight = 1.0  # 尝试 0.5 (更低) 或 1.5 (适中)
```

## 📝 实验日志

建议记录实验关键信息：

```markdown
### 实验日期: 2026-01-XX

**分支**: hrx/feat/yaw-loss-xxx
**训练步数**: 48000
**配置**: config_fastbev_v6

**结果**:
- 最终yaw_loss: X.XXX
- 震荡幅度: [描述]
- 曲线图: [截图]
- 可视化效果: [描述]

**结论**:
[是否推荐使用此方案？为什么？]
```

## 🔍 故障排查

### 问题1: Loss出现NaN
**可能原因**:
- 学习率过高
- 数值不稳定

**解决方案**:
- 降低学习率: 3e-4 → 2e-4
- 检查GT数据质量
- 增加eps稳定项

### 问题2: Loss仍然震荡
**可能原因**:
- 权重过大
- batch size过小

**解决方案**:
- 进一步降低yaw_loss权重
- 增加warmup步数
- 使用更大batch size

### 问题3: 收敛过慢
**可能原因**:
- beta过小
- lambda过大

**解决方案**:
- 增大beta: 1.0 → 2.0
- 降低lambda: 0.5 → 0.3

## 📚 参考资料

1. **Smooth L1 Loss**: 
   - Paper: Fast R-CNN (Ross Girshick, 2015)
   - PyTorch文档: `torch.nn.functional.smooth_l1_loss`

2. **角度周期性处理**:
   - 关键技巧: `atan2(sin(diff), cos(diff))` 归一化到[-π, π]
   - 适用于所有需要处理角度差的场景

3. **Cosine Similarity Loss**:
   - 理论基础: 向量夹角度量
   - 适合单位向量表示的目标（如归一化后的sin/cos）

## ✅ 实验检查清单

训练前确认：
- [ ] 已切换到正确的实验分支
- [ ] 配置文件使用config_fastbev_v6
- [ ] TensorBoard已启动
- [ ] 有足够的存储空间保存checkpoint

训练中监控：
- [ ] Loss曲线正常下降（无NaN）
- [ ] yaw_loss震荡幅度合理
- [ ] GPU利用率正常

训练后分析：
- [ ] 记录最终loss值
- [ ] 保存TensorBoard曲线截图
- [ ] 运行可视化验证yaw预测效果
- [ ] 填写结果对比表格

---

## 🎓 总结

三个方案的设计思路：

1. **Smooth L1**: 从梯度角度改进L1 Loss的零点不平滑问题
2. **PeriodicL1**: 从角度周期性角度添加显式约束
3. **优化Cosine**: 从数值稳定性角度优化现有方案

每个方案都针对不同的问题假设，通过实验可以找出真正的瓶颈所在。

**预测**: PeriodicL1最有可能成为最终方案，因为它同时解决了梯度平滑性和角度周期性两个问题。

祝实验顺利！🚀
