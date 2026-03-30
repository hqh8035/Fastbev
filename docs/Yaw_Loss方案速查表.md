# Yaw Loss 方案速查表

## 🎯 快速选择

### 当前问题
- ✅ Cosine Similarity Loss训练震荡严重
- ✅ 蓝绿色曲线(step 48000: 0.2577)比原始L1(0.2468)更差

### 推荐顺序
1. **首选**: `hrx/feat/yaw-loss-periodic-l1` - 最完备，理论最佳
2. **备选**: `hrx/feat/yaw-loss-smooth-l1` - 最简单，稳定性好  
3. **探索**: `hrx/feat/yaw-loss-cosine-optimized` - 数值优化

---

## 📋 三分支对比

| | Smooth L1 | PeriodicL1 | 优化Cosine |
|---|-----------|-----------|-----------|
| **分支** | `hrx/feat/yaw-loss-smooth-l1` | `hrx/feat/yaw-loss-periodic-l1` | `hrx/feat/yaw-loss-cosine-optimized` |
| **核心** | L2→L1平滑过渡 | 主项+周期约束 | 数值稳定化 |
| **公式** | `smooth_l1(pred, gt)` | `smooth_l1 + λ·periodic` | `1 - cos(pred, gt)` 优化版 |
| **梯度** | ✅ 平滑 | ✅ 平滑 | ⚠️ 非线性 |
| **周期性** | ❌ 不处理 | ✅ 显式处理 | ✅ 隐式处理 |
| **复杂度** | ⭐ 简单 | ⭐⭐ 中等 | ⭐ 简单 |
| **稳定性预期** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **适用场景** | 通用yaw预测 | 跨±π边界场景 | 保持原架构 |

---

## 🚀 一键训练

```bash
# 方案1: Smooth L1
git checkout hrx/feat/yaw-loss-smooth-l1
bash shell_scripts/01_train.sh

# 方案2: PeriodicL1 (推荐)
git checkout hrx/feat/yaw-loss-periodic-l1
bash shell_scripts/01_train.sh

# 方案3: 优化Cosine
git checkout hrx/feat/yaw-loss-cosine-optimized
bash shell_scripts/01_train.sh
```

---

## 📊 监控指标

### TensorBoard关键曲线
```bash
tensorboard --logdir checkpoints_logs --port 6006
```

**必看**:
- `Loss/yaw_loss` - 主要对比（震荡+收敛）
- `Loss/train_step` - 整体稳定性

**PeriodicL1专属**:
- `Loss/yaw_regression_loss` - 主项
- `Loss/yaw_periodic_loss` - 辅助项

---

## 🔧 核心代码差异

### Smooth L1
```python
# utils/loss.py
yaw_loss = smooth_l1_loss_for_yaw(pred, gt, mask, beta=1.0)
```

### PeriodicL1  
```python
# utils/loss.py
yaw_loss, reg_loss, periodic_loss = periodic_l1_loss_for_yaw(
    pred, gt, mask, beta=1.0, lambda_periodic=0.5
)
```

### 优化Cosine
```python
# model/fastbev.py
yaw_pred_norm = F.normalize(yaw_pred + eps, p=2, dim=2)  # +eps稳定性
cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # 裁剪
yaw_loss_weight = 1.0  # 降权重 (原2.0)
```

---

## 🎓 理论对比

### 问题1: L1 Loss零点梯度恒定
- ❌ **原L1**: 梯度=±1，精细调整困难
- ✅ **Smooth L1**: 零点附近梯度=x/beta，平滑过渡
- ✅ **PeriodicL1**: 继承Smooth L1优势
- ⚠️ **Cosine**: 非线性梯度，可能不稳定

### 问题2: 角度2π周期性
- ❌ **L1/Smooth L1**: 不处理，跨±π会错误惩罚
- ✅ **PeriodicL1**: 辅助项显式处理，映射到[-π,π]
- ⚠️ **Cosine**: 理论上处理，但实际震荡

### 问题3: 数值稳定性
- ✅ **Smooth L1**: PyTorch内置，稳定
- ✅ **PeriodicL1**: 加eps，稳定
- ⚠️ **原Cosine**: 可能重复归一化
- ✅ **优化Cosine**: 加eps+裁剪+检查GT

---

## 💡 快速诊断

### 症状 → 方案

| 症状 | 推荐方案 | 理由 |
|------|---------|------|
| 震荡严重 | Smooth L1 | 平滑梯度 |
| 跨±π跳变 | PeriodicL1 | 周期约束 |
| NaN/Inf | 优化Cosine | 数值稳定 |
| 收敛慢 | 调大beta | 更激进 |
| 仍震荡 | 降yaw权重 | 减少影响 |

---

## 📝 实验结果速记

| 方案 | Loss@48k | 震荡 | 速度 | 推荐 |
|------|---------|------|------|------|
| 原L1 | 0.2468 | 中 | 中 | ⭐⭐ |
| 原Cosine | 0.2577 | 高⚠️ | 慢 | ⭐ |
| Smooth L1 | ___ | ___ | ___ | ___ |
| PeriodicL1 | ___ | ___ | ___ | ___ |
| 优化Cosine | ___ | ___ | ___ | ___ |

---

## ⚙️ 快速调参

```python
# Smooth L1
beta = 1.0  # 0.5→更平滑  2.0→更激进

# PeriodicL1  
lambda_periodic = 0.5  # 0.3→轻约束  1.0→强约束

# 通用
yaw_loss_weight = 2.0  # 降到1.0减震荡
```

---

## 📞 完整文档

详细说明请参考: [`Yaw_Loss实验指南.md`](./Yaw_Loss实验指南.md)

---

**记住**: PeriodicL1理论最完备，Smooth L1最简单有效！🎯
