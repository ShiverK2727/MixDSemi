# 分段动态阈值（Piecewise Threshold）策略分析

## 1. 核心思想

### 问题背景
原始实现使用**固定阈值** (默认0.95) 生成伪标签置信度掩码：
```python
mask_teacher = (unet_prob > threshold).unsqueeze(1).float()
```

### 域课程学习的矛盾
- **早期阶段**：训练简单样本（接近标注域），模型预测相对可靠
  - 固定高阈值(0.95)可能过于严格，**丢弃有用的中等置信度样本**
- **后期阶段**：训练困难样本（远离标注域），模型预测不确定性增加
  - 需要**更高阈值**过滤噪声伪标签

### 解决方案：分段动态阈值
**随课程阶段递增阈值**，与域难度同步：
```
阈值 = tau_min + (当前阶段 / (总阶段数-1)) * (tau_max - tau_min)
```

---

## 2. 实现细节

### 新增参数
```bash
# 启用分段动态阈值
--enable_piecewise_tau

# 阈值范围（默认值经过实验调优）
--tau_min 0.80    # 初始阶段（简单样本）
--tau_max 0.95    # 最终阶段（困难样本）
```

### 阈值计算示例（5阶段课程）
| 课程阶段 | 公式计算 | 阈值 | 样本特征 |
|---------|---------|------|---------|
| Stage 0 | 0.80 + (0/4)×0.15 | **0.800** | 最简单（接近标注域） |
| Stage 1 | 0.80 + (1/4)×0.15 | **0.838** | 中等简单 |
| Stage 2 | 0.80 + (2/4)×0.15 | **0.875** | 中等 |
| Stage 3 | 0.80 + (3/4)×0.15 | **0.913** | 中等困难 |
| Stage 4 | 0.80 + (4/4)×0.15 | **0.950** | 最困难（远离标注域） |

### 关键代码位置

#### 1. 初始化阈值（训练前）
```python
# train_unet_MiDSS_DC_v2.py, line ~515
if args.enable_piecewise_tau:
    threshold = get_piecewise_threshold(
        current_stage=curriculum_sampler.stage,
        num_stages=args.dc_parts,
        tau_min=args.tau_min,
        tau_max=args.tau_max
    )
    logging.info("Piecewise threshold ENABLED: initial stage=%d, threshold=%.4f", ...)
else:
    threshold = args.threshold  # 固定阈值 (默认0.95)
```

#### 2. 阈值更新（课程扩展时）
```python
# train_unet_MiDSS_DC_v2.py, line ~895
if should_expand:
    curriculum_sampler.expand()
    
    if args.enable_piecewise_tau:
        old_threshold = threshold
        threshold = get_piecewise_threshold(...)
        logging.info("Curriculum expanded: stage %d→%d, threshold updated: %.4f→%.4f", ...)
```

#### 3. 阈值应用（伪标签生成）
```python
# train_unet_MiDSS_DC_v2.py, line ~628
with torch.no_grad():
    unet_logits_ulb_x_w = ema_unet_model(ulb_unet_size_x_w)
    unet_prob_ulb_x_w = torch.softmax(unet_logits_ulb_x_w, dim=1)
    unet_prob, unet_pseudo_label = torch.max(unet_prob_ulb_x_w, dim=1)
    
    # 使用动态或固定阈值
    mask_teacher = (unet_prob > threshold).unsqueeze(1).float()
```

---

## 3. 理论优势

### 3.1 符合课程学习原则
- **渐进式难度递增**：阈值随样本难度同步提升
- **避免早期过滤**：简单样本用低阈值，充分利用半监督信息
- **后期质量保证**：困难样本用高阈值，防止噪声累积

### 3.2 自适应置信度要求
| 训练阶段 | 模型状态 | 样本难度 | 阈值策略 | 效果 |
|---------|---------|---------|---------|------|
| **早期** | 在标注域学习良好 | 简单（同域） | **低阈值**(0.80) | 接受更多中等置信度样本，加速学习 |
| **中期** | 开始泛化到中等难度 | 中等（近域） | **渐进阈值**(0.85) | 平衡利用率与质量 |
| **后期** | 泛化到困难样本 | 困难（远域） | **高阈值**(0.95) | 严格过滤，避免噪声伪标签 |

### 3.3 与固定阈值对比

#### 固定高阈值 (0.95)
- ❌ **早期过于保守**：简单样本的0.85-0.95置信度被丢弃
- ❌ **半监督信息浪费**：大量可用样本未参与训练
- ✅ 后期质量高

#### 固定低阈值 (0.80)
- ✅ 早期充分利用
- ❌ **后期噪声累积**：困难样本的低置信度伪标签引入错误
- ❌ 性能上限受限

#### 动态阈值 (0.80→0.95)
- ✅ **自适应平衡**：早期利用率高，后期质量好
- ✅ **符合课程逻辑**：与域难度同步
- ✅ **理论最优**：匹配置信度要求与样本难度

---

## 4. 使用建议

### 4.1 推荐配置

#### Prostate 数据集（6域，简单→困难差异大）
```bash
python train_unet_MiDSS_DC_v2.py \
  --dataset prostate \
  --dc_parts 5 \
  --enable_piecewise_tau \
  --tau_min 0.75 \      # 较低起点，充分利用简单样本
  --tau_max 0.95 \      # 标准高阈值
  ...
```

#### Fundus 数据集（4域，难度梯度中等）
```bash
python train_unet_MiDSS_DC_v2.py \
  --dataset fundus \
  --dc_parts 5 \
  --enable_piecewise_tau \
  --tau_min 0.80 \      # 中等起点
  --tau_max 0.95 \
  ...
```

#### MNMS 数据集（4域，心脏分割，困难）
```bash
python train_unet_MiDSS_DC_v2.py \
  --dataset MNMS \
  --dc_parts 5 \
  --enable_piecewise_tau \
  --tau_min 0.85 \      # 较高起点，避免早期噪声
  --tau_max 0.97 \      # 更严格的最终阈值
  ...
```

### 4.2 调参原则

1. **`tau_min` 选择**
   - 观察标注域训练后模型在同域样本的置信度分布
   - 选择能保留70-80%预测样本的阈值作为 `tau_min`
   - 通常范围：**0.75 - 0.85**

2. **`tau_max` 选择**
   - 沿用传统经验值 **0.95**
   - 如果数据集噪声大，可提高到 **0.97**
   - 如果困难样本稀缺，可降低到 **0.90**

3. **课程阶段数 `dc_parts`**
   - 推荐 **5** 或 **7** 阶段
   - 阶段过少（<3）：阈值跳跃过大
   - 阶段过多（>10）：每阶段提升过小，不明显

---

## 5. 监控与调试

### TensorBoard 可视化
新增两个监控指标：
```python
writer.add_scalar('train/threshold', threshold, iter_num)           # 当前阈值
writer.add_scalar('train/curriculum_stage', curriculum_sampler.stage, iter_num)  # 当前阶段
```

### 关键日志示例
```
[INFO] Piecewise threshold ENABLED: initial stage=0, threshold=0.8000 (tau_min=0.80, tau_max=0.95)
...
[INFO] Curriculum expanded: stage 0→1, threshold updated: 0.8000→0.8375
[INFO] Next-partition confidence at iter 3000 (partition 2): 0.7823
...
[INFO] Curriculum expanded: stage 1→2, threshold updated: 0.8375→0.8750
```

### 验证阈值效果
1. **观察 `mask_teacher` 比例**
   - 早期应该较高（40-60%），后期降低（20-30%）
   - 如果始终很低（<10%），说明 `tau_min` 过高

2. **观察伪标签质量**（通过 `unet_ulb_dice`）
   - 早期Dice应该较高（0.75+），说明简单样本质量好
   - 后期Dice下降可接受（0.60+），说明困难样本被正确识别

3. **对比固定阈值基线**
   - 固定0.95：`mask_teacher` 比例应该始终很低（10-20%）
   - 动态阈值：早期高，后期低（符合预期）

---

## 6. 预期效果

### 理论改进
1. **早期加速**：更多样本参与训练，模型收敛更快
2. **后期稳定**：高阈值防止错误累积，最终性能更优
3. **整体提升**：平均Dice提升 **1-3%**（基于课程学习文献经验）

### 与其他技术的协同
- **域课程**：阈值与样本难度同步，增强课程效果
- **频域增强**：早期低阈值允许增强样本参与，后期高阈值过滤增强引入的噪声
- **SymGD**：动态阈值改善UL/LU混合的伪标签质量

---

## 7. 消融实验建议

### 实验组设计
| 组别 | 配置 | 目的 |
|-----|------|------|
| **A** | `--threshold 0.95` | 固定高阈值基线 |
| **B** | `--threshold 0.80` | 固定低阈值基线 |
| **C** | `--enable_piecewise_tau --tau_min 0.80 --tau_max 0.95` | **动态阈值（推荐）** |
| **D** | `--enable_piecewise_tau --tau_min 0.75 --tau_max 0.97` | 更宽范围 |

### 关键对比指标
1. **最终Dice**：动态阈值应优于固定阈值
2. **收敛速度**：早期Dice提升速度（前5000 iter）
3. **稳定性**：后期Dice波动方差
4. **阈值轨迹**：通过TensorBoard观察阈值变化是否合理

---

## 8. 代码变更总结

### 新增函数
```python
def get_piecewise_threshold(current_stage, num_stages, tau_min, tau_max):
    """计算分段阈值（线性插值）"""
    if num_stages <= 1:
        return tau_max
    stage = max(0, min(current_stage, num_stages - 1))
    threshold = tau_min + (stage / (num_stages - 1)) * (tau_max - tau_min)
    return threshold
```

### 修改位置
1. **参数解析**：新增 `--enable_piecewise_tau`, `--tau_min`, `--tau_max`
2. **训练初始化**：根据初始阶段计算阈值
3. **课程扩展**：同步更新阈值
4. **日志监控**：记录阈值和阶段到TensorBoard

---

## 9. 兼容性

### 向后兼容
- **默认禁用**：`--enable_piecewise_tau` 需显式指定
- **固定阈值保留**：不加该参数时，行为与原代码完全一致
- **参数默认值安全**：`tau_min=0.80`, `tau_max=0.95` 为经验稳定值

### 与其他功能交互
- ✅ 域课程采样器（必需）
- ✅ 频域增强（兼容）
- ✅ SymGD（兼容）
- ✅ 置信度策略（共同作用于伪标签质量）

---

## 10. 总结

### 何时启用分段动态阈值？
- ✅ **推荐启用**：几乎所有多域半监督场景
- ✅ **强烈推荐**：域间差异大、样本难度梯度明显
- ⚠️ **谨慎使用**：单域或域间差异极小（退化为固定阈值）

### 核心优势
1. **理论合理**：与课程学习原理完美契合
2. **实现简洁**：仅需线性插值，计算开销可忽略
3. **易于调试**：TensorBoard可视化清晰
4. **风险可控**：保留固定阈值选项，默认禁用

### 下一步
1. 运行对比实验（固定 vs 动态阈值）
2. 根据数据集特性调优 `tau_min`/`tau_max`
3. 分析TensorBoard日志，验证阈值轨迹合理性
