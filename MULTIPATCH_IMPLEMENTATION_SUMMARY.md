# 多数据集 Multi-Patch 实现总结

## ✅ 完成的工作

### 1. 数据集类添加到 `pretrain_dataloader.py`
已将以下三个数据集类添加到 `/app/MixDSemi/SynFoCLIP/code/dataloaders/pretrain_dataloader.py`:
- **FundusSegmentation** (视网膜眼底图像)
- **MNMSSegmentation** (心脏MRI)
- **BUSISegmentation** (乳腺超声)

所有类都支持 `patch_sampler` 参数,与 ProstateSegmentation 保持一致。

### 2. 标签映射验证 ✓

所有数据集的标签映射关系已验证正确:

| 数据集 | 原始格式 | _normalize_label 后 | ToTensor 后 | 符合要求 |
|--------|----------|---------------------|-------------|----------|
| **Prostate** | 0=FG, 255=BG | 255=FG, 0=BG (反转) | [0.0, 255.0] | ✅ FG>0, BG=0 |
| **Fundus** | 0=BG, 128=cup, 255=disc | 无变化 | [0.0, 128.0, 255.0] | ✅ FG>0, BG=0 |
| **MNMS** | RGB (R=LV, G=MYO, B=RV) | 0=BG, 1=LV, 2=MYO, 3=RV | [0.0, 1.0, 2.0, 3.0] | ✅ FG>0, BG=0 |
| **BUSI** | 0=BG, 255=tumor | 无变化 | [0.0, 255.0] | ✅ FG>0, BG=0 |

**关键点**:
- `RandomPatchSamplerWithClass` 要求: 背景=0, 前景>0
- 所有数据集的 `_normalize_label()` 方法都正确实现了此要求
- `ToTensor` **不会**将 mask 归一化到 [0,1],保持原始值范围
- MNMS 的 RGB mask 被正确转换为类别索引 (0/1/2/3)

### 3. 可视化脚本更新

#### 修改内容:
1. **Import 修正**: 所有数据集类现在从 `pretrain_dataloader.py` 导入
   ```python
   from dataloaders.pretrain_dataloader import (
       ProstateSegmentation, 
       FundusSegmentation, 
       MNMSSegmentation, 
       BUSISegmentation
   )
   ```

2. **输出目录按数据集分类**: 避免覆盖
   ```python
   # 修改前: out_dir = '../../results/patch_batch_viz'
   # 修改后: out_dir = '../../results/patch_batch_viz/{dataset_name}'
   ```
   - Prostate → `/results/patch_batch_viz/prostate/`
   - Fundus → `/results/patch_batch_viz/fundus/`
   - MNMS → `/results/patch_batch_viz/MNMS/`
   - BUSI → `/results/patch_batch_viz/BUSI/`

3. **Mask 可视化标注**: 添加"(白=FG)"标签,明确前景为白色

### 4. 测试验证 ✓

所有数据集已通过测试:

```bash
# 测试命令
python test_pretrain_dataloader.py

# 结果
Prostate    : ✓ PASS
Fundus      : ✓ PASS
MNMS        : ✓ PASS
BUSI        : ✓ PASS
```

可视化测试:
```bash
python visualize_patches_batch.py --dataset prostate --batch-size 1
python visualize_patches_batch.py --dataset fundus --batch-size 1
python visualize_patches_batch.py --dataset MNMS --batch-size 1
python visualize_patches_batch.py --dataset BUSI --batch-size 1
```

## 📋 文件清单

### 修改的文件:
1. `/app/MixDSemi/SynFoCLIP/code/dataloaders/pretrain_dataloader.py`
   - 添加 FundusSegmentation (lines 277-498)
   - 添加 MNMSSegmentation (lines 501-719)
   - 添加 BUSISegmentation (lines 722-975)

2. `/app/MixDSemi/SynFoCLIP/code/visualize_patches_batch.py`
   - 修改 import 语句 (line 20)
   - 修改输出目录逻辑 (line 178)

### 新增的文件:
1. `/app/MixDSemi/SynFoCLIP/code/test_pretrain_dataloader.py`
   - 测试所有 4 个数据集的 multi-patch 功能
   - 验证标签映射正确性

## 🎯 关键实现细节

### _normalize_label() 方法实现

#### Prostate (需要反转):
```python
def _normalize_label(self, label_pil):
    label_np = np.array(label_pil)
    normalized = np.zeros_like(label_np)
    normalized[label_np == 0] = 255    # FG: 0 -> 255
    normalized[label_np > 0] = 0       # BG: 255 -> 0
    return Image.fromarray(normalized.astype(np.uint8))
```

#### Fundus (不变):
```python
def _normalize_label(self, label_pil):
    return label_pil  # 0=BG, 128=cup, 255=disc (已正确)
```

#### MNMS (RGB转类别):
```python
def _normalize_label(self, label_pil):
    if label_pil.mode == 'RGB':
        target_np = np.array(label_pil)
        new_target = np.zeros((target_np.shape[0], target_np.shape[1]), dtype=np.uint8)
        for n in range(3):
            new_target[target_np[:, :, n] == 255] = n + 1
        return Image.fromarray(new_target)
    else:
        return label_pil
```

#### BUSI (不变):
```python
def _normalize_label(self, label_pil):
    return label_pil  # 0=BG, 255=tumor (已正确)
```

## 🔍 重要说明

### 1. MNMS "全黑 patch 但 label=1" 问题
这**不是 bug**!原因:
- `patch_sampler` 基于 patch 内实际前景像素比例判断标签
- 如果 patch 内前景比例 > `fg_threshold` (默认 1%),则 label=1
- 即使可视化看起来很黑,但只要有少量前景像素,label 就为 1
- MNMS 的前景值为 1/2/3 (而非 255),在灰度可视化中几乎看不见

### 2. ToTensor 不归一化 mask
```python
# custom_transforms.py, line 756
map = np.array(sample['label']).astype(np.uint8)  # 保持原值
# line 785
map = torch.from_numpy(map).float()  # 转 float 但不 /255
```

这是**有意设计**,因为:
- Image 输入需要归一化到 [-1, 1] (通过 Normalize_tf)
- Mask 需要保持原始类别索引 (0, 1, 2, 3... 或 0, 255)

### 3. 文件组织架构
```
/app/MixDSemi/SynFoCLIP/code/dataloaders/
├── dataloader.py           # 原始单图版本(不应修改)
└── pretrain_dataloader.py  # Multi-patch 版本(所有修改在此)
```

**原则**: 
- `dataloader.py` 保持与 `/app/MixDSemi/SynFoC/code/dataloaders/dataloader.py` 一致
- 所有 multi-patch 功能放在 `pretrain_dataloader.py`

## ✨ 使用示例

```python
from dataloaders.pretrain_dataloader import FundusSegmentation
from dataloaders.custom_transforms import RandomPatchSamplerWithClass
from torchvision import transforms as T

# 创建 patch sampler
patch_sampler = RandomPatchSamplerWithClass(
    num_patches=4,
    num_fg=2,
    min_ratio=0.5,
    fg_threshold=0.01
)

# 创建数据集
dataset = FundusSegmentation(
    base_dir='/app/MixDSemi/data/Fundus',
    phase='train',
    splitid=1,
    domain=[1,2,3,4],
    weak_transform=None,
    strong_tranform=None,
    normal_toTensor=T.Compose([
        Normalize_tf(),
        ToTensor()
    ]),
    patch_sampler=patch_sampler  # 启用 multi-patch 模式
)

# 获取样本
sample = dataset[0]
# 输出格式:
# {
#     'image': Tensor[num_patches, C, H, W],
#     'label': Tensor[num_patches, H, W],
#     'patch_labels': Tensor[num_patches],  # 0/1
#     'img_name': str,
#     'dc': int,
#     'num_patches': int
# }
```

## 📊 测试结果

### 数据集加载统计:
- Prostate: 1510 samples (6 domains)
- Fundus: 789 samples (4 domains)
- MNMS: 3447 samples (4 vendors)
- BUSI: 518 samples (2 classes: benign/malignant)

### 可视化输出:
```
/app/MixDSemi/results/patch_batch_viz/
├── prostate/
│   └── batch_sample_0.png
├── fundus/
│   └── batch_sample_0.png
├── MNMS/
│   └── batch_sample_0.png
└── BUSI/
    └── batch_sample_0.png
```

## ✅ 验证清单

- [x] 所有 4 个数据集类添加到 pretrain_dataloader.py
- [x] _normalize_label() 方法正确实现
- [x] patch_sampler 兼容性验证
- [x] 标签映射关系验证 (BG=0, FG>0)
- [x] 可视化脚本更新 (按数据集分目录)
- [x] 测试脚本通过
- [x] 原始 dataloader.py 保持不变

## 🎉 总结

所有功能已完成并验证:
1. ✅ 三个数据集 (Fundus, MNMS, BUSI) 成功添加 multi-patch 支持
2. ✅ 标签映射关系正确 (所有数据集满足 BG=0, FG>0)
3. ✅ 可视化按数据集分目录,不会相互覆盖
4. ✅ 所有测试通过

**架构原则**: pretrain_dataloader.py 用于 multi-patch 版本,dataloader.py 保持原样。
