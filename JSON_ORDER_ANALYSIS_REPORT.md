"""
JSON键值对顺序和Style索引分析报告
=====================================

## 问题回答

**问题**: `for texts in text_descriptions[llm].values()`，这个在典型JSON的排序是什么？你确定现在的过程里style还是在最后一位吗，详细检查整个过程的索引变化

## 分析结果

### 1. JSON文件结构分析

所有4个数据集的JSON文件都采用统一的结构：
- **ProstateSlice**: ['prostate', 'style']  (2个类型)
- **Fundus**: ['cup', 'disc', 'style']      (3个类型)
- **MNMS**: ['left ventricle (LV)', 'left ventricle myocardium (MYO)', 'right ventricle (RV)', 'style']  (4个类型)
- **BUSI**: ['breast tumor', 'style']        (2个类型)

**关键发现**: 在所有数据集中，'style'键都**严格位于最后一位**。

### 2. Python字典顺序保证

- Python 3.7+保证字典保持插入顺序
- `text_descriptions[llm].values()`的遍历顺序与JSON文件中的键顺序**完全一致**
- 多次遍历结果一致，无随机性

### 3. 文本处理过程索引映射

```python
# preprocess.py中的处理逻辑
total_types = len(text_descriptions[llm])  # 获取类型数量
selected_texts = []

for texts in text_descriptions[llm].values():  # 按JSON顺序遍历
    selected_texts.extend(texts[:describe_nums])

# 索引映射关系 (以BUSI为例)
# 索引 0: 'breast tumor' texts -> reshaped_logits[0, :]
# 索引 1: 'style' texts        -> reshaped_logits[1, :]  (最后一个)
```

### 4. 2D张量格式中的Style分离

```python
# 张量形状: [total_types, describe_nums]
reshaped_logits = image_logits.reshape(total_types, describe_nums)

# Style总是在最后一行
style_scores = reshaped_logits[-1, :]      # shape: [describe_nums]
content_scores = reshaped_logits[:-1, :]   # shape: [total_types-1, describe_nums]
```

### 5. 各数据集的具体映射

#### BUSI (total_types=2)
- `reshaped_logits[0, :] -> 'breast tumor'` (content)
- `reshaped_logits[1, :] -> 'style'`        (style)
- 分离: `style = [-1, :]`, `content = [:-1, :]`

#### Fundus (total_types=3)
- `reshaped_logits[0, :] -> 'cup'`         (content)
- `reshaped_logits[1, :] -> 'disc'`        (content)  
- `reshaped_logits[2, :] -> 'style'`       (style)
- 分离: `style = [-1, :]`, `content = [:-1, :]`

#### MNMS (total_types=4)
- `reshaped_logits[0, :] -> 'left ventricle (LV)'`        (content)
- `reshaped_logits[1, :] -> 'left ventricle myocardium (MYO)'` (content)
- `reshaped_logits[2, :] -> 'right ventricle (RV)'`       (content)
- `reshaped_logits[3, :] -> 'style'`                      (style)
- 分离: `style = [-1, :]`, `content = [:-1, :]`

## 结论

**YES - Style确实在最后一位！**

1. **JSON结构正确**: 所有数据集的'style'键都在最后位置
2. **Python顺序保证**: `text_descriptions[llm].values()`按JSON顺序遍历
3. **索引映射准确**: Style类型始终映射到张量的最后一行 `[-1, :]`
4. **分离逻辑正确**: 使用`[:-1, :]`和`[-1, :]`可以正确分离content和style

### 推荐的Style分离代码

```python
# 假设按顺序total_types中最后一个types是图像风格提示
def separate_style_content(reshaped_logits):
    \"\"\"
    输入: reshaped_logits shape=[total_types, describe_nums]
    输出: content_scores, style_scores
    \"\"\"
    # Style分数 (最后一个type)
    style_scores = reshaped_logits[-1, :]  # shape: [describe_nums]
    
    # Content分数 (除了最后一个type的所有types)  
    content_scores = reshaped_logits[:-1, :]  # shape: [total_types-1, describe_nums]
    
    return content_scores, style_scores
```

**答案**: 使用`[:-1]`获取content，使用`[-1, :]`获取style是**完全正确**的。
"""