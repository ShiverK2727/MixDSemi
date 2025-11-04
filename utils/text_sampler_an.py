import json
import os
import random

class TextSampler:
    """
    文本采样器，按照 preprocess.py 的逻辑加载文本，并生成具有重叠的子集。
    """
    def __init__(self, text_root):
        """
        初始化文本采样器。

        Args:
            text_root (str): 文本 JSON 文件的根目录路径。
        """
        self.text_root = text_root

    def load_texts(self, dataset, llm, describe_nums):
        """
        加载指定数据集、LLM 和描述数量的文本。

        设计更改：该函数现在总是返回一个二元组
        (all_texts_dict, flat_list)，其中 all_texts_dict 为按类别组织的字典，
        flat_list 为把所有类别按 key 顺序拼接后的扁平文本列表。

        Args:
            dataset (str): 数据集名称（如 'ProstateSlice'）。
            llm (str): LLM 名称（如 'gemini'）。
            describe_nums (int): 每个类别的描述数量。

        Returns:
            tuple: (dict, list) -> (all_texts_dict, flat_list)
        """
        text_file = os.path.join(self.text_root, f"{dataset}.json")
        with open(text_file, 'r') as f:
            text_descriptions = json.load(f)

        # 读取指定 LLM 下的所有类别文本，按原始顺序构造一个 list-of-lists
        llm_texts = text_descriptions[llm]

        per_class_lists = []
        # 保持文件中类别的原始顺序（JSON 加载会保留顺序）
        for type_key, texts in llm_texts.items():
            # 不再单独处理 'style'，直接把每个类别（可能为 style）的文本作为一项
            per_class_lists.append(texts[:describe_nums])

        # 构造按类顺序打平的列表：先是第一个类的所有文本，接着第二个类的所有文本，依此类推
        flat_list = []
        for lst in per_class_lists:
            flat_list.extend(lst)

        # 返回 (per_class_lists, flat_list)
        return per_class_lists, flat_list

    def sample_subsets(self, all_texts, num_samples):
        """
        为每个类别生成具有约 50% 重叠的文本子集（兼容 list-of-lists 输入）。

        Args:
            all_texts (list of lists): [[class0_texts], [class1_texts], ...]
            num_samples (int): 采样份数 X。

        Returns:
            tuple: (per_type_subsets, flat_subsets)
                   - per_type_subsets: list (length=num_samples) 每项为 list-of-lists (每个类别的文本列表)
                   - flat_subsets: list (length=num_samples) 每项为一个扁平化的文本列表（类别顺序对应输入顺序）
        """
        import math

        num_classes = len(all_texts)
        # per_type_subsets[s] 是当前样本 s 的类别列表（每个类别是一个 list）
        per_type_subsets = [[[] for _ in range(num_classes)] for _ in range(num_samples)]

        # 对每个类别按原逻辑分配
        for cls_idx, texts in enumerate(all_texts):
            N = len(texts)
            if N == 0:
                # 每个子集该类别为空列表
                for s in range(num_samples):
                    per_type_subsets[s][cls_idx] = []
                continue

            overlap_size = N // (num_samples + 1)
            all_indices = list(range(N))
            common_indices = random.sample(all_indices, overlap_size) if overlap_size > 0 else []
            common_set = set(common_indices)
            common_texts = [texts[i] for i in common_indices]

            remaining_indices = [i for i in all_indices if i not in common_set]
            random.shuffle(remaining_indices)

            rem_len = len(remaining_indices)
            if rem_len == 0:
                for s in range(num_samples):
                    per_type_subsets[s][cls_idx] = list(common_texts)
                continue

            block_target = math.ceil(rem_len / num_samples)

            pos = 0
            for s in range(num_samples):
                take = []
                if pos < rem_len:
                    take_count = min(block_target, rem_len - pos)
                    take_indices = remaining_indices[pos:pos + take_count]
                    pos += take_count
                    take.extend(take_indices)

                if len(take) < block_target:
                    need = block_target - len(take)
                    if rem_len > 0:
                        extra_indices = random.choices(remaining_indices, k=need)
                    else:
                        extra_indices = random.choices(all_indices, k=need)
                    take.extend(extra_indices)

                unique_texts = [texts[i] for i in take]
                per_type_subsets[s][cls_idx] = list(common_texts + unique_texts)

        # 检查每个子集的总长度是否相等；若不等则尝试自动平衡（保留原逻辑）
        totals = [sum(len(per_type_subsets[s][c]) for c in range(num_classes)) for s in range(num_samples)]
        if len(set(totals)) != 1:
            import warnings
            target = max(totals)
            adjusted = False

            # 预先构建每个子集的扁平化带类别信息的列表 -> list of (text, cls_idx)
            flat_with_cls = []
            for s in range(num_samples):
                items = []
                for cls_idx in range(num_classes):
                    items.extend([(txt, cls_idx) for txt in per_type_subsets[s][cls_idx]])
                flat_with_cls.append(items)

            for s in range(num_samples):
                cur_total = totals[s]
                if cur_total >= target:
                    continue
                need = target - cur_total

                donor_pool = []
                for prev in range(0, s):
                    donor_pool.extend(flat_with_cls[prev])

                if len(donor_pool) == 0:
                    for other in range(num_samples):
                        if other == s:
                            continue
                        donor_pool.extend(flat_with_cls[other])

                existing_texts = set(txt for txt, _ in flat_with_cls[s])
                donor_pool_unique = [(txt, ci) for (txt, ci) in donor_pool if txt not in existing_texts]

                sampled = []
                if len(donor_pool_unique) >= need:
                    sampled = random.sample(donor_pool_unique, need)
                else:
                    sampled = list(donor_pool_unique)
                    still_need = need - len(sampled)
                    all_original = []
                    for ci, texts in enumerate(all_texts):
                        all_original.extend([(txt, ci) for txt in texts if txt not in existing_texts and (txt, ci) not in sampled])

                    if len(all_original) >= still_need:
                        sampled.extend(random.sample(all_original, still_need))
                    else:
                        if len(donor_pool) > 0:
                            for _ in range(still_need):
                                sampled.append(random.choice(donor_pool))
                        else:
                            raise RuntimeError("Unable to auto-balance subsets: no donor items available")

                for txt, ci in sampled:
                    per_type_subsets[s][ci].append(txt)
                    flat_with_cls[s].append((txt, ci))

                totals[s] = sum(len(per_type_subsets[s][c]) for c in range(num_classes))
                adjusted = True

            if adjusted:
                warnings.warn(f"Subsets had unequal sizes {totals} initially; auto-balanced to target size {target} by sampling from earlier subsets.")

        # 生成扁平化子集（类别顺序与输入 all_texts 保持一致）
        flat_subsets = []
        for s in range(num_samples):
            flat = []
            for cls_idx in range(num_classes):
                flat.extend(per_type_subsets[s][cls_idx])
            flat_subsets.append(flat)

        return per_type_subsets, flat_subsets

# 示例用法
if __name__ == "__main__":
    # 示例参数
    text_root = "/app/MixDSemi/SynFoCLIP/code/text"
    dataset = "ProstateSlice"
    llm = "gemini"
    describe_nums = 80
    num_samples = 2

    sampler = TextSampler(text_root)
    per_class_lists, flat_texts = sampler.load_texts(dataset, llm, describe_nums)
    # print(per_class_lists)
    print(f"Loaded texts: {len(per_class_lists)} types, {len(flat_texts)} total flat items")
    print(type(per_class_lists), type(flat_texts))
    # 打印第一个类别的信息（如果存在）
    if len(per_class_lists) > 0:
        print(f"First type (index 0) has {len(per_class_lists[0])} texts")

    per_type_subsets, flat_subsets = sampler.sample_subsets(per_class_lists, num_samples)
    print(f"\nGenerated {len(flat_subsets)} subsets:")
    for i, subset in enumerate(flat_subsets):
        print(f"Subset {i+1}: {len(subset)} texts")

    # 检查每个子集的类别数量（按原始类别统计），使用扁平子集计算
    print('\nPer-type counts in each subset:')
    for i, subset in enumerate(flat_subsets):
        print(f" Subset {i+1}:")
        for type_idx, texts in enumerate(per_class_lists):
            # 统计当前子集中属于该类别的文本数量
            texts_set = set(texts)
            count = sum(1 for t in subset if t in texts_set)
            print(f"  - class_{type_idx}: {count} texts")

    # 检查覆盖
    all_original_texts = set()
    for texts in per_class_lists:
        all_original_texts.update(texts)

    all_subset_texts = set()
    for subset in flat_subsets:
        all_subset_texts.update(subset)

    coverage = all_subset_texts.issuperset(all_original_texts)
    print(f"\nCoverage check: All original texts covered? {coverage}")
    if not coverage:
        missing = all_original_texts - all_subset_texts
        print(f"Missing texts: {len(missing)}")
