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

        # 将 'style' 与其余目标文本分离。style 不参与后续的分组/划分采样。
        llm_texts = text_descriptions[llm]
        style_texts = llm_texts.get('style', [])

        # targets_texts 包含除了 'style' 之外的所有目标类描述
        targets_texts = {}
        for type_key, texts in llm_texts.items():
            if type_key == 'style':
                continue
            targets_texts[type_key] = texts[:describe_nums]  # 选取前 describe_nums 个文本

        # 扁平化仅对目标文本（不包含 style）进行拼接
        flat_list = []
        for type_key in targets_texts.keys():
            flat_list.extend(targets_texts[type_key])

        # 返回 (targets_dict, style_list, flat_targets_list)
        return targets_texts, style_texts, flat_list

    def sample_subsets(self, all_texts, num_samples):
        """
        为每个类别生成具有约 50% 重叠的文本子集。

        Args:
            all_texts (dict): {type_key: [text1, text2, ...]}。
            num_samples (int): 采样份数 X。

        Returns:
            tuple: (per_type_subsets, flat_subsets)
                   - per_type_subsets: list of dicts (按类别组织的子集列表)
                   - flat_subsets: list of flat lists (每个子集为一维文本列表)
        """
        # 构造 X 个子集的容器（按类别组织的字典），后面可选择打平返回
        per_type_subsets = [dict() for _ in range(num_samples)]

        # 按类别处理，确保每个子集对每个类别的数量完全相同（允许在必要时重复抽样）
        # 如果传入的 all_texts 中包含 'style' 键，则忽略它（style 不参与分组采样）。
        import math
        for type_key, texts in [(k, v) for k, v in all_texts.items() if k != 'style']:
            N = len(texts)
            if N == 0:
                # 每个子集该类别为空列表
                for s in range(num_samples):
                    per_type_subsets[s][type_key] = []
                continue

            # 公共池大小 O，取 N // (num_samples + 1)
            overlap_size = N // (num_samples + 1)

            all_indices = list(range(N))
            common_indices = random.sample(all_indices, overlap_size) if overlap_size > 0 else []
            common_set = set(common_indices)
            common_texts = [texts[i] for i in common_indices]

            # 剩余可分配的唯一索引
            remaining_indices = [i for i in all_indices if i not in common_set]
            random.shuffle(remaining_indices)

            # 要保证每个子集对该类别的数量相同，我们设定每个子集的 block_target = ceil(len(remaining)/num_samples)
            rem_len = len(remaining_indices)
            if rem_len == 0:
                for s in range(num_samples):
                    per_type_subsets[s][type_key] = list(common_texts)
                continue

            block_target = math.ceil(rem_len / num_samples)

            # 逐个子集分配 block_target 个索引；优先使用 remaining_indices（不重复），不足时允许重复采样以满足数量
            pos = 0
            for s in range(num_samples):
                take = []
                # 取不重复的部分
                if pos < rem_len:
                    take_count = min(block_target, rem_len - pos)
                    take_indices = remaining_indices[pos:pos + take_count]
                    pos += take_count
                    take.extend(take_indices)

                # 若仍不足，从 remaining_indices 中随机抽样（允许重复）补齐
                if len(take) < block_target:
                    need = block_target - len(take)
                    if rem_len > 0:
                        extra_indices = random.choices(remaining_indices, k=need)
                    else:
                        # 没有剩余索引，只能从所有 indices 中重复抽样
                        extra_indices = random.choices(all_indices, k=need)
                    take.extend(extra_indices)

                unique_texts = [texts[i] for i in take]
                per_type_subsets[s][type_key] = list(common_texts + unique_texts)

        # 检查每个子集的总长度是否相等；若不等则抛错（用户要求）
        totals = [sum(len(v) for v in per_type_subsets[s].values()) for s in range(num_samples)]
        if len(set(totals)) != 1:
                import warnings
                target = max(totals)
                adjusted = False

                # 预先构建每个子集的扁平化带类别信息的列表 -> list of (text, type_key)
                flat_with_type = []
                for s in range(num_samples):
                    items = []
                    for tkey, lst in per_type_subsets[s].items():
                        items.extend([(txt, tkey) for txt in lst])
                    flat_with_type.append(items)

                for s in range(num_samples):
                    cur_total = totals[s]
                    if cur_total >= target:
                        continue
                    need = target - cur_total

                    # 构建候选池：优先使用前面的子集（index < s）的所有条目
                    donor_pool = []
                    for prev in range(0, s):
                        donor_pool.extend(flat_with_type[prev])

                    # 如果前面的子集没有提供候选，则使用其他子集（除了当前）作为后备
                    if len(donor_pool) == 0:
                        for other in range(num_samples):
                            if other == s:
                                continue
                            donor_pool.extend(flat_with_type[other])

                    # 去掉已经存在于当前子集的文本，避免重复
                    existing_texts = set(txt for txt, _ in flat_with_type[s])
                    donor_pool_unique = [(txt, tk) for (txt, tk) in donor_pool if txt not in existing_texts]

                    sampled = []
                    if len(donor_pool_unique) >= need:
                        sampled = random.sample(donor_pool_unique, need)
                    else:
                        # 如果候选不足，先取全部无重复的候选，再从所有其它文本中补齐（允许在极端情况下重复）
                        sampled = list(donor_pool_unique)
                        still_need = need - len(sampled)
                        # 后备池：所有原始文本（按类别）除去已有的
                        all_original = []
                        for tkey, texts in all_texts.items():
                            all_original.extend([(txt, tkey) for txt in texts if txt not in existing_texts and (txt, tkey) not in sampled])

                        if len(all_original) >= still_need:
                            sampled.extend(random.sample(all_original, still_need))
                        else:
                            # 最后退化：允许重复采样自 donor_pool（有可能产生重复文本），但保留类型信息
                            if len(donor_pool) > 0:
                                for _ in range(still_need):
                                    sampled.append(random.choice(donor_pool))
                            else:
                                # 不太可能发生：没有任何可供采样的候选
                                raise RuntimeError("Unable to auto-balance subsets: no donor items available")

                    # 将 sampled 项按所属类别放回 per_type_subsets[s]
                    for txt, tkey in sampled:
                        if tkey not in per_type_subsets[s]:
                            per_type_subsets[s][tkey] = []
                        per_type_subsets[s][tkey].append(txt)
                        flat_with_type[s].append((txt, tkey))

                    totals[s] = sum(len(v) for v in per_type_subsets[s].values())
                    adjusted = True

                if adjusted:
                    warnings.warn(f"Subsets had unequal sizes {totals} initially; auto-balanced to target size {target} by sampling from earlier subsets.")

        # 同时返回按类别组织的子集和扁平化的子集列表，便于调用处选择使用
        flat_subsets = []
        for s in range(num_samples):
            flat = []
            # 保持类别顺序与 all_texts keys 一致，但排除 'style'
            for type_key in [k for k in all_texts.keys() if k != 'style']:
                flat.extend(per_type_subsets[s].get(type_key, []))
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
    all_texts, style_texts, flat_texts = sampler.load_texts(dataset, llm, describe_nums)
    # print(all_texts)
    print(f"Loaded texts: {len(all_texts)} types, {len(flat_texts)} total flat items")
    print(f"style texts: {len(style_texts)} items")
    print(type(all_texts), type(flat_texts))
    # 打印第一个类别的信息（如果存在）
    if len(all_texts) > 0:
        first_key = next(iter(all_texts.keys()))
        print(f"First type '{first_key}' has {len(all_texts[first_key])} texts")

    per_type_subsets, flat_subsets = sampler.sample_subsets(all_texts, num_samples)
    print(f"\nGenerated {len(flat_subsets)} subsets:")
    for i, subset in enumerate(flat_subsets):
        print(f"Subset {i+1}: {len(subset)} texts")

    # 检查每个子集的类别数量（按原始类别统计），使用扁平子集计算
    print('\nPer-type counts in each subset:')
    for i, subset in enumerate(flat_subsets):
        print(f" Subset {i+1}:")
        for type_key, texts in all_texts.items():
            # 统计当前子集中属于该类别的文本数量
            texts_set = set(texts)
            count = sum(1 for t in subset if t in texts_set)
            print(f"  - {type_key}: {count} texts")

    # 检查覆盖
    all_original_texts = set()
    for texts in all_texts.values():
        all_original_texts.update(texts)

    all_subset_texts = set()
    for subset in flat_subsets:
        all_subset_texts.update(subset)

    coverage = all_subset_texts.issuperset(all_original_texts)
    print(f"\nCoverage check: All original texts covered? {coverage}")
    if not coverage:
        missing = all_original_texts - all_subset_texts
        print(f"Missing texts: {len(missing)}")
