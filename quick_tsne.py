#!/usr/bin/env python3

"""

专门针对GPT5-80描述的域感知能力快速分析脚本

"""



import os

import sys

import torch

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings('ignore')



# 设置matplotlib后端和中文字体

import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端

plt.rcParams['font.size'] = 10

plt.rcParams['figure.dpi'] = 100



def quick_domain_analysis(dataset_name='ProstateSlice',

                         llm='DeepSeek',

                         describe_nums=80,

                         preprocess_root='/app/MixDSemi/SynFoCLIP/preprocess'):

    """

    快速分析指定数据集的域感知能力

    """

    print(f"\n🔍 Analyzing {dataset_name} with {llm}-{describe_nums}")

    print("-" * 50)

   

    # 检查数据路径

    dataset_path = os.path.join(preprocess_root, dataset_name)

    if not os.path.exists(dataset_path):

        print(f"❌ Dataset path not found: {dataset_path}")

        return None

   

    # 加载所有域的数据

    domain_data = {}

    domains = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

   

    print(f"Found domains: {domains}")

   

    for domain in domains:

        score_file = os.path.join(dataset_path, domain, f"{llm}_{describe_nums}.pt")

        if os.path.exists(score_file):

            print(f"📂 Loading {domain}...")

            try:

                image_text_match = torch.load(score_file, map_location='cpu')

                images = list(image_text_match.keys())

                scores = torch.stack(list(image_text_match.values())).numpy()

               

                domain_data[domain] = {

                    'images': images,

                    'scores': scores

                }

                print(f"   ✅ {len(images)} images, {scores.shape[1]} text features")

            except Exception as e:

                print(f"   ❌ Error loading {domain}: {e}")

        else:

            print(f"   ⚠️ Score file not found for {domain}: {score_file}")

   

    if not domain_data:

        print("❌ No data loaded!")

        return None

   

    # 准备数据进行t-SNE

    print("\n🧮 Computing t-SNE...")

    all_scores = []

    all_labels = []

    all_domains = []

   

    for domain, data in domain_data.items():

        all_scores.append(data['scores'])

        all_labels.extend([domain] * len(data['images']))

        all_domains.append(domain)

   

    all_scores = np.vstack(all_scores)

    print(f"Total samples: {all_scores.shape[0]}, Features: {all_scores.shape[1]}")

   

    # 如果特征太多，先PCA降维

    if all_scores.shape[1] > 50:

        print("   Applying PCA preprocessing...")

        pca = PCA(n_components=50, random_state=42)

        all_scores = pca.fit_transform(all_scores)

   

    # t-SNE降维

    print("   Running t-SNE...")

    perplexity = min(30, max(5, len(all_scores)//4))

    try:

        tsne = TSNE(n_components=2, perplexity=perplexity,

                   random_state=42, max_iter=1000)

    except TypeError:

        # 兼容老版本sklearn

        tsne = TSNE(n_components=2, perplexity=perplexity,

                   random_state=42)

    tsne_coords = tsne.fit_transform(all_scores)

   

    # 绘制结果

    print("\n📊 Creating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

   

    # 为每个域分配颜色

    colors = plt.cm.Set1(np.linspace(0, 1, len(all_domains)))

    domain_colors = dict(zip(all_domains, colors))

   

    # 左图: t-SNE散点图

    ax1 = axes[0]

    start_idx = 0

    for domain, data in domain_data.items():

        end_idx = start_idx + len(data['images'])

        coords = tsne_coords[start_idx:end_idx]

       

        ax1.scatter(coords[:, 0], coords[:, 1],

                   c=[domain_colors[domain]],

                   label=f'{domain} (n={len(coords)})',

                   alpha=0.7, s=30)

        start_idx = end_idx

   

    ax1.set_title(f'{dataset_name} Domain Distribution\\n{llm} with {describe_nums} descriptions')

    ax1.set_xlabel('t-SNE Component 1')

    ax1.set_ylabel('t-SNE Component 2')

    ax1.legend()

    ax1.grid(True, alpha=0.3)

   

    # 右图: 域间距离分析

    ax2 = axes[1]

   

    # 计算域中心和距离矩阵

    domain_centers = {}

    start_idx = 0

    for domain, data in domain_data.items():

        end_idx = start_idx + len(data['images'])

        coords = tsne_coords[start_idx:end_idx]

        domain_centers[domain] = np.mean(coords, axis=0)

        start_idx = end_idx

   

    # 距离矩阵

    n_domains = len(all_domains)

    distance_matrix = np.zeros((n_domains, n_domains))

   

    for i, domain1 in enumerate(all_domains):

        for j, domain2 in enumerate(all_domains):

            if i != j:

                dist = np.linalg.norm(domain_centers[domain1] - domain_centers[domain2])

                distance_matrix[i, j] = dist

   

    sns.heatmap(distance_matrix, annot=True, fmt='.2f',

               xticklabels=all_domains, yticklabels=all_domains,

               cmap='YlOrRd', ax=ax2)

    ax2.set_title('Inter-Domain Distance Matrix')

   

    plt.tight_layout()

   

    # 保存图片

    output_file = f"{dataset_name}_{llm}_{describe_nums}_domain_analysis.png"

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.close()

   

    print(f"💾 Saved plot: {output_file}")

   

    # 计算和打印分离度指标

    print("\n📈 Domain Separation Analysis:")

    print("-" * 30)

   

    # 域内紧密度

    intra_spreads = []

    start_idx = 0

    for domain, data in domain_data.items():

        end_idx = start_idx + len(data['images'])

        coords = tsne_coords[start_idx:end_idx]

        center = domain_centers[domain]

        spread = np.mean(np.linalg.norm(coords - center, axis=1))

        intra_spreads.append(spread)

        print(f"{domain:12}: {len(coords):3d} samples, spread: {spread:.3f}")

        start_idx = end_idx

   

    # 域间距离

    inter_distances = []

    for i, domain1 in enumerate(all_domains):

        for j, domain2 in enumerate(all_domains):

            if i < j:

                dist = np.linalg.norm(domain_centers[domain1] - domain_centers[domain2])

                inter_distances.append(dist)

   

    avg_intra_spread = np.mean(intra_spreads)

    avg_inter_distance = np.mean(inter_distances)

    separation_ratio = avg_inter_distance / avg_intra_spread if avg_intra_spread > 0 else 0

   

    print(f"\nSummary:")

    print(f"Avg intra-domain spread: {avg_intra_spread:.3f}")

    print(f"Avg inter-domain distance: {avg_inter_distance:.3f}")

    print(f"Separation ratio: {separation_ratio:.3f}")

   

    if separation_ratio > 2.0:

        print("✅ Excellent domain separation!")

    elif separation_ratio > 1.5:

        print("🟡 Good domain separation")

    elif separation_ratio > 1.0:

        print("🟠 Moderate domain separation")

    else:

        print("❌ Poor domain separation")

   

    return {

        'dataset': dataset_name,

        'separation_ratio': separation_ratio,

        'avg_intra_spread': avg_intra_spread,

        'avg_inter_distance': avg_inter_distance,

        'n_domains': len(all_domains),

        'total_samples': all_scores.shape[0],

        'plot_file': output_file

    }



def analyze_all_available_datasets():

    """分析所有可用的数据集"""

    preprocess_root = '/app/MixDSemi/SynFoCLIP/preprocess'

    available_datasets = [d for d in os.listdir(preprocess_root)

                         if os.path.isdir(os.path.join(preprocess_root, d))]

   

    print(f"🔍 Found datasets: {available_datasets}")

   

    results = []

    for dataset in available_datasets:

        try:

            result = quick_domain_analysis(dataset)

            if result:

                results.append(result)

        except Exception as e:

            print(f"❌ Error analyzing {dataset}: {e}")

   

    # 总结所有结果

    if results:

        print(f"\n{'='*60}")

        print("🏆 OVERALL DOMAIN AWARENESS SUMMARY")

        print('='*60)

       

        # 按分离度排序

        results.sort(key=lambda x: x['separation_ratio'], reverse=True)

       

        for i, result in enumerate(results, 1):

            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"

            print(f"{status} {result['dataset']:15} | Ratio: {result['separation_ratio']:5.2f} | "

                  f"{result['n_domains']} domains | {result['total_samples']} samples")

   

    return results



if __name__ == "__main__":

    if len(sys.argv) > 1:

        dataset_name = sys.argv[1]

        print(f"Analyzing specific dataset: {dataset_name}")

        quick_domain_analysis(dataset_name)

    else:

        print("Analyzing all available datasets...")

        analyze_all_available_datasets()

