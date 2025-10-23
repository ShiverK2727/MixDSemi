#!/usr/bin/env python3
"""
支持新二维格式的t-SNE域感知分析
支持分离内容类型和风格提示分析
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

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

def analyze_2d_format_domain_awareness(dataset_name='ProstateSlice', 
                                     llm='GPT5', 
                                     describe_nums=80,
                                     preprocess_root='/app/MixDSemi/SynFoCLIP/preprocess',
                                     separate_style=True,
                                     style_at_end=True):
    """
    分析新二维格式的域感知能力，支持风格提示分离
    
    Args:
        separate_style: 是否分离风格提示进行单独分析
        style_at_end: 风格提示是否在最后一个type位置
    """
    print(f"\n🔍 Analyzing {dataset_name} with {llm}-{describe_nums} (2D Format)")
    print("-" * 60)
    
    # 检查数据路径
    dataset_path = os.path.join(preprocess_root, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return None
    
    # 加载所有域的数据
    domain_data = {}
    domains = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Found domains: {domains}")
    
    data_format = None  # 检测数据格式
    
    for domain in domains:
        score_file = os.path.join(dataset_path, domain, f"{llm}_{describe_nums}.pt")
        if os.path.exists(score_file):
            print(f"📂 Loading {domain}...")
            try:
                image_text_match = torch.load(score_file, map_location='cpu')
                images = list(image_text_match.keys())
                
                # 检测数据格式
                sample_data = list(image_text_match.values())[0]
                if len(sample_data.shape) == 1:
                    print(f"   📊 Detected 1D format: {sample_data.shape}")
                    data_format = '1D'
                    scores = torch.stack(list(image_text_match.values())).numpy()
                elif len(sample_data.shape) == 2:
                    print(f"   📊 Detected 2D format: {sample_data.shape}")
                    data_format = '2D'
                    total_types, desc_nums = sample_data.shape
                    print(f"   📋 Types: {total_types}, Descriptions per type: {desc_nums}")
                    
                    # 收集2D数据
                    scores_2d = torch.stack(list(image_text_match.values())).numpy()  # [n_images, total_types, desc_nums]
                    
                    if separate_style and style_at_end:
                        # 分离风格提示和内容类型
                        content_scores = scores_2d[:, :-1, :]  # [n_images, total_types-1, desc_nums]
                        style_scores = scores_2d[:, -1, :]     # [n_images, desc_nums]
                        
                        print(f"   🎨 Style scores shape: {style_scores.shape}")
                        print(f"   📝 Content scores shape: {content_scores.shape}")
                        
                        # 重塑内容类型分数为分析格式
                        content_flat = content_scores.reshape(len(images), -1)  # [n_images, (total_types-1)*desc_nums]
                        scores = content_flat  # 只分析内容类型
                        
                        # 保存风格分数用于后续分析
                        domain_data[domain] = {
                            'images': images,
                            'content_scores': scores,
                            'style_scores': style_scores,
                            'original_2d': scores_2d
                        }
                    else:
                        # 不分离，展平整个2D数据
                        scores = scores_2d.reshape(len(images), -1)  # [n_images, total_types*desc_nums]
                        domain_data[domain] = {
                            'images': images,
                            'scores': scores,
                            'original_2d': scores_2d
                        }
                else:
                    print(f"   ❌ Unsupported data format: {sample_data.shape}")
                    continue
                
                if not separate_style:
                    domain_data[domain] = {
                        'images': images,
                        'scores': scores
                    }
                
                print(f"   ✅ {len(images)} images loaded successfully")
                
            except Exception as e:
                print(f"   ❌ Error loading {domain}: {e}")
        else:
            print(f"   ⚠️ Score file not found for {domain}: {score_file}")
    
    if not domain_data:
        print("❌ No data loaded!")
        return None
    
    # 选择用于t-SNE的数据
    analysis_data = {}
    if separate_style and 'content_scores' in list(domain_data.values())[0]:
        print(f"\n🎯 Analyzing CONTENT types only (excluding style)")
        for domain, data in domain_data.items():
            analysis_data[domain] = {
                'images': data['images'],
                'scores': data['content_scores']
            }
    else:
        print(f"\n🎯 Analyzing ALL types together")
        analysis_data = domain_data
    
    # 执行t-SNE分析
    tsne_results = compute_tsne_analysis(analysis_data)
    if not tsne_results:
        return None
    
    # 创建可视化
    fig = create_enhanced_visualization(tsne_results, dataset_name, llm, describe_nums, separate_style)
    
    # 保存结果
    suffix = "_content_only" if separate_style else "_all_types"
    output_file = f"{dataset_name}_{llm}_{describe_nums}_2D{suffix}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"💾 Saved analysis: {output_file}")
    
    # 如果有风格数据，额外分析风格分布
    if separate_style and 'style_scores' in list(domain_data.values())[0]:
        analyze_style_distribution(domain_data, dataset_name, llm, describe_nums)
    
    return tsne_results

def compute_tsne_analysis(analysis_data):
    """计算t-SNE分析"""
    print(f"\n🧮 Computing t-SNE...")
    
    # 准备数据
    all_scores = []
    all_labels = []
    all_domains = list(analysis_data.keys())
    
    for domain, data in analysis_data.items():
        all_scores.append(data['scores'])
        all_labels.extend([domain] * len(data['images']))
    
    all_scores = np.vstack(all_scores)
    print(f"Total samples: {all_scores.shape[0]}, Features: {all_scores.shape[1]}")
    
    # PCA预处理
    if all_scores.shape[1] > 50:
        print("   Applying PCA preprocessing...")
        pca = PCA(n_components=50, random_state=42)
        all_scores = pca.fit_transform(all_scores)
        print(f"   Reduced to {all_scores.shape[1]} dimensions")
    
    # t-SNE降维
    print("   Running t-SNE...")
    perplexity = min(30, max(5, len(all_scores)//4))
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, 
                   random_state=42, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    
    tsne_coords = tsne.fit_transform(all_scores)
    
    # 重组结果
    results = {}
    start_idx = 0
    for domain, data in analysis_data.items():
        end_idx = start_idx + len(data['images'])
        results[domain] = {
            'images': data['images'],
            'tsne_coords': tsne_coords[start_idx:end_idx]
        }
        start_idx = end_idx
    
    return results

def create_enhanced_visualization(tsne_results, dataset_name, llm, describe_nums, separate_style):
    """创建增强的可视化"""
    print(f"\n📊 Creating enhanced visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name} - Domain Analysis (2D Format)\\n{llm} with {describe_nums} descriptions' + 
                (f' - Content Only' if separate_style else ' - All Types'), fontsize=14)
    
    # 为每个域分配颜色
    domains = list(tsne_results.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(domains)))
    domain_colors = dict(zip(domains, colors))
    
    # 1. t-SNE散点图
    ax1 = axes[0, 0]
    for domain, data in tsne_results.items():
        coords = data['tsne_coords']
        ax1.scatter(coords[:, 0], coords[:, 1], 
                   c=[domain_colors[domain]], 
                   label=f'{domain} (n={len(coords)})',
                   alpha=0.7, s=30)
    
    ax1.set_title('t-SNE Domain Distribution')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 域间距离矩阵
    ax2 = axes[0, 1]
    domain_centers = {}
    for domain, data in tsne_results.items():
        coords = data['tsne_coords']
        domain_centers[domain] = np.mean(coords, axis=0)
    
    n_domains = len(domains)
    distance_matrix = np.zeros((n_domains, n_domains))
    
    for i, domain1 in enumerate(domains):
        for j, domain2 in enumerate(domains):
            if i != j:
                dist = np.linalg.norm(domain_centers[domain1] - domain_centers[domain2])
                distance_matrix[i, j] = dist
    
    sns.heatmap(distance_matrix, annot=True, fmt='.2f',
               xticklabels=domains, yticklabels=domains,
               cmap='YlOrRd', ax=ax2)
    ax2.set_title('Inter-Domain Distance Matrix')
    
    # 3. 域内紧密度分析
    ax3 = axes[1, 0]
    spreads = []
    domain_names = []
    
    for domain, data in tsne_results.items():
        coords = data['tsne_coords']
        center = domain_centers[domain]
        spread = np.mean(np.linalg.norm(coords - center, axis=1))
        spreads.append(spread)
        domain_names.append(domain)
    
    bars = ax3.bar(range(len(spreads)), spreads, color=[domain_colors[d] for d in domain_names])
    ax3.set_xticks(range(len(domain_names)))
    ax3.set_xticklabels(domain_names, rotation=45)
    ax3.set_title('Intra-Domain Spread')
    ax3.set_ylabel('Average Distance from Center')
    
    # 4. 分离度指标
    ax4 = axes[1, 1]
    
    # 计算分离度指标
    avg_intra_spread = np.mean(spreads)
    
    inter_distances = []
    for i, domain1 in enumerate(domains):
        for j, domain2 in enumerate(domains):
            if i < j:
                dist = np.linalg.norm(domain_centers[domain1] - domain_centers[domain2])
                inter_distances.append(dist)
    
    avg_inter_distance = np.mean(inter_distances)
    separation_ratio = avg_inter_distance / avg_intra_spread if avg_intra_spread > 0 else 0
    
    # 绘制指标
    metrics = ['Intra-Spread', 'Inter-Distance', 'Separation Ratio']
    values = [avg_intra_spread, avg_inter_distance, separation_ratio]
    
    bars = ax4.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax4.set_title('Separation Metrics')
    ax4.set_ylabel('Value')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def analyze_style_distribution(domain_data, dataset_name, llm, describe_nums):
    """分析风格提示的分布"""
    print(f"\n🎨 Analyzing Style Distribution...")
    
    # 收集所有风格分数
    all_style_scores = []
    style_labels = []
    
    for domain, data in domain_data.items():
        style_scores = data['style_scores']  # [n_images, desc_nums]
        all_style_scores.append(style_scores)
        style_labels.extend([domain] * len(style_scores))
    
    all_style_scores = np.vstack(all_style_scores)
    print(f"Style analysis: {all_style_scores.shape[0]} samples, {all_style_scores.shape[1]} style features")
    
    # 计算每个域的风格统计
    print(f"\n📈 Style Statistics by Domain:")
    for domain, data in domain_data.items():
        style_scores = data['style_scores']
        mean_style = np.mean(style_scores)
        std_style = np.std(style_scores)
        max_style = np.max(style_scores)
        
        print(f"   {domain:12}: mean={mean_style:.4f}, std={std_style:.4f}, max={max_style:.4f}")

if __name__ == "__main__":
    # 测试新的分析函数
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        separate_style = len(sys.argv) > 2 and sys.argv[2].lower() == 'true'
        analyze_2d_format_domain_awareness(dataset_name, separate_style=separate_style)
    else:
        print("Usage: python script.py <dataset_name> [separate_style]")
        print("Example: python script.py ProstateSlice true")