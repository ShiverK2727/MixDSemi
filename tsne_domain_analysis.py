#!/usr/bin/env python3
"""
基于CLIP文本-图像匹配分数的t-SNE域感知分析
分析指定LLM和描述数量下的域分离能力
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict
import json

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class CLIPDomainAnalyzer:
    def __init__(self, preprocess_root='/app/MixDSemi/SynFoCLIP/preprocess'):
        self.preprocess_root = preprocess_root
        self.datasets = ['ProstateSlice', 'Fundus', 'MNMS', 'BUSI']
        self.domain_colors = {}
        self.setup_colors()
        
    def setup_colors(self):
        """为不同域设置颜色"""
        # 为每个数据集的域设置不同颜色
        color_palettes = {
            'ProstateSlice': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33'],
            'Fundus': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            'MNMS': ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
            'BUSI': ['#bcbd22', '#17becf']
        }
        
        domain_names = {
            'ProstateSlice': ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL'],
            'Fundus': ['Domain1', 'Domain2', 'Domain3', 'Domain4'],
            'MNMS': ['vendorA', 'vendorB', 'vendorC', 'vendorD'],
            'BUSI': ['benign', 'malignant']
        }
        
        for dataset in self.datasets:
            self.domain_colors[dataset] = {}
            for i, domain in enumerate(domain_names[dataset]):
                self.domain_colors[dataset][domain] = color_palettes[dataset][i]
    
    def load_clip_scores(self, dataset_name, llm='GPT5', describe_nums=80):
        """
        加载指定数据集、LLM和描述数量下的CLIP匹配分数
        
        Returns:
            dict: {domain: {'images': list, 'scores': np.array}}
        """
        dataset_path = os.path.join(self.preprocess_root, dataset_name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        domain_data = {}
        domains = os.listdir(dataset_path)
        
        for domain in domains:
            if os.path.isdir(os.path.join(dataset_path, domain)):
                score_file = os.path.join(dataset_path, domain, f"{llm}_{describe_nums}.pt")
                if os.path.exists(score_file):
                    print(f"Loading scores for {dataset_name}/{domain} from {score_file}")
                    
                    # 加载保存的匹配分数
                    image_text_match = torch.load(score_file, map_location='cpu')
                    
                    images = list(image_text_match.keys())
                    scores = torch.stack(list(image_text_match.values())).numpy()
                    
                    domain_data[domain] = {
                        'images': images,
                        'scores': scores  # shape: [n_images, n_texts]
                    }
                    print(f"  Loaded {len(images)} images with {scores.shape[1]} text features")
                else:
                    print(f"Warning: Score file not found: {score_file}")
        
        return domain_data
    
    def compute_tsne(self, domain_data, n_components=2, perplexity=30, random_state=42):
        """
        对CLIP匹配分数进行t-SNE降维
        
        Args:
            domain_data: {domain: {'images': list, 'scores': np.array}}
            n_components: t-SNE输出维度
            perplexity: t-SNE参数
            random_state: 随机种子
            
        Returns:
            dict: {domain: {'images': list, 'tsne_coords': np.array}}
        """
        # 收集所有数据
        all_scores = []
        all_domains = []
        all_images = []
        
        for domain, data in domain_data.items():
            all_scores.append(data['scores'])
            all_domains.extend([domain] * len(data['images']))
            all_images.extend(data['images'])
        
        # 合并所有分数
        all_scores = np.vstack(all_scores)
        print(f"Computing t-SNE for {all_scores.shape[0]} samples with {all_scores.shape[1]} features...")
        
        # 可选：先用PCA降维以加速t-SNE
        if all_scores.shape[1] > 50:
            print(f"Applying PCA to reduce dimensions from {all_scores.shape[1]} to 50...")
            pca = PCA(n_components=50, random_state=random_state)
            all_scores = pca.fit_transform(all_scores)
        
        # t-SNE降维
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   random_state=random_state, n_iter=1000)
        tsne_coords = tsne.fit_transform(all_scores)
        
        # 按域重新组织结果
        result = {}
        start_idx = 0
        for domain, data in domain_data.items():
            end_idx = start_idx + len(data['images'])
            result[domain] = {
                'images': data['images'],
                'tsne_coords': tsne_coords[start_idx:end_idx]
            }
            start_idx = end_idx
        
        return result
    
    def plot_tsne_domain_analysis(self, dataset_name, llm='GPT5', describe_nums=80, 
                                 save_path=None, figsize=(12, 8)):
        """
        绘制t-SNE域感知分析图
        """
        # 加载数据
        try:
            domain_data = self.load_clip_scores(dataset_name, llm, describe_nums)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        
        if not domain_data:
            print(f"No data found for {dataset_name}")
            return None
        
        # 计算t-SNE
        tsne_results = self.compute_tsne(domain_data)
        
        # 绘制图像
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图: 按域着色的t-SNE图
        ax1 = axes[0]
        for domain, data in tsne_results.items():
            coords = data['tsne_coords']
            color = self.domain_colors[dataset_name][domain]
            ax1.scatter(coords[:, 0], coords[:, 1], 
                       c=color, label=f'{domain} (n={len(coords)})',
                       alpha=0.6, s=20)
        
        ax1.set_title(f'{dataset_name} - Domain Distribution\n{llm} with {describe_nums} descriptions')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 右图: 域间分离度量化分析
        ax2 = axes[1]
        
        # 计算域间距离矩阵和分离度指标
        domain_centers = {}
        domain_spreads = {}
        
        for domain, data in tsne_results.items():
            coords = data['tsne_coords']
            center = np.mean(coords, axis=0)
            spread = np.mean(np.linalg.norm(coords - center, axis=1))
            domain_centers[domain] = center
            domain_spreads[domain] = spread
        
        # 计算域间距离
        domains = list(domain_centers.keys())
        n_domains = len(domains)
        distance_matrix = np.zeros((n_domains, n_domains))
        
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains):
                if i != j:
                    dist = np.linalg.norm(domain_centers[domain1] - domain_centers[domain2])
                    distance_matrix[i, j] = dist
        
        # 绘制距离热力图
        sns.heatmap(distance_matrix, annot=True, fmt='.2f', 
                   xticklabels=domains, yticklabels=domains,
                   cmap='YlOrRd', ax=ax2)
        ax2.set_title('Inter-Domain Distance Matrix')
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Domain')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
        
        # 打印分析结果
        self.print_domain_analysis(tsne_results, dataset_name, llm, describe_nums)
        
        return fig
    
    def print_domain_analysis(self, tsne_results, dataset_name, llm, describe_nums):
        """打印域感知分析结果"""
        print(f"\n" + "="*60)
        print(f"Domain Awareness Analysis: {dataset_name}")
        print(f"LLM: {llm}, Descriptions: {describe_nums}")
        print("="*60)
        
        # 计算域内紧密度和域间分离度
        domain_centers = {}
        domain_spreads = {}
        
        for domain, data in tsne_results.items():
            coords = data['tsne_coords']
            center = np.mean(coords, axis=0)
            spread = np.mean(np.linalg.norm(coords - center, axis=1))
            domain_centers[domain] = center
            domain_spreads[domain] = spread
            
            print(f"{domain:12}: {len(coords):4d} samples, intra-spread: {spread:.3f}")
        
        # 计算平均域间距离
        domains = list(domain_centers.keys())
        inter_distances = []
        
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains):
                if i < j:
                    dist = np.linalg.norm(domain_centers[domain1] - domain_centers[domain2])
                    inter_distances.append(dist)
                    print(f"{domain1} <-> {domain2}: distance = {dist:.3f}")
        
        avg_intra_spread = np.mean(list(domain_spreads.values()))
        avg_inter_distance = np.mean(inter_distances)
        separation_ratio = avg_inter_distance / avg_intra_spread
        
        print(f"\nSummary Metrics:")
        print(f"Average intra-domain spread: {avg_intra_spread:.3f}")
        print(f"Average inter-domain distance: {avg_inter_distance:.3f}")
        print(f"Separation ratio (higher is better): {separation_ratio:.3f}")
        
        if separation_ratio > 2.0:
            print("✅ Good domain separation detected!")
        elif separation_ratio > 1.5:
            print("🟡 Moderate domain separation")
        else:
            print("❌ Poor domain separation")
    
    def batch_analyze_all_datasets(self, llm='GPT5', describe_nums=80, 
                                  output_dir='tsne_analysis'):
        """批量分析所有数据集的域感知能力"""
        os.makedirs(output_dir, exist_ok=True)
        
        results_summary = []
        
        for dataset_name in self.datasets:
            print(f"\n{'='*60}")
            print(f"Analyzing {dataset_name}...")
            print('='*60)
            
            try:
                save_path = os.path.join(output_dir, f"{dataset_name}_{llm}_{describe_nums}_tsne.png")
                fig = self.plot_tsne_domain_analysis(dataset_name, llm, describe_nums, save_path)
                
                if fig:
                    results_summary.append({
                        'dataset': dataset_name,
                        'llm': llm,
                        'describe_nums': describe_nums,
                        'plot_saved': save_path,
                        'status': 'success'
                    })
                else:
                    results_summary.append({
                        'dataset': dataset_name,
                        'llm': llm,
                        'describe_nums': describe_nums,
                        'plot_saved': None,
                        'status': 'failed'
                    })
                    
            except Exception as e:
                print(f"Error analyzing {dataset_name}: {e}")
                results_summary.append({
                    'dataset': dataset_name,
                    'llm': llm,
                    'describe_nums': describe_nums,
                    'plot_saved': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        # 保存结果总结
        summary_file = os.path.join(output_dir, f"analysis_summary_{llm}_{describe_nums}.json")
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Batch Analysis Complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Summary saved to: {summary_file}")
        print('='*60)
        
        return results_summary

def main():
    """主函数 - 示例用法"""
    analyzer = CLIPDomainAnalyzer()
    
    # 示例1: 分析单个数据集
    print("Example 1: Analyzing ProstateSlice dataset")
    analyzer.plot_tsne_domain_analysis('ProstateSlice', llm='GPT5', describe_nums=80)
    
    # 示例2: 批量分析所有数据集
    print("\nExample 2: Batch analysis of all datasets")
    analyzer.batch_analyze_all_datasets(llm='GPT5', describe_nums=80)

if __name__ == "__main__":
    main()