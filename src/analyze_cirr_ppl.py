"""
Analyze CIRR Test Set with PPL-based Perturbation Detection
对CIRR测试集进行困惑度分析，区分干净样本和扰动样本
"""

import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ppl_detector import PPLDetector

# 设置路径
base_path = Path(__file__).resolve().parent.parent
cirr_caption_path = base_path / 'cirr_dataset' / 'cirr' / 'captions'


def load_cirr_captions(split='test1'):
    """
    加载CIRR数据集的caption
    
    Args:
        split: 'train', 'val', 'test1'
    
    Returns:
        captions: caption文本列表
        triplets: 完整的triplet数据（包含pair_id等信息）
        valid_indices: 有效样本的索引列表
    """
    caption_file = cirr_caption_path / f'cap.rc2.{split}.json'
    
    if not caption_file.exists():
        raise FileNotFoundError(f"Caption file not found: {caption_file}")
    
    print(f"Loading CIRR {split} captions from {caption_file}...")
    with open(caption_file, 'r') as f:
        triplets = json.load(f)
    
    # 提取所有caption文本，并过滤空文本
    captions = []
    valid_triplets = []
    valid_indices = []
    empty_count = 0
    
    for i, item in enumerate(triplets):
        caption = item.get('caption', '').strip()
        if caption:  # 只保留非空文本
            captions.append(caption)
            valid_triplets.append(item)
            valid_indices.append(i)
        else:
            empty_count += 1
            print(f"Warning: Empty caption at index {i}, pairid={item.get('pairid', 'N/A')}")
    
    print(f"Loaded {len(captions)} valid captions from CIRR {split} set")
    if empty_count > 0:
        print(f"Filtered out {empty_count} empty captions")
    
    return captions, valid_triplets, valid_indices


def analyze_cirr_ppl(split='test1', 
                     threshold=None, 
                     percentile=95,
                     save_dir='./cirr_ppl_analysis',
                     model_name='gpt2'):
    """
    分析CIRR数据集的PPL分布
    
    Args:
        split: 数据集split
        threshold: PPL阈值（None则自动计算）
        percentile: 自动阈值的百分位数
        save_dir: 保存结果的目录
        model_name: GPT模型名称
    
    Returns:
        results: 分析结果字典
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    captions, triplets, valid_indices = load_cirr_captions(split)
    
    # 初始化PPL检测器
    print(f"\nInitializing PPL Detector with {model_name}...")
    detector = PPLDetector(model_name=model_name, device='cuda')
    
    # 计算PPL
    print(f"\nComputing perplexity for {len(captions)} captions...")
    ppls = detector.compute_perplexity(captions, batch_size=16)
    
    # 处理nan值
    nan_mask = np.isnan(ppls)
    nan_count = np.sum(nan_mask)
    valid_mask = ~nan_mask
    ppls_valid = ppls[valid_mask]
    
    if nan_count > 0:
        print(f"\n⚠️  Warning: {nan_count} samples have NaN PPL (likely computation errors)")
    
    if len(ppls_valid) == 0:
        raise ValueError("All PPL values are NaN! Cannot proceed.")
    
    # 统计信息（只用有效值）
    print(f"\n{'='*60}")
    print(f"PPL Statistics for CIRR {split.upper()}")
    print(f"{'='*60}")
    print(f"Total samples: {len(ppls)}")
    print(f"Valid samples: {len(ppls_valid)} ({100*len(ppls_valid)/len(ppls):.1f}%)")
    print(f"Mean PPL: {ppls_valid.mean():.2f} ± {ppls_valid.std():.2f}")
    print(f"Median PPL: {np.median(ppls_valid):.2f}")
    print(f"Min/Max PPL: {ppls_valid.min():.2f} / {ppls_valid.max():.2f}")
    print(f"Percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(ppls_valid, p):.2f}")
    
    # 确定阈值（使用有效值）
    if threshold is None:
        # 使用更智能的自适应阈值策略
        p50 = np.percentile(ppls_valid, 50)  # 中位数
        p75 = np.percentile(ppls_valid, 75)
        p90 = np.percentile(ppls_valid, 90)
        p95 = np.percentile(ppls_valid, 95)
        p99 = np.percentile(ppls_valid, 99)
        
        # 检测是否存在显著的长尾分布（表明有攻击样本）
        if p99 > 10 * p75:  # 强烈的长尾，存在极端outliers
            # 使用基于IQR的策略（更鲁棒）
            q1 = np.percentile(ppls_valid, 25)
            q3 = np.percentile(ppls_valid, 75)
            iqr = q3 - q1
            # 使用1.5 IQR规则，但设置合理的上下界
            threshold = min(q3 + 3 * iqr, 500)  # 上限500，避免过于严格
            threshold = max(threshold, 100)  # 下限100，避免误判正常文本
            print(f"\n📊 Detected long-tail distribution (attack samples likely present)")
            print(f"   Using IQR-based threshold: {threshold:.2f}")
            print(f"   (Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f})")
        elif p95 > 3 * p75:  # 中等长尾
            # 使用P90作为阈值（更保守）
            threshold = min(p90, 300)
            print(f"\nUsing P90 threshold (conservative): {threshold:.2f}")
        else:  # 正常分布
            threshold = np.percentile(ppls_valid, percentile)
            print(f"\nAuto-selected threshold (P{percentile}): {threshold:.2f}")
    else:
        print(f"\nUsing manual threshold: {threshold:.2f}")
    
    # 分类（nan值单独处理，不计入clean或perturbed）
    is_clean = ppls < threshold
    # nan值标记为False（既不是clean也不是perturbed）
    is_clean[nan_mask] = False
    
    clean_indices = np.where(is_clean)[0]
    perturbed_indices = np.where((~is_clean) & valid_mask)[0]  # 只统计有效的perturbed
    nan_indices = np.where(nan_mask)[0]
    
    print(f"\n{'='*60}")
    print(f"Classification Results")
    print(f"{'='*60}")
    print(f"Clean samples: {len(clean_indices)} ({100*len(clean_indices)/len(ppls):.1f}%)")
    print(f"Perturbed samples: {len(perturbed_indices)} ({100*len(perturbed_indices)/len(ppls):.1f}%)")
    if nan_count > 0:
        print(f"NaN samples: {nan_count} ({100*nan_count/len(ppls):.1f}%)")
    
    # 显示高PPL样本（可能是扰动）
    print(f"\n{'='*60}")
    print(f"Top 10 Highest PPL Samples (Likely Perturbed)")
    print(f"{'='*60}")
    top_indices = np.argsort(ppls)[-10:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank}. PPL={ppls[idx]:.2f}")
        print(f"   Caption: {captions[idx]}")
        if 'pairid' in triplets[idx]:
            print(f"   Pair ID: {triplets[idx]['pairid']}")
        print()
    
    # 显示低PPL样本（正常）
    print(f"{'='*60}")
    print(f"Top 10 Lowest PPL Samples (Likely Clean)")
    print(f"{'='*60}")
    bottom_indices = np.argsort(ppls)[:10]
    for rank, idx in enumerate(bottom_indices, 1):
        print(f"{rank}. PPL={ppls[idx]:.2f}")
        print(f"   Caption: {captions[idx]}")
        if 'pairid' in triplets[idx]:
            print(f"   Pair ID: {triplets[idx]['pairid']}")
        print()
    
    # 绘制分布图
    print("Plotting PPL distribution...")
    plot_cirr_ppl_distribution(ppls, threshold, split, save_dir)
    
    # 保存结果
    results = {
        'split': split,
        'total_samples': len(ppls),
        'valid_samples': len(ppls_valid),
        'nan_samples': int(nan_count),
        'threshold': float(threshold),
        'n_clean': int(len(clean_indices)),
        'n_perturbed': int(len(perturbed_indices)),
        'ppl_mean': float(ppls_valid.mean()),
        'ppl_std': float(ppls_valid.std()),
        'ppl_median': float(np.median(ppls_valid)),
        'clean_indices': clean_indices.tolist(),
        'perturbed_indices': perturbed_indices.tolist(),
        'nan_indices': nan_indices.tolist() if nan_count > 0 else [],
        'all_ppls': ppls.tolist()
    }
    
    # 保存JSON结果
    results_file = os.path.join(save_dir, f'cirr_{split}_ppl_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # 保存被标记为扰动的样本详细信息
    perturbed_samples = []
    for idx in perturbed_indices:
        sample = {
            'index': int(idx),
            'ppl': float(ppls[idx]),
            'caption': captions[idx],
            'reference': triplets[idx]['reference'],
        }
        if 'pairid' in triplets[idx]:
            sample['pairid'] = triplets[idx]['pairid']
        if 'target_hard' in triplets[idx]:
            sample['target_hard'] = triplets[idx]['target_hard']
        perturbed_samples.append(sample)
    
    perturbed_file = os.path.join(save_dir, f'cirr_{split}_perturbed_samples.json')
    with open(perturbed_file, 'w') as f:
        json.dump(perturbed_samples, f, indent=2)
    print(f"Perturbed samples saved to: {perturbed_file}")
    
    return results


def plot_cirr_ppl_distribution(ppls, threshold, split, save_dir):
    """绘制PPL分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 子图1: 直方图
    ax = axes[0]
    ax.hist(ppls, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold={threshold:.1f}')
    ax.set_xlabel('Perplexity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'CIRR {split.upper()} PPL Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图2: Log scale直方图
    ax = axes[1]
    ax.hist(ppls, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold={threshold:.1f}')
    ax.set_xlabel('Perplexity', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title(f'CIRR {split.upper()} PPL Distribution (Log)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 子图3: CDF
    ax = axes[2]
    sorted_ppls = np.sort(ppls)
    cdf = np.arange(1, len(sorted_ppls) + 1) / len(sorted_ppls)
    ax.plot(sorted_ppls, cdf, linewidth=2, color='steelblue')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold={threshold:.1f}')
    ax.axhline(np.sum(ppls < threshold) / len(ppls), 
               color='green', linestyle=':', linewidth=1.5,
               label=f'Clean: {100*np.sum(ppls < threshold)/len(ppls):.1f}%')
    ax.set_xlabel('Perplexity', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title(f'Cumulative Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, f'cirr_{split}_ppl_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def compare_splits(save_dir='./cirr_ppl_analysis', model_name='gpt2'):
    """比较不同split的PPL分布"""
    results = {}
    
    for split in ['train', 'val', 'test1']:
        print(f"\n{'#'*70}")
        print(f"Analyzing CIRR {split.upper()} split")
        print(f"{'#'*70}\n")
        
        try:
            result = analyze_cirr_ppl(
                split=split,
                save_dir=os.path.join(save_dir, split),
                model_name=model_name
            )
            results[split] = result
        except Exception as e:
            print(f"Error analyzing {split}: {e}")
            continue
    
    # 打印汇总表
    print(f"\n\n{'='*80}")
    print("SUMMARY: PPL Analysis Across All Splits")
    print(f"{'='*80}")
    print(f"{'Split':<10} {'Total':>8} {'Clean':>8} {'Perturbed':>10} {'Mean PPL':>10} {'Threshold':>10}")
    print(f"{'-'*80}")
    
    for split, res in results.items():
        print(f"{split:<10} {res['total_samples']:>8} {res['n_clean']:>8} "
              f"{res['n_perturbed']:>10} {res['ppl_mean']:>10.2f} {res['threshold']:>10.2f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze CIRR captions with PPL')
    parser.add_argument('--split', type=str, default='test1', 
                       choices=['train', 'val', 'test1', 'all'],
                       help='Dataset split to analyze')
    parser.add_argument('--threshold', type=float, default=None,
                       help='PPL threshold (None for auto)')
    parser.add_argument('--percentile', type=float, default=95,
                       help='Percentile for auto threshold')
    parser.add_argument('--model', type=str, default='gpt2',
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
                       help='GPT model for PPL computation')
    parser.add_argument('--save_dir', type=str, default='./cirr_ppl_analysis',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.split == 'all':
        # 分析所有split
        compare_splits(save_dir=args.save_dir, model_name=args.model)
    else:
        # 分析单个split
        analyze_cirr_ppl(
            split=args.split,
            threshold=args.threshold,
            percentile=args.percentile,
            save_dir=args.save_dir,
            model_name=args.model
        )
    
    print("\n✅ Analysis completed!")