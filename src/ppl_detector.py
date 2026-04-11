"""
PPL-based Text Perturbation Detector
基于困惑度的文本扰动检测器

用于检测对抗性文本攻击（字符级、词级扰动）
适用于Composed Image Retrieval的鲁棒性研究
"""

import torch
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import os


class PPLDetector:
    """基于困惑度的文本扰动检测器"""
    
    def __init__(self, model_name: str = 'gpt2', device: str = 'cuda'):
        """
        初始化PPL检测器
        
        Args:
            model_name: 预训练语言模型名称
                       'gpt2' (124M, 快速)
                       'gpt2-medium' (355M, 平衡)
                       'gpt2-large' (774M, 精确)
            device: 'cuda' 或 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[PPL Detector] Loading {model_name} on {self.device}...")
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[PPL Detector] Model loaded successfully!")
    
    @torch.no_grad()
    def compute_perplexity(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        计算文本列表的困惑度
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小（加速计算）
        
        Returns:
            ppls: 困惑度数组，shape [len(texts)]
        """
        ppls = []
        
        # 批处理计算
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize with padding
            encodings = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=77  # CLIP text length
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**encodings, labels=encodings.input_ids)
            
            # 处理批量loss
            # outputs.loss是平均loss，需要对每个样本单独计算
            logits = outputs.logits  # [batch, seq_len, vocab]
            labels = encodings.input_ids  # [batch, seq_len]
            
            # 计算每个样本的PPL
            for j in range(len(batch_texts)):
                # 获取单个样本的有效长度（排除padding）
                attention_mask = encodings.attention_mask[j]
                valid_length = attention_mask.sum().item()
                
                if valid_length == 0:
                    ppls.append(float('inf'))
                    continue
                
                # 计算单个样本的loss
                shift_logits = logits[j, :-1, :].contiguous()
                shift_labels = labels[j, 1:].contiguous()
                shift_mask = attention_mask[j, 1:].contiguous()
                
                # 只计算有效token的loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(shift_logits, shift_labels)
                
                # 使用mask过滤padding
                masked_losses = losses * shift_mask
                avg_loss = masked_losses.sum() / shift_mask.sum()
                
                # PPL = exp(loss)
                ppl = torch.exp(avg_loss).item()
                ppls.append(ppl)
        
        return np.array(ppls)
    
    def detect_perturbations(
        self, 
        texts: List[str], 
        threshold: Optional[float] = None,
        percentile: float = 95
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        检测扰动文本
        
        Args:
            texts: 待检测文本列表
            threshold: PPL阈值（None则自动计算）
            percentile: 自动阈值的百分位数
        
        Returns:
            is_clean: bool数组，True表示干净文本
            ppls: 困惑度数组
            threshold: 使用的阈值
        """
        ppls = self.compute_perplexity(texts)
        
        # 自动阈值
        if threshold is None:
            threshold = np.percentile(ppls, percentile)
            print(f"[PPL Detector] Auto threshold (P{percentile}): {threshold:.2f}")
        
        is_clean = ppls < threshold
        
        # 统计信息
        n_clean = is_clean.sum()
        n_total = len(texts)
        print(f"[PPL Detector] Clean: {n_clean}/{n_total} ({100*n_clean/n_total:.1f}%)")
        
        return is_clean, ppls, threshold
    
    def analyze_distribution(
        self,
        clean_texts: List[str],
        perturbed_texts: List[str],
        save_dir: Optional[str] = None,
        attack_name: str = "unknown"
    ) -> Dict:
        """
        分析干净文本和扰动文本的PPL分布
        
        Args:
            clean_texts: 干净文本列表
            perturbed_texts: 扰动文本列表
            save_dir: 保存结果的目录
            attack_name: 攻击方法名称
        
        Returns:
            results: 包含统计信息的字典
        """
        print(f"\n{'='*60}")
        print(f"PPL Distribution Analysis: {attack_name}")
        print(f"{'='*60}")
        
        print("Computing PPL for clean texts...")
        clean_ppls = self.compute_perplexity(clean_texts)
        
        print("Computing PPL for perturbed texts...")
        perturbed_ppls = self.compute_perplexity(perturbed_texts)
        
        # 统计信息
        print(f"\n--- Clean Texts ---")
        print(f"  Count: {len(clean_ppls)}")
        print(f"  Mean: {clean_ppls.mean():.2f} ± {clean_ppls.std():.2f}")
        print(f"  Median: {np.median(clean_ppls):.2f}")
        print(f"  Min/Max: {clean_ppls.min():.2f} / {clean_ppls.max():.2f}")
        
        print(f"\n--- Perturbed Texts ({attack_name}) ---")
        print(f"  Count: {len(perturbed_ppls)}")
        print(f"  Mean: {perturbed_ppls.mean():.2f} ± {perturbed_ppls.std():.2f}")
        print(f"  Median: {np.median(perturbed_ppls):.2f}")
        print(f"  Min/Max: {perturbed_ppls.min():.2f} / {perturbed_ppls.max():.2f}")
        
        # 计算分离度
        separation = (perturbed_ppls.mean() - clean_ppls.mean()) / clean_ppls.std()
        print(f"\n--- Separability ---")
        print(f"  Cohen's d: {separation:.2f}")
        
        # ROC分析
        from sklearn.metrics import roc_curve, auc, accuracy_score
        
        y_true = np.concatenate([
            np.zeros(len(clean_ppls)),  # 0 = clean
            np.ones(len(perturbed_ppls))  # 1 = perturbed
        ])
        y_scores = np.concatenate([clean_ppls, perturbed_ppls])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 最优阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\n--- Detection Performance ---")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.2f}")
        print(f"  TPR (Detection Rate): {tpr[optimal_idx]:.2%}")
        print(f"  FPR (False Alarm): {fpr[optimal_idx]:.2%}")
        print(f"  Accuracy: {accuracy_score(y_true, y_scores > optimal_threshold):.2%}")
        
        # 绘图
        self._plot_distribution(
            clean_ppls, 
            perturbed_ppls, 
            optimal_threshold,
            attack_name,
            save_dir
        )
        
        results = {
            'clean_ppls': clean_ppls,
            'perturbed_ppls': perturbed_ppls,
            'optimal_threshold': optimal_threshold,
            'roc_auc': roc_auc,
            'tpr': tpr[optimal_idx],
            'fpr': fpr[optimal_idx],
            'separation': separation
        }
        
        return results
    
    def _plot_distribution(
        self,
        clean_ppls: np.ndarray,
        perturbed_ppls: np.ndarray,
        threshold: float,
        attack_name: str,
        save_dir: Optional[str]
    ):
        """绘制PPL分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 子图1: 直方图
        ax = axes[0]
        ax.hist(clean_ppls, bins=50, alpha=0.6, label='Clean', 
                density=True, color='green', edgecolor='black')
        ax.hist(perturbed_ppls, bins=50, alpha=0.6, label=f'Perturbed ({attack_name})', 
                density=True, color='red', edgecolor='black')
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, 
                   label=f'Threshold={threshold:.1f}')
        ax.set_xlabel('Perplexity', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('PPL Distribution (log scale)', fontsize=14)
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 子图2: CDF
        ax = axes[1]
        sorted_clean = np.sort(clean_ppls)
        sorted_pert = np.sort(perturbed_ppls)
        ax.plot(sorted_clean, np.arange(len(sorted_clean))/len(sorted_clean), 
                label='Clean', linewidth=2, color='green')
        ax.plot(sorted_pert, np.arange(len(sorted_pert))/len(sorted_pert), 
                label=f'Perturbed ({attack_name})', linewidth=2, color='red')
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, 
                   label=f'Threshold={threshold:.1f}')
        ax.set_xlabel('Perplexity', fontsize=12)
        ax.set_ylabel('CDF', fontsize=12)
        ax.set_title('Cumulative Distribution', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 子图3: Box Plot
        ax = axes[2]
        bp = ax.boxplot([clean_ppls, perturbed_ppls], 
                        labels=['Clean', f'Perturbed\n({attack_name})'],
                        patch_artist=True,
                        widths=0.6)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        for box in bp['boxes']:
            box.set_alpha(0.6)
        ax.axhline(threshold, color='blue', linestyle='--', linewidth=2, 
                   label=f'Threshold={threshold:.1f}')
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title('Box Plot Comparison', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'ppl_distribution_{attack_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[PPL Detector] Plot saved to: {save_path}")
        
        plt.show()
    
    def batch_analyze(
        self,
        clean_texts: List[str],
        attack_results: Dict[str, List[str]],
        save_dir: str = './ppl_analysis'
    ):
        """
        批量分析多种攻击方法
        
        Args:
            clean_texts: 干净文本列表
            attack_results: {attack_name: perturbed_texts} 字典
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        summary = {}
        
        for attack_name, perturbed_texts in attack_results.items():
            print(f"\n\n{'#'*70}")
            print(f"Analyzing: {attack_name}")
            print(f"{'#'*70}")
            
            results = self.analyze_distribution(
                clean_texts,
                perturbed_texts,
                save_dir=save_dir,
                attack_name=attack_name
            )
            
            summary[attack_name] = results
        
        # 打印汇总表
        print(f"\n\n{'='*80}")
        print("SUMMARY: PPL-based Detection Performance")
        print(f"{'='*80}")
        print(f"{'Attack Method':<20} {'ROC AUC':>10} {'Threshold':>12} {'TPR':>8} {'FPR':>8}")
        print(f"{'-'*80}")
        
        for attack_name, res in summary.items():
            print(f"{attack_name:<20} {res['roc_auc']:>10.4f} {res['optimal_threshold']:>12.2f} "
                  f"{res['tpr']:>8.2%} {res['fpr']:>8.2%}")
        
        return summary


def example_usage():
    """使用示例"""
    
    # 初始化检测器
    detector = PPLDetector(model_name='gpt2', device='cuda')
    
    # 示例数据
    clean_texts = [
        "a man in a blue shirt",
        "the dog is running in the park",
        "change the color to red"
    ]
    
    # 模拟扰动文本（字符级攻击）
    perturbed_texts = [
        "a m@n in a bl_ue sh1rt",
        "the d0g is runn1ng in the p@rk",
        "chang3 the c0lor to r3d"
    ]
    
    # 方法1: 简单检测
    is_clean, ppls, threshold = detector.detect_perturbations(clean_texts)
    print(f"Clean flags: {is_clean}")
    print(f"PPLs: {ppls}")
    
    # 方法2: 分布分析
    results = detector.analyze_distribution(
        clean_texts,
        perturbed_texts,
        save_dir='./ppl_results',
        attack_name='CharSwap'
    )
    
    print(f"\nOptimal threshold: {results['optimal_threshold']:.2f}")
    print(f"Detection AUC: {results['roc_auc']:.4f}")


if __name__ == "__main__":
    example_usage()