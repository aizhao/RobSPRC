"""
CIRR Dataset Defense Pipeline
处理CIRR数据集，检测并修复扰动文本
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from ppl_detector import PPLDetector
from text_corrector import TextCorrector

# 路径配置
base_path = Path(__file__).resolve().parent.parent
cirr_caption_path = base_path / 'cirr_dataset' / 'cirr' / 'captions'


class CIRRDefensePipeline:
    """CIRR数据集防御流程"""
    
    def __init__(
        self,
        ppl_threshold=300,
        corrector_method='qwen',
        corrector_model_path=None
    ):
        """
        初始化
        
        Args:
            ppl_threshold: PPL阈值
            corrector_method: 修复方法 ('qwen', 'gpt', 'llama')
            corrector_model_path: 修复模型路径
        """
        print("="*70)
        print("Initializing CIRR Defense Pipeline")
        print("="*70)
        
        # PPL检测器
        print("\n[1/2] Loading PPL Detector...")
        self.ppl_detector = PPLDetector(model_name='gpt2', device='cuda')
        self.ppl_threshold = ppl_threshold
        
        # 文本修复器
        print("\n[2/2] Loading Text Corrector...")
        self.text_corrector = TextCorrector(
            method=corrector_method,
            model_path=corrector_model_path
        )
        
        print(f"\n✓ Pipeline initialized (threshold={ppl_threshold})")
        print("="*70)
    
    def load_cirr_data(self, split='test2'):
        """
        加载CIRR数据
        
        Args:
            split: 'train', 'val', 'test1'
        
        Returns:
            data: 原始数据列表
        """
        caption_file = cirr_caption_path / f'cap.rc2.{split}.json'
        
        if not caption_file.exists():
            raise FileNotFoundError(f"Caption file not found: {caption_file}")
        
        print(f"\nLoading CIRR {split} data from: {caption_file}")
        with open(caption_file, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} samples")
        
        return data
    
    def process_cirr_split(
        self,
        split='test2',
        save_dir='./cirr_defense_results'
    ):
        """
        处理CIRR数据集的一个split
        
        Args:
            split: 数据集split
            save_dir: 保存目录
        
        Returns:
            results: 处理结果
        """
        print(f"\n{'='*70}")
        print(f"Processing CIRR {split.upper()} Split")
        print(f"{'='*70}")
        
        # 加载数据
        data = self.load_cirr_data(split)
        
        # 提取caption
        captions = []
        valid_indices = []
        for i, item in enumerate(data):
            caption = item.get('caption', '').strip()
            if caption:
                captions.append(caption)
                valid_indices.append(i)
        
        print(f"\nValid captions: {len(captions)}/{len(data)}")
        
        # Step 1: PPL检测
        print(f"\n{'─'*70}")
        print("[Step 1/3] Detecting perturbations with PPL...")
        print(f"{'─'*70}")
        
        ppls = self.ppl_detector.compute_perplexity(captions, batch_size=16)
        is_perturbed = ppls > self.ppl_threshold
        
        n_perturbed = is_perturbed.sum()
        n_total = len(captions)
        
        print(f"\n📊 Detection Results:")
        print(f"   Total samples: {n_total}")
        print(f"   Clean samples: {n_total - n_perturbed} ({100*(n_total-n_perturbed)/n_total:.1f}%)")
        print(f"   Perturbed samples: {n_perturbed} ({100*n_perturbed/n_total:.1f}%)")
        print(f"   Mean PPL (all): {ppls.mean():.2f}")
        if n_perturbed > 0:
            print(f"   Mean PPL (perturbed): {ppls[is_perturbed].mean():.2f}")
        
        # Step 2: 修复扰动文本
        print(f"\n{'─'*70}")
        print(f"[Step 2/3] Correcting {n_perturbed} perturbed texts...")
        print(f"{'─'*70}")
        
        clean_captions = []
        corrections = []
        
        for i, (caption, is_pert, ppl) in enumerate(tqdm(
            zip(captions, is_perturbed, ppls),
            total=len(captions),
            desc="Processing"
        )):
            if is_pert:
                # 需要修复
                try:
                    corrected, is_changed = self.text_corrector.correct_single(caption)
                    clean_captions.append(corrected)
                    
                    corrections.append({
                        'original_index': valid_indices[i],
                        'ppl': float(ppl),
                        'original': caption,
                        'corrected': corrected,
                        'is_changed': is_changed
                    })
                except Exception as e:
                    print(f"\n⚠️  Error correcting index {i}: {e}")
                    clean_captions.append(caption)  # 保留原文
            else:
                # 无需修复
                clean_captions.append(caption)
        
        # Step 3: 验证修复效果
        print(f"\n{'─'*70}")
        print("[Step 3/3] Verifying corrections...")
        print(f"{'─'*70}")
        
        n_fixed = 0
        if n_perturbed > 0:
            corrected_texts = [c['corrected'] for c in corrections]
            ppls_after = self.ppl_detector.compute_perplexity(corrected_texts, batch_size=16)
            
            n_fixed = (ppls_after < self.ppl_threshold).sum()
            
            print(f"\n✅ Verification Results:")
            print(f"   Successfully fixed: {n_fixed}/{n_perturbed} ({100*n_fixed/n_perturbed:.1f}%)")
            print(f"   PPL reduction: {ppls[is_perturbed].mean():.1f} → {ppls_after.mean():.1f}")
            print(f"   Average PPL drop: {ppls[is_perturbed].mean() - ppls_after.mean():.1f}")
            
            # 更新corrections中的after_ppl
            for i, corr in enumerate(corrections):
                corr['ppl_after'] = float(ppls_after[i])
                corr['ppl_reduction'] = float(ppls[is_perturbed][i] - ppls_after[i])
        
        # 构建输出数据（保持原始结构）
        output_data = []
        clean_idx = 0
        
        for i, item in enumerate(data):
            if i in valid_indices:
                # 更新caption为修复后的版本
                new_item = item.copy()
                new_item['caption'] = clean_captions[clean_idx]
                new_item['original_caption'] = item['caption']  # 保留原始caption
                output_data.append(new_item)
                clean_idx += 1
            else:
                # 空caption，保持原样
                output_data.append(item)
        
        # 保存结果
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 保存修复后的完整数据集
        output_file = os.path.join(save_dir, f'cap.rc2.{split}.clean.json')
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Cleaned dataset saved to: {output_file}")
        
        # 2. 保存修复记录
        corrections_file = os.path.join(save_dir, f'{split}_corrections.json')
        with open(corrections_file, 'w') as f:
            json.dump(corrections, f, indent=2, ensure_ascii=False)
        print(f"💾 Corrections log saved to: {corrections_file}")
        
        # 3. 保存统计信息
        stats = {
            'split': split,
            'n_total': n_total,
            'n_clean': int(n_total - n_perturbed),
            'n_perturbed': int(n_perturbed),
            'n_fixed': int(n_fixed),
            'ppl_threshold': self.ppl_threshold,
            'mean_ppl_before': float(ppls.mean()),
            'mean_ppl_perturbed_before': float(ppls[is_perturbed].mean()) if n_perturbed > 0 else 0,
            'mean_ppl_perturbed_after': float(ppls_after.mean()) if n_perturbed > 0 else 0,
        }
        
        stats_file = os.path.join(save_dir, f'{split}_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"💾 Statistics saved to: {stats_file}")
        
        return {
            'clean_data': output_data,
            'corrections': corrections,
            'stats': stats
        }
    
    def show_correction_samples(self, corrections, n=10):
        """展示修复样本"""
        print(f"\n{'='*70}")
        print(f"Top {n} Correction Examples")
        print(f"{'='*70}")
        
        # 按PPL降序排列
        sorted_corrections = sorted(corrections, key=lambda x: x['ppl'], reverse=True)
        
        for i, corr in enumerate(sorted_corrections[:n], 1):
            ppl_before = corr['ppl']
            ppl_after = corr.get('ppl_after', 0)
            ppl_red = corr.get('ppl_reduction', 0)
            
            print(f"\n[{i}] PPL: {ppl_before:.1f} → {ppl_after:.1f} (↓{ppl_red:.1f})")
            print(f"    Original:  {corr['original']}")
            print(f"    Corrected: {corr['corrected']}")
            if not corr['is_changed']:
                print(f"    ⚠️  No change made by LLM")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CIRR Defense Pipeline')
    parser.add_argument('--split', type=str, default='test2',
                       choices=['train', 'val', 'test1','test2'],
                       help='Dataset split to process')
    parser.add_argument('--threshold', type=float, default=300,
                       help='PPL threshold')
    parser.add_argument('--corrector', type=str, default='qwen',
                       choices=['qwen', 'gpt', 'llama'],
                       help='Text corrector method')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to corrector model (for qwen/llama)')
    parser.add_argument('--save_dir', type=str, default='./cirr_defense_results',
                       help='Directory to save results')
    parser.add_argument('--show_samples', type=int, default=10,
                       help='Number of correction samples to show')
    
    args = parser.parse_args()
    
    # 初始化流程
    pipeline = CIRRDefensePipeline(
        ppl_threshold=args.threshold,
        corrector_method=args.corrector,
        corrector_model_path=args.model_path
    )
    
    # 处理数据
    results = pipeline.process_cirr_split(
        split=args.split,
        save_dir=args.save_dir
    )
    
    # 展示修复样本
    if results['corrections']:
        pipeline.show_correction_samples(
            results['corrections'],
            n=args.show_samples
        )
    
    print(f"\n{'='*70}")
    print("✅ CIRR Defense Pipeline Completed!")
    print(f"{'='*70}")
    print(f"\n📁 Output files:")
    print(f"   - Clean dataset: {args.save_dir}/cap.rc2.{args.split}.clean.json")
    print(f"   - Corrections log: {args.save_dir}/{args.split}_corrections.json")
    print(f"   - Statistics: {args.save_dir}/{args.split}_stats.json")
    print(f"\n💡 Next step: Use the clean dataset for CIR validation")
    print(f"   python validate_blip.py --clean_captions {args.save_dir}/cap.rc2.{args.split}.clean.json")


if __name__ == "__main__":
    main()




