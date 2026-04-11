"""
CIRR Defense Pipeline with Unified VLM
使用统一VLM的CIRR防御流程（PPL检测 + 迭代修复）
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from unified_vlm_corrector import UnifiedVLMCorrector


class CIRRDefenseUnified:
    """统一VLM的CIRR防御系统"""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        ppl_threshold: float = 50.0,
        max_iterations: int = 3,
        device: str = 'cuda'
    ):
        """
        初始化防御系统
        
        Args:
            model_path: VLM模型路径
                - "Qwen/Qwen2-VL-2B-Instruct" (推荐，2B，支持图像)
                - "Qwen/Qwen-VL-Chat" (7B，支持图像)
                - "Qwen/Qwen2.5-3B-Instruct" (3B，纯文本)
            ppl_threshold: PPL阈值
            max_iterations: 最大迭代次数
            device: 设备
        """
        self.corrector = UnifiedVLMCorrector(
            model_path=model_path,
            ppl_threshold=ppl_threshold,
            max_iterations=max_iterations,
            device=device
        )
        
        self.base_path = Path(__file__).parent.parent
    
    def load_cirr_data(self, split: str = 'test4'):
        """
        加载CIRR数据
        
        Args:
            split: 数据集分割 ('test1', 'val', 'train')
        
        Returns:
            captions: caption列表
            image_paths: 图像路径列表
            triplets: 原始triplets数据
            name_to_relpath: 图像名到相对路径的映射
        """
        print(f"\nLoading CIRR {split} dataset...")
        
        # 加载captions
        caption_file = self.base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json'
        
        if not caption_file.exists():
            raise FileNotFoundError(f"Caption file not found: {caption_file}")
        
        with open(caption_file, 'r') as f:
            triplets = json.load(f)
        
        # 加载image splits
        splits_file = self.base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json'
        
        if not splits_file.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
        
        with open(splits_file, 'r') as f:
            name_to_relpath = json.load(f)
        
        # 提取数据
        captions = []
        image_paths = []
        valid_indices = []
        
        for i, item in enumerate(triplets):
            caption = item.get('caption', '').strip()
            reference_name = item.get('reference', '')
            
            if caption and reference_name in name_to_relpath:
                captions.append(caption)
                
                # 构建完整图像路径
                rel_path = name_to_relpath[reference_name]
                full_path = self.base_path / 'cirr_dataset' / rel_path
                image_paths.append(str(full_path))
                
                valid_indices.append(i)
        
        print(f"✓ Loaded {len(captions)} samples from {split} split")
        
        return captions, image_paths, triplets, name_to_relpath, valid_indices
    
    def run_defense(
        self, 
        split: str = 'test1',
        output_dir: str = 'cirr_defense_unified',
        use_images: bool = True,
        save_results: bool = True
    ) -> Dict:
        """
        运行防御流程
        
        Args:
            split: 数据集分割
            output_dir: 输出目录名
            use_images: 是否使用图像引导修复（需要VLM）
            save_results: 是否保存结果
        
        Returns:
            stats: 统计信息
        """
        print(f"\n{'='*70}")
        print(f"CIRR Defense with Unified VLM")
        print(f"Split: {split}")
        print(f"Model: {self.corrector.model_path}")
        print(f"Use Images: {use_images and self.corrector.is_vlm}")
        print(f"{'='*70}\n")
        
        # 1. 加载数据
        captions, image_paths, triplets, name_to_relpath, valid_indices = self.load_cirr_data(split)
        
        # 2. 批量迭代修复
        if use_images and self.corrector.is_vlm:
            print("\nUsing image-guided correction...")
            correction_results = self.corrector.correct_batch_iterative(
                captions, 
                image_paths,
                verbose=False
            )
        else:
            print("\nUsing text-only correction...")
            correction_results = self.corrector.correct_batch_iterative(
                captions,
                image_paths=None,
                verbose=False
            )
        
        # 3. 更新triplets（确保所有值都是JSON可序列化的）
        for i, result_idx in enumerate(valid_indices):
            result = correction_results[i]
            
            triplets[result_idx]['caption'] = str(result['final_text'])
            triplets[result_idx]['original_caption'] = str(result['original_text'])
            triplets[result_idx]['original_ppl'] = float(result['original_ppl'])
            triplets[result_idx]['final_ppl'] = float(result['final_ppl'])
            triplets[result_idx]['iterations'] = int(result['iterations'])
            triplets[result_idx]['converged'] = bool(result['converged'])
            triplets[result_idx]['ppl_reduction'] = float(result['total_ppl_reduction'])
            triplets[result_idx]['was_corrected'] = bool(not result.get('skipped', False))
        
        # 4. 保存结果
        if save_results:
            output_path = self.base_path / 'src' / output_dir
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存清洗后的数据
            clean_file = output_path / f'cap.rc2.{split}.clean.json'
            with open(clean_file, 'w') as f:
                json.dump(triplets, f, indent=2)
            print(f"\n✓ Saved clean data to: {clean_file}")
            
            # 保存修复日志（只保存实际修复的样本）
            corrected_samples = [
                {
                    'index': valid_indices[i],
                    'pair_id': triplets[valid_indices[i]].get('pairid', 'N/A'),
                    'reference': triplets[valid_indices[i]].get('reference', 'N/A'),
                    **correction_results[i]
                }
                for i in range(len(correction_results))
                if not correction_results[i].get('skipped', False)
            ]
            
            if corrected_samples:
                corrections_file = output_path / f'{split}_corrections.json'
                with open(corrections_file, 'w') as f:
                    json.dump(corrected_samples, f, indent=2)
                print(f"✓ Saved corrections to: {corrections_file}")
            
            # 保存统计信息
            stats = self._compute_statistics(correction_results, split)
            stats_file = output_path / f'{split}_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"✓ Saved statistics to: {stats_file}")
        else:
            stats = self._compute_statistics(correction_results, split)
        
        print(f"\n{'='*70}")
        print("Defense Complete!")
        print(f"{'='*70}\n")
        
        return stats
    
    def _compute_statistics(self, results: List[Dict], split: str) -> Dict:
        """计算统计信息"""
        total = len(results)
        skipped = sum(1 for r in results if r.get('skipped', False))
        processed = total - skipped
        converged = sum(1 for r in results if r['converged'] and not r.get('skipped', False))
        
        original_ppls = [r['original_ppl'] for r in results if not np.isnan(r['original_ppl'])]
        final_ppls = [r['final_ppl'] for r in results if not np.isnan(r['final_ppl'])]
        
        ppl_reductions = [r['total_ppl_reduction'] for r in results if not r.get('skipped', False)]
        iterations = [r['iterations'] for r in results if not r.get('skipped', False)]
        
        stats = {
            'split': split,
            'model': self.corrector.model_path,
            'ppl_threshold': self.corrector.ppl_threshold,
            'max_iterations': self.corrector.max_iterations,
            'total_samples': total,
            'clean_samples': skipped,
            'processed_samples': processed,
            'converged_samples': converged,
            'convergence_rate': float(converged / processed) if processed > 0 else 0.0,
            'avg_iterations': float(np.mean(iterations)) if iterations else 0.0,
            'avg_ppl_reduction': float(np.mean(ppl_reductions)) if ppl_reductions else 0.0,
            'ppl_stats': {
                'original': {
                    'mean': float(np.mean(original_ppls)),
                    'median': float(np.median(original_ppls)),
                    'std': float(np.std(original_ppls)),
                    'min': float(np.min(original_ppls)),
                    'max': float(np.max(original_ppls))
                },
                'final': {
                    'mean': float(np.mean(final_ppls)),
                    'median': float(np.median(final_ppls)),
                    'std': float(np.std(final_ppls)),
                    'min': float(np.min(final_ppls)),
                    'max': float(np.max(final_ppls))
                }
            }
        }
        
        return stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CIRR Defense with Unified VLM')
    parser.add_argument('--split', type=str, default='test4', 
                       choices=['test1', 'val', 'train','test4'],
                       help='Dataset split')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2-VL-2B-Instruct',
                       help='Model path (default: Qwen/Qwen2-VL-2B-Instruct)')
    parser.add_argument('--ppl-threshold', type=float, default=50.0,
                       help='PPL threshold (default: 50.0)')
    parser.add_argument('--max-iterations', type=int, default=3,
                       help='Max correction iterations (default: 3)')
    parser.add_argument('--no-images', action='store_true',
                       help='Disable image-guided correction')
    parser.add_argument('--output-dir', type=str, default='cirr_defense_unified',
                       help='Output directory name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 初始化防御系统
    defense = CIRRDefenseUnified(
        model_path=args.model,
        ppl_threshold=args.ppl_threshold,
        max_iterations=args.max_iterations,
        device=args.device
    )
    
    # 运行防御
    stats = defense.run_defense(
        split=args.split,
        output_dir=args.output_dir,
        use_images=not args.no_images,
        save_results=True
    )
    
    # 打印关键统计
    print("\n" + "="*70)
    print("Key Statistics:")
    print("="*70)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Clean samples: {stats['clean_samples']} ({stats['clean_samples']/stats['total_samples']*100:.1f}%)")
    print(f"Processed samples: {stats['processed_samples']}")
    print(f"Converged: {stats['converged_samples']}/{stats['processed_samples']} ({stats['convergence_rate']*100:.1f}%)")
    print(f"Avg iterations: {stats['avg_iterations']:.2f}")
    print(f"Avg PPL reduction: {stats['avg_ppl_reduction']:.2f}")
    print(f"Original PPL: {stats['ppl_stats']['original']['mean']:.2f} ± {stats['ppl_stats']['original']['std']:.2f}")
    print(f"Final PPL: {stats['ppl_stats']['final']['mean']:.2f} ± {stats['ppl_stats']['final']['std']:.2f}")
    print("="*70)


if __name__ == "__main__":
    main()

