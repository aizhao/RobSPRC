"""
CIRR测试集文本扰动脚本
为测试集添加不同比例的文本扰动（20%, 40%, 60%, 80%, 100%）
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

# 添加SPRC项目路径以导入text_perturbation
sprc_path = Path('/home/caoyu/mnt/zhaoai/SPRC/src')
if str(sprc_path) not in sys.path:
    sys.path.insert(0, str(sprc_path))

from robustness.text_perturbation import TextPerturber


class CIRRPerturbationGenerator:
    """CIRR数据集文本扰动生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化扰动生成器
        
        Args:
            seed: 随机种子，用于可复现性
        """
        self.seed = seed
        self.perturber = TextPerturber(seed=seed)
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        
        # 获取所有可用的扰动类型（排除慢速扰动）
        all_types = []
        for level, types in TextPerturber.get_all_types().items():
            all_types.extend(types)
        
        # 排除可能较慢或需要额外资源的扰动
        slow_augs = ['contextual_word', 'reserved_word']
        self.available_perturbations = [t for t in all_types if t not in slow_augs]
        
        print(f"可用扰动类型 ({len(self.available_perturbations)}): {self.available_perturbations}")
    
    def perturb_caption(self, caption: str) -> tuple:
        """
        对单个caption应用随机扰动
        
        Args:
            caption: 原始caption
            
        Returns:
            (perturbed_caption, perturbation_type, severity)
        """
        # 随机选择扰动类型和严重程度
        perturbation_type = self.rng.choice(self.available_perturbations)
        severity = self.rng.randint(1, 6)  # 1-5
        
        try:
            perturbed = self.perturber.apply(caption, perturbation_type, severity)
            return perturbed, perturbation_type, severity
        except Exception as e:
            print(f"警告: 扰动失败 ({perturbation_type}, severity={severity}): {e}")
            return caption, None, None
    
    def generate_perturbed_dataset(
        self,
        input_file: Path,
        output_file: Path,
        perturbation_ratio: float
    ) -> Dict:
        """
        生成指定比例的扰动数据集
        
        Args:
            input_file: 输入CIRR caption文件路径
            output_file: 输出文件路径
            perturbation_ratio: 扰动比例 (0.0-1.0)
            
        Returns:
            统计信息字典
        """
        # 加载原始数据
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        total_samples = len(data)
        num_perturbed = int(total_samples * perturbation_ratio)
        
        print(f"\n{'='*60}")
        print(f"生成 {perturbation_ratio*100:.0f}% 扰动数据集")
        print(f"总样本数: {total_samples}")
        print(f"扰动样本数: {num_perturbed}")
        print(f"{'='*60}")
        
        # 随机选择要扰动的样本索引
        perturbed_indices = set(self.rng.choice(
            total_samples, 
            size=num_perturbed, 
            replace=False
        ))
        
        # 统计信息
        stats = {
            'total_samples': total_samples,
            'num_perturbed': num_perturbed,
            'perturbation_ratio': perturbation_ratio,
            'perturbation_types': {},
            'severity_distribution': {i: 0 for i in range(1, 6)},
            'perturbed_samples': []
        }
        
        # 应用扰动
        for i, item in enumerate(data):
            original_caption = item.get('caption', '')
            
            if i in perturbed_indices and original_caption.strip():
                # 应用扰动
                perturbed_caption, pert_type, severity = self.perturb_caption(original_caption)
                
                # 更新item
                item['caption'] = perturbed_caption
                item['original_caption'] = original_caption
                item['perturbation_type'] = pert_type
                item['perturbation_severity'] = severity
                item['is_perturbed'] = True
                
                # 更新统计
                if pert_type:
                    stats['perturbation_types'][pert_type] = stats['perturbation_types'].get(pert_type, 0) + 1
                    stats['severity_distribution'][severity] += 1
                    
                    # 记录样本（只保存前100个示例）
                    if len(stats['perturbed_samples']) < 100:
                        stats['perturbed_samples'].append({
                            'index': i,
                            'pair_id': item.get('pairid', 'N/A'),
                            'original': original_caption,
                            'perturbed': perturbed_caption,
                            'type': pert_type,
                            'severity': severity
                        })
                
                if (i + 1) % 100 == 0:
                    print(f"已处理: {i+1}/{num_perturbed} 个扰动样本")
            else:
                # 不扰动的样本
                item['is_perturbed'] = False
        
        # 保存扰动后的数据
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ 已保存扰动数据到: {output_file}")
        
        # 保存统计信息
        stats_file = output_file.parent / f"{output_file.stem}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ 已保存统计信息到: {stats_file}")
        
        # 打印统计摘要
        print(f"\n扰动类型分布:")
        for pert_type, count in sorted(stats['perturbation_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {pert_type}: {count} ({count/num_perturbed*100:.1f}%)")
        
        print(f"\n严重程度分布:")
        for severity, count in stats['severity_distribution'].items():
            print(f"  Severity {severity}: {count} ({count/num_perturbed*100:.1f}%)")
        
        return stats


def main():
    """主函数"""
    # 配置路径
    base_path = Path('/home/caoyu/mnt/zhaoai/RobSPRC')
    input_file = base_path / 'cirr_dataset' / 'cirr' / 'captions' / 'cap.rc2.test1.json'
    output_dir = base_path / 'cirr_dataset_perturbed'
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    
    # 创建扰动生成器
    generator = CIRRPerturbationGenerator(seed=42)
    
    # 定义扰动比例
    perturbation_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # 为每个比例生成扰动数据集
    all_stats = {}
    for ratio in perturbation_ratios:
        ratio_name = f"{int(ratio*100)}percent"
        output_file = output_dir / ratio_name / 'cap.rc2.test1.json'
        
        stats = generator.generate_perturbed_dataset(
            input_file=input_file,
            output_file=output_file,
            perturbation_ratio=ratio
        )
        
        all_stats[ratio_name] = stats
    
    # 保存总体统计
    summary_file = output_dir / 'perturbation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ 所有扰动数据集生成完成!")
    print(f"✓ 总体统计保存到: {summary_file}")
    print(f"{'='*60}")
    
    # 打印目录结构
    print(f"\n生成的目录结构:")
    print(f"{output_dir}/")
    for ratio in perturbation_ratios:
        ratio_name = f"{int(ratio*100)}percent"
        print(f"  ├── {ratio_name}/")
        print(f"  │   ├── cap.rc2.test1.json")
        print(f"  │   └── cap.rc2.test1_stats.json")
    print(f"  └── perturbation_summary.json")


if __name__ == '__main__':
    main()