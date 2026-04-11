# CIRR Defense Pipeline - 使用指南

## 📖 概述

这是一个**零训练防御流程**，用于检测和修复CIRR数据集中的扰动文本。

### 核心思想
```
扰动文本 → [PPL检测] → [LLM修复] → 干净文本 → CIR模型
```

## 🛠️ 文件说明

- `ppl_detector.py` - PPL检测器（困惑度计算）
- `text_corrector.py` - 文本修复器（使用Qwen/GPT/Llama）
- `cirr_defense.py` - CIRR数据集防御流程（主程序）
- `analyze_cirr_ppl.py` - PPL分析工具

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install transformers torch numpy tqdm matplotlib scikit-learn
```

### 2. 运行防御流程

#### 基本用法
```bash
cd /home/caoyu/mnt/zhaoai/RobSPRC/src

# 处理CIRR测试集
python cirr_defense.py --split test1
```

#### 自定义参数
```bash
# 使用不同的PPL阈值
python cirr_defense.py --split test1 --threshold 300

# 使用不同的修复模型
python cirr_defense.py --split test1 --corrector qwen

# 指定模型路径（本地模型）
python cirr_defense.py --split test1 --corrector qwen --model_path /path/to/qwen

# 指定保存目录
python cirr_defense.py --split test1 --save_dir ./my_defense_results
```

#### 处理所有split
```bash
# 训练集
python cirr_defense.py --split train --threshold 300

# 验证集
python cirr_defense.py --split val --threshold 300

# 测试集
python cirr_defense.py --split test1 --threshold 300
```

## 📊 输出文件

运行后会在`cirr_defense_results`目录生成：

### 1. 清洗后的数据集
`cap.rc2.test1.clean.json`
- 可以直接用于CIR模型推理
- 包含修复后的caption和原始caption

```json
{
  "caption": "change the background to gray",
  "original_caption": "cahnge the background to gray",
  "reference": "test1-139-3-img1",
  "pairid": 12165
}
```

### 2. 修复记录
`test1_corrections.json`
- 记录所有被修复的样本
- 包含PPL变化信息

```json
{
  "original_index": 46,
  "ppl": 1157.01,
  "ppl_after": 45.23,
  "ppl_reduction": 1111.78,
  "original": "cahnge the background",
  "corrected": "change the background"
}
```

### 3. 统计信息
`test1_stats.json`
- 整体统计数据

```json
{
  "n_total": 4148,
  "n_clean": 3950,
  "n_perturbed": 198,
  "n_fixed": 195
}
```

## 🔄 后续使用

### 在验证脚本中使用清洗后的数据

```python
import json

# 加载清洗后的数据
with open('./cirr_defense_results/cap.rc2.test1.clean.json', 'r') as f:
    clean_data = json.load(f)

# 使用clean caption进行推理
for item in clean_data:
    caption = item['caption']  # 修复后的文本
    # ... CIR推理代码 ...
```

## 📈 实验对比

### 对比实验设置

| 实验 | 描述 | 数据 |
|------|------|------|
| Baseline | 无防御 | 原始数据 |
| Defense | PPL+LLM修复 | 清洗后数据 |

运行对比：
```bash
# Baseline（原始数据）
python validate_blip.py --split test1

# Defense（清洗后数据）
python validate_blip.py --split test1 --clean_captions ./cirr_defense_results/cap.rc2.test1.clean.json
```

## 🔧 高级用法

### 1. 只做PPL分析（不修复）
```bash
python analyze_cirr_ppl.py --split test1 --threshold 300
```

### 2. 测试文本修复器
```python
from text_corrector import TextCorrector

corrector = TextCorrector(method='qwen')
corrected, is_changed = corrector.correct_single(
    "cahnge the background to gray"
)
print(corrected)  # "change the background to gray"
```

### 3. 自定义PPL阈值选择

根据分析结果选择合适的阈值：
- **保守策略**（threshold=500）：只修复极端扰动
- **平衡策略**（threshold=300）：修复明显异常
- **激进策略**（threshold=100）：修复所有疑似问题

## 📝 论文写作建议

### 方法部分
```
我们提出一种零训练防御方法：
1. 使用GPT-2计算文本困惑度(PPL)
2. 识别PPL>threshold的异常样本
3. 使用大语言模型(Qwen-7B)修复文本
4. 验证修复效果
```

### 实验结果
```
在CIRR测试集上：
- 检测到198个扰动样本(4.8%)
- 成功修复195个(98.5%)
- 平均PPL从1157降至52
- Recall@1提升X%
```

## ⚠️ 注意事项

1. **首次运行需要下载模型**：Qwen模型约14GB
2. **GPU内存**：建议至少16GB显存
3. **处理时间**：test1约需30-60分钟（取决于GPU）
4. **阈值选择**：建议先运行`analyze_cirr_ppl.py`查看分布

## 🐛 常见问题

### Q1: 内存不足
```bash
# 使用更小的batch size
# 在ppl_detector.py中修改batch_size=8
```

### Q2: 修复效果不好
```bash
# 调整阈值
python cirr_defense.py --threshold 500

# 或使用更强的模型
python cirr_defense.py --corrector gpt
```

### Q3: 运行太慢
```bash
# 只处理扰动样本的子集
# 或使用更小的模型
```

## 📧 联系方式

如有问题，请检查：
1. GPU是否可用
2. 依赖是否完整
3. 数据路径是否正确




