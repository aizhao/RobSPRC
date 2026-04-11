import json
import argparse
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import random
import string

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, extract_index_blip_features, device
from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics

# ==========================================
# 🚀 1. 动态文本干扰器 (Text Corruptor)
# ==========================================
class TextCorruptor:
    def __init__(self):
        self.misspell_map = {'their': 'there', 'there': 'their', 'your': 'youre', 'definitely': 'definately', 'separate': 'seperate', 'a lot': 'alot', 'receive': 'recieve', 'until': 'untill'}
        self.homophone_map = {'in': 'inn', 'to': 'too', 'write': 'right', 'see': 'sea', 'one': 'won', 'two': 'too', 'for': 'four', 'sun': 'son', 'hair': 'hare', 'night': 'knight', 'flower': 'flour'}
        self.qwerty_map = {'a': 's', 's': 'd', 'd': 'f', 'o': 'p', 'i': 'o', 'e': 'r', 'r': 't', 't': 'y', 'y': 'u', 'u': 'i', 'w': 'e', 'q': 'w', 'h': 'j', 'j': 'k', 'k': 'l'}

    def apply(self, text: str, c_type: str) -> str:
        if not text or c_type in ['none', 'clean', '']: return text
        words = text.split()
        if len(words) == 0: return text
        c_type = c_type.lower()
        
        if c_type == 'swap':
            idx = random.randint(0, len(words) - 1)
            word = list(words[idx])
            if len(word) > 1:
                c_idx = random.randint(0, len(word) - 2)
                word[c_idx], word[c_idx+1] = word[c_idx+1], word[c_idx]
                words[idx] = "".join(word)
        elif c_type == 'qwerty':
            idx = random.randint(0, len(words) - 1)
            word = list(words[idx])
            if len(word) > 0:
                c_idx = random.randint(0, len(word) - 1)
                char = word[c_idx].lower()
                if char in self.qwerty_map:
                    new_char = self.qwerty_map[char].upper() if word[c_idx].isupper() else self.qwerty_map[char]
                    word[c_idx] = new_char
                words[idx] = "".join(word)
        elif c_type == 'removechar':
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if len(word) > 1:
                c_idx = random.randint(0, len(word) - 1)
                words[idx] = word[:c_idx] + word[c_idx+1:]
        elif c_type == 'removespace':
            if len(words) > 1:
                idx = random.randint(0, len(words) - 2)
                words[idx] = words[idx] + words[idx+1]
                del words[idx+1]
        elif c_type == 'misspelling':
            for i, w in enumerate(words):
                clean_w = w.lower().strip(string.punctuation)
                if clean_w in self.misspell_map:
                    words[i] = w.lower().replace(clean_w, self.misspell_map[clean_w])
                    break
        elif c_type == 'repetition':
            idx = random.randint(0, len(words) - 1)
            words.insert(idx, words[idx])
        elif c_type == 'homophone':
            for i, w in enumerate(words):
                clean_w = w.lower().strip(string.punctuation)
                if clean_w in self.homophone_map:
                    words[i] = w.lower().replace(clean_w, self.homophone_map[clean_w])
                    break
        return " ".join(words)

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# ==========================================
# 🚀 2. 验证集评估逻辑
# ==========================================
def clip_finetune_fiq(val_dress_types: List[str], blip_model_name, backbone, model_path, args):
    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    
    # 设置消融开关
    blip_model.use_validity_scorer = args.use_validity_scorer
    blip_model.hard_negative_mining = args.hard_negative_mining
    blip_model.use_delta_constraint = args.use_delta_constraint

    checkpoint = torch.load(model_path, map_location=device)
    msg = blip_model.load_state_dict(checkpoint.get(blip_model.__class__.__name__, checkpoint), strict=False)
    print("Missing keys {}".format(msg.missing_keys))

    # 🌟 处理器劫持：动态注入干扰
    if args.text_corruption != 'none':
        print(f"⚠️ [WARNING] 正在对 FashionIQ 验证集应用文本干扰: {args.text_corruption}")
        corruptor = TextCorruptor()
        original_eval_processor = txt_processors["eval"]
        def wrapped_eval_processor(text):
            return original_eval_processor(corruptor.apply(text, args.text_corruption))
        txt_processors["eval"] = wrapped_eval_processor

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    
    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_datasets.append(FashionIQDataset('val', [dress_type], 'relative', preprocess))
        classic_val_datasets.append(FashionIQDataset('val', [dress_type], 'classic', preprocess))

    blip_model.eval()
    recalls_at10 = []
    recalls_at50 = []

    for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets, idx_to_dress_mapping):
        index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model)
        recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model, index_features, index_names, txt_processors)
        recalls_at10.append(recall_at10)
        recalls_at50.append(recall_at50)
        torch.cuda.empty_cache()

    results_dict = {}
    for i in range(len(recalls_at10)):
        results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
        results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
    results_dict.update({
        f'average_recall_at10': mean(recalls_at10),
        f'average_recall_at50': mean(recalls_at50),
        f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
    })

    print(json.dumps(results_dict, indent=4))

def blip_validate_cirr(blip_model_name, backbone, blip_model_path, args):
    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    
    # 设置消融开关
    blip_model.use_validity_scorer = args.use_validity_scorer
    blip_model.hard_negative_mining = args.hard_negative_mining
    blip_model.use_delta_constraint = args.use_delta_constraint

    checkpoint = torch.load(blip_model_path, map_location=device)
    
    # 核心修复：精准捕获你保存时使用的 'model' 键
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # 兼容旧版本可能包裹在类名下的情况
        state_dict = checkpoint.get(blip_model.__class__.__name__, checkpoint)

    # 加载权重
    msg = blip_model.load_state_dict(state_dict, strict=False)
    
    # 打印缺失数量，直观判断是否加载成功
    print(f"✅ 加载完成，Missing keys 数量: {len(msg.missing_keys)}")
    if len(msg.missing_keys) > 0:
        print(f"⚠️ 前 10 个缺失的 keys: {msg.missing_keys[:10]}")

    # 🌟 处理器劫持：动态注入干扰
    if args.text_corruption != 'none':
        print(f"\n{'='*50}")
        print(f"⚠️ 正在对 CIRR 验证集动态应用文本干扰: [{args.text_corruption.upper()}]")
        print(f"{'='*50}\n")
        corruptor = TextCorruptor()
        original_eval_processor = txt_processors["eval"]
        
        # 劫持 eval 处理器
        def wrapped_eval_processor(text):
            corrupted_text = corruptor.apply(text, args.text_corruption)
            return original_eval_processor(corrupted_text)
            
        txt_processors["eval"] = wrapped_eval_processor

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)

    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    val_index_features, val_index_kappas, val_index_names = extract_index_blip_features(
                classic_val_dataset, blip_model
            )
    results = compute_cirr_val_metrics(relative_val_dataset, blip_model, val_index_features, val_index_names, txt_processors, val_index_kappas)
    
    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results
    results_dict = {
        'group_recall_at1': group_recall_at1,
        'group_recall_at2': group_recall_at2,
        'group_recall_at3': group_recall_at3,
        'recall_at1': recall_at1,
        'recall_at5': recall_at5,
        'recall_at10': recall_at10,
        'recall_at50': recall_at50,
        'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
        'arithmetic_mean': mean(results),
        'harmonic_mean': harmonic_mean(results),
        'geometric_mean': geometric_mean(results)
    }
    print(json.dumps(results_dict, indent=4))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str) # 修改了默认模型名
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--model-path", type=str, required=True)
    
    # 🎛️ 文本干扰参数
    parser.add_argument("--text-corruption", type=str, default="none", 
                        choices=["none", "swap", "qwerty", "removechar", "removespace", "misspelling", "repetition", "homophone"],
                        help="选择要动态注入的文本干扰类型")
    
    # 🎛️ 消融实验参数 (保持和你训练时一致)
    parser.add_argument("--use-validity-scorer", type=str2bool, default=True)
    parser.add_argument("--hard-negative-mining", type=str2bool, default=True)
    parser.add_argument("--use-delta-constraint", type=str2bool, default=True)

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    if args.dataset.lower() == 'cirr':
        blip_validate_cirr(args.blip_model_name, args.backbone, args.model_path, args)
    elif args.dataset.lower() == 'fashioniq':
        clip_finetune_fiq(['dress', 'toptee', 'shirt'], args.blip_model_name, args.backbone, args.model_path, args)