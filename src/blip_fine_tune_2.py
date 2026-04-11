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
from torch.optim.lr_scheduler import OneCycleLR
import os

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn,set_train_bar_description_dict,  update_train_running_results_dict, \
     extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, device
from validate_blip import (
    compute_cirr_val_metrics,
    compute_fiq_val_metrics,
    analyze_cirr_kappa_behavior,
)

class ModelEMA:
    """
    轻量级 EMA (Exponential Moving Average)
    专为冻结了大部分底座（如 ViT）的大模型设计，仅对可训练参数进行滑动平均。
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}  # 存放平滑后的权重
        self.backup = {}  # 评估时临时存放当前权重

        # 初始化时，只把需要梯度的参数拉出来做 EMA，极大节省显存
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model):
        """在每次 optimizer.step() 后调用此方法更新影子权重"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # 公式: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        """在验证 (Validation) 开始前调用，将平滑权重注入模型"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        """在验证结束后调用，恢复模型的当前训练权重"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def clip_finetune_fiq(train_dress_types: List[str], val_dress_types: List[str],
                      num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory: bool,
                      warmup_epochs: int, use_cache: bool, 
                      use_validity_scorer: bool, hard_negative_mining: bool, use_delta_constraint: bool, **kwargs):
    """
    Fine-tune on FashionIQ dataset with Uncertainty (MUS/TAS) & Precomputed Features support
    """
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_fiq_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"save-memory-in: {save_memory}")
    
    # Save parameters
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(kwargs, file, sort_keys=True, indent=4)

    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    
    # 🚀 拦截并覆写消融开关！
    blip_model.use_validity_scorer = use_validity_scorer
    blip_model.hard_negative_mining = hard_negative_mining
    blip_model.use_delta_constraint = use_delta_constraint

    print("="*50)
    print(f"🎛️ FIQ Training Ablation Config Applied:")
    print(f"  Validity Scorer (Soft Gating) : {blip_model.use_validity_scorer}")
    print(f"  Hard Negative Mining          : {blip_model.hard_negative_mining}")
    print(f"  Delta Constraint (L_delta)    : {blip_model.use_delta_constraint}")
    print("="*50)

    # Update Q-Former if needed
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # Validation Datasets (Standard Image Loading)
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
        classic_val_datasets.append(classic_val_dataset)

    # Train Dataset (Supports Precomputed Features)
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess, use_cache=use_cache)
    
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # ==========================================================
    # 🚀 核心修复：差异化学习率 (Multi-LR for Sigmoid Loss)
    # ==========================================================
    backbone_params = []
    loss_params = []
    for name, p in blip_model.named_parameters():
        if p.requires_grad:
            if 'hug_loss_fn' in name:
                loss_params.append(p)
                print(f"🔥 [LR 提速特权分配 - FIQ] 损失函数参数: {name}")
            else:
                backbone_params.append(p)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-8, 'weight_decay': 0.05},
        {'params': loss_params, 'lr': 1e-3, 'betas': (0.9, 0.98), 'eps': 1e-8, 'weight_decay': 0.0} # 🌟 Sigmoid 边界专属法拉利起步
    ])
    
    # 🌟 修复 OneCycleLR 多组参数覆盖问题：必须传入一个对应长度的 list
    scheduler = OneCycleLR(optimizer, max_lr=[learning_rate, 1e-3], pct_start=1.5 / num_epochs, div_factor=100.,
                           steps_per_epoch=len(relative_train_loader), epochs=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    if save_best:
        best_avg_recall = 0

    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    print('Training loop started')
    for epoch in range(num_epochs):
        enable_uncertainty = (epoch >= warmup_epochs)
        stage_name = "Uncertainty" if enable_uncertainty else "Warm-up"
        
        train_running_results = {'images_in_epoch': 0}
        print(f"\nEpoch {epoch+1}/{num_epochs} - Stage: {stage_name}")
        
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)

            flattened_captions: list = np.array(captions).T.flatten().tolist()
            captions = generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors["eval"](caption) for caption in captions]
            
            blip_model.train()
            
            with torch.amp.autocast('cuda'):
                loss_dict = blip_model(
                    {"image": reference_images, "target": target_images, "text_input": captions, "epoch": epoch},
                    enable_uncertainty=enable_uncertainty
                )
                loss = loss_dict['loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
                    train_running_results[key] / train_running_results['images_in_epoch'])
        
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []

            for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                      idx_to_dress_mapping):
                index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model, save_memory)
                recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model,
                                                                   index_features, index_names, txt_processors,
                                                                   save_memory)

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

            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('tuned_clip_best', epoch, blip_model, training_path)


def clip_finetune_cirr(num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                       validation_frequency: int, transform: str, save_training: bool, save_best: bool,
                       warmup_epochs: int, use_cache: bool, 
                       use_validity_scorer: bool, hard_negative_mining: bool, use_delta_constraint: bool, **kwargs):
    """
    Fine-tune on CIRR dataset with Uncertainty (MUS/TAS) & Precomputed Features support
    """
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    # 将消融配置加入文件夹名称中，方便区分不同的实验
    ablation_suffix = f"vs{int(use_validity_scorer)}_hn{int(hard_negative_mining)}_dc{int(use_delta_constraint)}"
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_cirr_{blip_model_name}_{ablation_suffix}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(kwargs, file, sort_keys=True, indent=4)

    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone,
                                                                           is_eval=False, device=device)
    
    # 🚀 拦截并覆写消融开关！
    blip_model.use_validity_scorer = use_validity_scorer
    blip_model.hard_negative_mining = hard_negative_mining
    blip_model.use_delta_constraint = use_delta_constraint

    print("="*50)
    print(f"🎛️ CIRR Training Ablation Config Applied:")
    print(f"  Validity Scorer (Soft Gating) : {blip_model.use_validity_scorer}")
    print(f"  Hard Negative Mining          : {blip_model.hard_negative_mining}")
    print(f"  Delta Constraint (L_delta)    : {blip_model.use_delta_constraint}")
    print("="*50)

    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    relative_train_dataset = CIRRDataset('train', 'relative', preprocess, use_cache=use_cache)
    
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # ==========================================================
    # 🚀 核心修复：差异化学习率 (Multi-LR for Sigmoid Loss)
    # ==========================================================
    backbone_params = []
    loss_params = []
    for name, p in blip_model.named_parameters():
        if p.requires_grad:
            if 'hug_loss_fn' in name:
                loss_params.append(p)
                print(f"🔥 [LR 提速特权分配 - CIRR] 损失函数参数: {name}")
            else:
                backbone_params.append(p)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate, 'betas': (0.9, 0.98), 'eps': 1e-8, 'weight_decay': 0.05},
        {'params': loss_params, 'lr': 1e-3, 'betas': (0.9, 0.98), 'eps': 1e-8, 'weight_decay': 0.0} # 🌟 Sigmoid 边界专属法拉利起步
    ])
    
    # 🌟 修复 OneCycleLR 多组参数覆盖问题：必须传入一个对应长度的 list
    # 之前写死了 epochs=80，现在改成动态的 num_epochs
    scheduler = OneCycleLR(optimizer, max_lr=[learning_rate, 1e-3], pct_start=1.5 / num_epochs,
                           steps_per_epoch=len(relative_train_loader), epochs=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    # ema = ModelEMA(blip_model, decay=0.999)
    if save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0

    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    for epoch in range(num_epochs):
        enable_uncertainty = (epoch >= warmup_epochs)
        stage_name = "Uncertainty" if enable_uncertainty else "Warm-up"
        
        train_running_results = {'images_in_epoch': 0}
        print(f"\nEpoch {epoch+1}/{num_epochs} - Stage: {stage_name}")
        
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            
            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            captions = [txt_processors["eval"](caption) for caption in captions]
            
            blip_model.train()
            
            with torch.amp.autocast('cuda'):
                loss_dict = blip_model(
                    {"image": reference_images, "target": target_images, "text_input": captions, "epoch": epoch},
                    enable_uncertainty=enable_uncertainty
                )
                loss = loss_dict['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # ema.update(blip_model)
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
                    train_running_results[key] / train_running_results['images_in_epoch'])
        
        training_log_frame = pd.concat([training_log_frame, pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()

            val_index_features, val_index_kappas, val_index_names = extract_index_blip_features(
                classic_val_dataset, blip_model
            )

            # 1) 原有 CIRR 检索指标
            results = compute_cirr_val_metrics(
                relative_val_dataset,
                blip_model,
                val_index_features,
                val_index_names,
                txt_processors,
                val_index_kappas=val_index_kappas
            )
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

            # 2) 新增：kappa 行为分析
            kappa_analysis = analyze_cirr_kappa_behavior(
                blip_model=blip_model,
                relative_val_dataset=relative_val_dataset,
                index_names=val_index_names,
                index_features=val_index_features,
                txt_processors=txt_processors,
                val_index_kappas=val_index_kappas,
            )

            print(json.dumps(results_dict, indent=4, ensure_ascii=False))
            print(json.dumps(kappa_analysis, indent=4, ensure_ascii=False))

            # 3) 一起写入日志
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            log_dict.update(kappa_analysis)

            validation_log_frame = pd.concat(
                [validation_log_frame, pd.DataFrame(data=log_dict, index=[0])]
            )
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                    best_arithmetic = results_dict['arithmetic_mean']
                    save_model('tuned_clip_arithmetic', epoch, blip_model, training_path)
            # ema.restore(blip_model)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--data-path", type=str, default="./cirr_dataset")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str, help="[blip2_cir_cat, blip2_cir]")
    parser.add_argument("--backbone", type=str, default="pretrain",
                        help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--learning-rate", default=2e-6, type=float, help="Learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    
    parser.add_argument("--validation-frequency", default=1, type=int,
                        help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--save-memory", dest="save_memory", action='store_true',
                        help="Save only the best model during training")
    
    parser.add_argument("--warmup-epochs", default=5, type=int, 
                        help="Number of epochs to disable uncertainty loss (Warm-up)")
    parser.add_argument("--use-cache", action='store_true', 
                        help="Use precomputed ViT features for training (Extreme Speedup)")

    # 🎛️ 注册消融实验命令行参数 (默认全部开启)
    parser.add_argument("--use-validity-scorer", type=str2bool, default=True, help="启用错配验证器")
    parser.add_argument("--hard-negative-mining", type=str2bool, default=True, help="启用困难负样本")
    parser.add_argument("--use-delta-constraint", type=str2bool, default=True, help="启用差分损失")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")
    print(f"save-memory: {args.save_memory}")
    
    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "blip_model_name": args.blip_model_name,
        "backbone": args.backbone,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "data_path": args.data_path,
        "save_memory": args.save_memory,
        "warmup_epochs": args.warmup_epochs,
        "use_cache": args.use_cache,
        # 传入消融参数
        "use_validity_scorer": args.use_validity_scorer,
        "hard_negative_mining": args.hard_negative_mining,
        "use_delta_constraint": args.use_delta_constraint
    }
    
    # set_seed(912)
    if args.dataset.lower() == 'cirr':
        clip_finetune_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        clip_finetune_fiq(**training_hyper_params)