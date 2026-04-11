import multiprocessing
import random
from pathlib import Path
from typing import Union, Tuple, List, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import CIRRDataset, FashionIQDataset

# 全局设备定义
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_index_blip_features(dataset: Union[CIRRDataset, FashionIQDataset], blip_model, save_memory=False) -> \
        Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    提取验证集/测试集的目标图像特征及置信度 Kappa（构建 Gallery 索引）
    适配 vMF 连续空间 CIR 模型结构
    
    :param dataset: 'classic' 模式的 Dataset (返回 raw images)
    :param blip_model: 你的 vMF 模型
    :return: (特征张量 [N, Dim], 置信度张量 [N, 1], 图片名列表)
    """
    feature_dim = 256  # 默认投影维度，根据实际情况可能调整
    
    # 使用 classic 模式的 DataLoader，batch_size 可以适当大一点
    classic_val_loader = DataLoader(dataset=dataset, batch_size=64, num_workers=8,
                                    pin_memory=True, collate_fn=collate_fn, shuffle=False)
    
    index_features = []
    index_kappas = []  # 🔥 新增：用于存储每张目标图像的置信度
    index_names = []
    
    # 打印提示信息
    if isinstance(dataset, CIRRDataset):
        print(f"🚀 Extracting CIRR {dataset.split} index features and kappas...")
    elif isinstance(dataset, FashionIQDataset):
        print(f"🚀 Extracting FashionIQ {dataset.dress_types} - {dataset.split} index features and kappas...")
        
    for batch in tqdm(classic_val_loader, ncols=100):
        if len(batch) == 3:
            names, _, images = batch
            images, _, names = batch
        elif len(batch) == 2:
             names, images = batch 
        
        # 按照惯例解析 batch
        names = batch[0]
        images = batch[-1]

        images = images.to(device, non_blocking=True)
        
        with torch.no_grad():
            output = blip_model.extract_target_features(images)
            
            # 🔥 核心修改：同时提取特征和 Kappa
            if isinstance(output, tuple) and len(output) == 2:
                batch_features, batch_kappas = output
            elif isinstance(output, dict):
                # 如果返回 dict，提取特征和对应的 kappa
                batch_features = output.get('mu_t', output.get('image_embeds_proj'))
                batch_kappas = output.get('kappa_t')
                if batch_features is None:
                    raise ValueError(f"Unknown output keys: {output.keys()}")
            else:
                # 兼容旧版本：如果模型没有返回 kappa，生成一个全 1 的高置信度占位符
                batch_features = output
                batch_kappas = torch.ones((batch_features.size(0), 1), device=batch_features.device)

            # 确保特征是归一化的 (vMF 要求)
            batch_features = F.normalize(batch_features, dim=-1)

            if save_memory:
                batch_features = batch_features.cpu()
                batch_kappas = batch_kappas.cpu()  # 🔥 Kappa 也转到 CPU 省显存
                
            index_features.append(batch_features)
            index_kappas.append(batch_kappas)      # 🔥 收集 Kappa
            index_names.extend(names)
    
    # 拼接所有特征和 Kappas
    index_features = torch.vstack(index_features)
    index_kappas = torch.vstack(index_kappas)      # 🔥 拼接 Kappa
    
    print(f"✅ Index features extracted: {index_features.shape} | Kappas: {index_kappas.shape}")
    
    # 🔥 返回 3 个元素：特征、置信度、图片名
    return index_features, index_kappas, index_names


def element_wise_sum(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """
    Normalized element-wise sum of image features and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def update_train_running_results_dict(train_running_results: dict, loss_dict: dict, images_in_batch: int):
    """
    Update `train_running_results` dict during training (Supports multiple losses)
    """
    for key in loss_dict.keys():
        # 如果是 Tensor (如 loss 值)，转为 float
        value = loss_dict[key]
        if isinstance(value, torch.Tensor):
            value = value.item()
            
        if key not in train_running_results:
            train_running_results[key] = 0
        
        # 累加：当前 batch 的平均 loss * batch 大小
        train_running_results[key] += value * images_in_batch

    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description_dict(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar with all losses in the dict
    """ 
    images_in_epoch = train_running_results['images_in_epoch']
    bar_content = ''
    
    # 优先显示总 Loss
    if 'loss' in train_running_results:
        avg_loss = train_running_results['loss'] / images_in_epoch
        bar_content += f'Loss: {avg_loss:.4f} | '
        
    # 显示其他监控指标 (如 mus, tas)
    for key in train_running_results:
        if key != 'images_in_epoch' and key != 'loss':
            val = train_running_results[key] / images_in_epoch
            # 对于 mus/tas 这种指标，显示3位小数
            bar_content += f'{key}: {val:.3f} '
            
    train_bar.set_description(desc=f"[{epoch+1}/{num_epochs}] {bar_content}")   


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    """
    Save the weights of the model (Fix: Standardized key and DDP unwrapping)
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    
    # 🚀 核心修复 1: 解除多卡 DDP 封装，获取最纯净的底层 model
    if hasattr(model_to_save, 'module'):
        model_to_save = model_to_save.module
        
    # 🚀 核心修复 2: 强制将权重的 Key 命名为通用的 'model'
    torch.save({
        'epoch': cur_epoch,
        'model': model_to_save.state_dict(), 
    }, str(models_path / f'{name}.pt'))
    
    print(f"✅ 已完整保存 Epoch {cur_epoch} 的模型权重至: {models_path / f'{name}.pt'}")