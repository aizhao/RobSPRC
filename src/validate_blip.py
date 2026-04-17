import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import FashionIQDataset, CIRRDataset
from utils import collate_fn, device

def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, blip_model, index_features: torch.Tensor,
                            index_names: List[str], txt_processors, save_memory=False) -> Tuple[float, float]:
    """
    计算 FashionIQ 验证指标
    """
    pred_sim, target_names, reference_names, captions_all = generate_fiq_val_predictions(
        blip_model, relative_val_dataset, index_names, index_features, txt_processors, save_memory
    )

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1)
    )

    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


def generate_fiq_val_predictions(blip_model, relative_val_dataset: FashionIQDataset,
                                 index_names: List[str], index_features: torch.Tensor, 
                                 txt_processors, save_memory=False, val_index_kappas=None) -> Tuple[torch.Tensor, List[str], List[str], List[str]]:
    """
    生成 FashionIQ 预测结果 (支持 vMF 概率推断)
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions (vMF Enhanced)")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=4, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    gallery_features = index_features.to(device) if not save_memory else index_features
    
    if val_index_kappas is not None:
        gallery_kappas = val_index_kappas.to(device) if not save_memory else val_index_kappas
    else:
        raise ValueError("vMF inference requires val_index_kappas!")

    all_sims, target_names, reference_names_all, captions_all = [], [], [], []

    for batch_ref_imgs, batch_ref_names, batch_tgt_names, batch_captions in tqdm(relative_val_loader, ncols=100):
        # 处理文本
        flattened_captions: list = np.array(batch_captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" 
            for i in range(0, len(flattened_captions), 2)
        ]
        input_captions = [txt_processors["eval"](c) for c in input_captions]
        
        batch_ref_imgs = batch_ref_imgs.to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # 🔥 1. 提取 Query 分布 (方向 + 置信度)
                query_feat, query_kappas = blip_model.extract_query_features(batch_ref_imgs, input_captions)
                
                # Squeeze 多余的维度
                query_feat = query_feat.squeeze(1) # -> [B, 256]
                if query_kappas.dim() == 3:
                    query_kappas = query_kappas.squeeze(1) # -> [B, 1]
                
                # 🔥 2. 进行概率相似度推断
                sims = blip_model.inference(
                    query_feat=query_feat,
                    k_query=query_kappas,
                    target_feats=gallery_features,
                    k_targets=gallery_kappas
                )
                
                all_sims.append(sims.cpu())
            
        target_names.extend(batch_tgt_names)
        reference_names_all.extend(batch_ref_names)
        captions_all.extend(input_captions)
        
    return torch.vstack(all_sims), target_names, reference_names_all, captions_all


def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, blip_model, index_features: torch.Tensor,
                             index_names: List[str], txt_processors, val_index_kappas=None) -> Tuple[float, float, float, float, float, float, float]:
    """
    计算 CIRR 验证指标 (包含 Recall@K 和 Subset Recall@K)
    """
    pred_sim, reference_names, target_names, group_members, captions_all = \
        generate_cirr_val_predictions(blip_model, relative_val_dataset, index_names, index_features, txt_processors, val_index_kappas=val_index_kappas)

    print("Compute CIRR validation metrics")
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # CIRR 特有规则：从结果中剔除参考图 (Self-retrieval filtering)
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1)
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], -1)

    # 计算标签
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1)
    )

    # 计算 Subset (Group) 掩码
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    # 基础 Recall 指标
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    
    # Subset Recall 指标
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(blip_model, relative_val_dataset: CIRRDataset, 
                                  index_names: List[str], index_features: torch.Tensor, txt_processors,
                                  val_index_kappas=None) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]], List[str]]:
    """
    生成 CIRR 预测结果 (支持 vMF 概率推断)
    """
    print("Compute CIRR validation predictions (vMF Enhanced)")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=4,
                                     pin_memory=True, collate_fn=collate_fn)

    gallery_features = index_features.to(device)
    # 将图库的 kappa 也放进显存
    if val_index_kappas is not None:
        gallery_kappas = val_index_kappas.to(device)
    else:
        raise ValueError("vMF inference requires val_index_kappas!")

    all_sims, target_names, group_members, reference_names, captions_all = [], [], [], [], []

    for batch in tqdm(relative_val_loader, ncols=100):
        batch_ref_imgs, batch_ref_names, batch_tgt_names, batch_captions, batch_group_m = batch
        
        batch_group_m = np.array(batch_group_m).T.tolist()
        batch_captions = [txt_processors["eval"](c) for c in batch_captions]
        batch_ref_imgs = batch_ref_imgs.to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # 🔥 1. 提取 Query 分布 (方向 + 置信度)
                query_feat, query_kappas = blip_model.extract_query_features(batch_ref_imgs, batch_captions)
                
                # Squeeze 多余的维度
                query_feat = query_feat.squeeze(1) # -> [B, 256]
                if query_kappas.dim() == 3:
                    query_kappas = query_kappas.squeeze(1) # -> [B, 1]
                
                # 🔥 2. 使用底层模型封装的 inference 函数进行严密的概率相似度计算
                # 传入 query 和 target 的方向与集中度参数
                sims = blip_model.inference(
                    query_feat=query_feat,
                    k_query=query_kappas,
                    target_feats=gallery_features,
                    k_targets=gallery_kappas
                )
                
                all_sims.append(sims.cpu())

        target_names.extend(batch_tgt_names)
        group_members.extend(batch_group_m)
        reference_names.extend(batch_ref_names)
        captions_all += batch_captions
    
    return torch.vstack(all_sims), reference_names, target_names, group_members, captions_all

def analyze_cirr_kappa_behavior(blip_model, relative_val_dataset: CIRRDataset,
                                index_names: List[str], index_features: torch.Tensor,
                                txt_processors, val_index_kappas=None):
    """
    计算：
    - corr(k_q_raw, margin)
    - corr(k_q_final, margin)
    - corr(k_t, margin)
    - corr(k_v, margin)
    - low/mid/high bucket mean_margin
    - 三个 bucket 的样本数

    注意：
    这里的 margin 定义为：
        正样本相似度 - 最难负样本相似度
    并且是在整张 gallery 上计算，而不是 batch 内。
    """
    print("Analyze CIRR kappa behavior")

    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False
    )

    gallery_features = index_features.to(device)

    all_kq_raw = []
    all_kq_final = []
    all_kt = []
    all_kv = []
    all_margin = []

    # 建一个名字到索引的映射，方便找到 target 在 gallery 中的位置
    name_to_index = {name: idx for idx, name in enumerate(index_names)}

    for batch in tqdm(relative_val_loader, ncols=100):
        batch_ref_imgs, batch_ref_names, batch_tgt_names, batch_captions, batch_group_m = batch

        batch_captions = [txt_processors["eval"](c) for c in batch_captions]
        batch_ref_imgs = batch_ref_imgs.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                debug_out = blip_model.extract_query_debug_features(batch_ref_imgs, batch_captions)

                query_feat = debug_out["query_feat"]              # [B, D]
                k_q_raw = debug_out["k_q_raw"]                    # [B]
                # k_q_final = debug_out["k_q_final"]              # [B]
                k_q_final_mean = debug_out["k_q_final_mean"]
                k_q_final_tokens = debug_out["k_q_final_tokens"]
                k_t = debug_out["k_t"]                            # [B]
                k_v = debug_out["k_v"]                            # [B]

                # 用当前 inference（纯 cosine）算整库相似度
                # 修改 validate_blip.py 第 254 行附近：
                sims = blip_model.inference(
                    query_feat=query_feat,
                    k_query=k_q_final_tokens, 
                    target_feats=gallery_features,
                    k_targets=val_index_kappas.to(device) if val_index_kappas is not None else None
                )  # [B, N]

                # 对每个 query 找真实 target 的相似度
                target_indices = torch.tensor(
                    [name_to_index[name] for name in batch_tgt_names],
                    device=sims.device,
                    dtype=torch.long
                )  # [B]

                pos_sim = sims[torch.arange(sims.size(0), device=sims.device), target_indices]  # [B]

                # 屏蔽真实 target，自整库找 hardest negative
                neg_sims = sims.clone()
                neg_sims[torch.arange(sims.size(0), device=sims.device), target_indices] = -1e4
                hardest_neg = neg_sims.max(dim=1)[0]  # [B]

                margin = pos_sim - hardest_neg  # [B]

                all_kq_raw.append(k_q_raw.float().cpu().numpy())
                all_kq_final.append(k_q_final_mean.float().cpu().numpy()) 
                all_kt.append(k_t.float().cpu().numpy())
                all_kv.append(k_v.float().cpu().numpy())
                all_margin.append(margin.float().cpu().numpy())

    all_kq_raw = np.concatenate(all_kq_raw)
    all_kq_final = np.concatenate(all_kq_final)
    all_kt = np.concatenate(all_kt)
    all_kv = np.concatenate(all_kv)
    all_margin = np.concatenate(all_margin)

    def safe_corr(a, b):
        if len(a) < 2:
            return 0.0
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    corr_kq_raw = safe_corr(all_kq_raw, all_margin)
    corr_kq_final = safe_corr(all_kq_final, all_margin)
    corr_kt = safe_corr(all_kt, all_margin)
    corr_kv = safe_corr(all_kv, all_margin)

    # 按 k_q_raw 分三个 bucket
    q1 = np.quantile(all_kq_raw, 0.33)
    q2 = np.quantile(all_kq_raw, 0.66)

    low_mask = all_kq_raw <= q1
    mid_mask = (all_kq_raw > q1) & (all_kq_raw <= q2)
    high_mask = all_kq_raw > q2

    result = {
        "corr(k_q_raw, margin)": corr_kq_raw,
        "corr(k_q_final, margin)": corr_kq_final,
        "corr(k_t, margin)": corr_kt,
        "corr(k_v, margin)": corr_kv,

        "low bucket mean_margin": float(all_margin[low_mask].mean()) if low_mask.sum() > 0 else 0.0,
        "mid bucket mean_margin": float(all_margin[mid_mask].mean()) if mid_mask.sum() > 0 else 0.0,
        "high bucket mean_margin": float(all_margin[high_mask].mean()) if high_mask.sum() > 0 else 0.0,

        "low bucket size": int(low_mask.sum()),
        "mid bucket size": int(mid_mask.sum()),
        "high bucket size": int(high_mask.sum()),
    }

    print("\n===== CIRR KAPPA ANALYSIS =====")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result

def main():
    # 保持主函数逻辑，防止作为脚本运行时报错
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['CIRR', 'FashionIQ'])
    args = parser.parse_args()
    print(f"Validation functions loaded for {args.dataset}. Run training script to evaluate.")

if __name__ == '__main__':
    # 仅作提示
    print("This file contains validation functions. Please run 'src/blip_fine_tune_2.py' instead.")