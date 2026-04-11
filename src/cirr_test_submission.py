import json
import argparse  # 🔥 修复 1：补全了 argparse 的导入
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from data_utils import CIRRDataset, targetpad_transform, base_path
# ⚠️ 注意：确保你的 utils.py 中的 extract_index_blip_features 已经更新，
# 能够返回 (features, kappas, names) 三个变量！
from utils import device, extract_index_blip_features
from lavis.models import load_model_and_preprocess

def get_basename(n):
    return Path(str(n)).stem

def generate_cirr_test_submissions(file_name: str, blip_model, preprocess, txt_processors, rerank):
    classic_test_dataset = CIRRDataset('test1', 'classic', preprocess)
    
    # 🔥 修复 2：接收图库的特征 (features) 和 置信度 (kappas)
    print("Extracting gallery features and kappas...")
    index_features, index_kappas, index_names = extract_index_blip_features(classic_test_dataset, blip_model)
    
    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess)

    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(
        relative_test_dataset, blip_model, index_features, index_kappas, index_names, txt_processors, rerank, preprocess
    )

    submission = {'version': 'rc2', 'metric': 'recall'}
    group_submission = {'version': 'rc2', 'metric': 'recall_subset'}

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    submissions_folder_path = base_path / "submission" / 'CIRR'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    if rerank:
        file_name = file_name + f'_{rerank}'

    print(f"Saving CIRR test predictions to {submissions_folder_path}")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset: CIRRDataset, blip_model, 
                             index_features: torch.Tensor, index_kappas: torch.Tensor,
                             index_names: List[str], txt_processors, rerank, preprocess) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:

    predicted_sim, reference_names, group_members, pairs_id, captions_all, name2feat = \
        generate_cirr_test_predictions(blip_model, relative_test_dataset, index_names,
                                       index_features, index_kappas, txt_processors, preprocess)

    print(f"Compute CIRR prediction dicts")
    distances = 1 - predicted_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    if rerank:
        print('Reranking now...')
        i = 0
        step = 50
        top = 50
        while i < len(sorted_index_names):
            if step + i > len(sorted_index_names):
                step = len(sorted_index_names) - i
            reference_name = reference_names[i: i + step]
            caption = captions_all[i: i + step]
            targets_top100 = sorted_index_names[i: i + step, :top]
            
            reference_feats = torch.stack([name2feat[get_basename(n)] for n in reference_name]).to(device)
            target_feats = torch.stack([name2feat[get_basename(n)] for n in targets_top100.reshape(-1)]).to(device)
            
            with torch.no_grad():
                top100_rank = blip_model.inference_rerank(reference_feats, target_feats, caption)
            distances_top100 = 1 - top100_rank
            distances_top100 = distances_top100.reshape(-1, top)
            sorted_indices_top100 = torch.argsort(distances_top100, dim=-1).cpu()
            for j in range(step):
                sorted_index_names[i + j, :top] = sorted_index_names[i + j, :top][sorted_indices_top100[j]]
            i = i + step

    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1)
    )
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1] - 1)
    
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    pairid_to_predictions = {str(int(pair_id)): [get_basename(p) for p in prediction[:50]] for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): [get_basename(p) for p in prediction[:3]] for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(blip_model, relative_test_dataset: CIRRDataset, index_names: List[str], 
                                   index_features: torch.Tensor, index_kappas: torch.Tensor, 
                                   txt_processors, preprocess):
    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=4, pin_memory=True)

    safe_index_names = [get_basename(n) for n in index_names]
    # 保留用于 rerank 的图像特征字典 (不包含 kappa)
    name_to_feat = dict(zip(safe_index_names, index_features))

    pairs_id = []
    group_members = []
    reference_names = []
    distance = []
    captions_all = []

    # 🔥 修复 3：图库特征和置信度一次性推到 GPU，拒绝在循环内反复搬砖！
    target_features = index_features.to(device)
    target_kappas = index_kappas.to(device)

    blip_model.eval()
    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(relative_test_loader):
        batch_group_members = np.array(batch_group_members).T.tolist()
        captions = [txt_processors["eval"](caption) for caption in captions]
        safe_batch_ref_names = [get_basename(n) for n in batch_reference_names]

        with torch.no_grad():
            batch_ref_imgs = []
            
            # 读取当前 batch 的所有参考图像
            for name in safe_batch_ref_names:
                img_path = base_path / 'cirr_dataset' / 'test1' / f"{name}.png"
                if not img_path.exists():
                    img_path = base_path / 'cirr_dataset' / 'test1' / 'images' / f"{name}.png"
                
                img = Image.open(img_path).convert('RGB')
                processed_img = preprocess(img)
                batch_ref_imgs.append(processed_img)

            # 堆叠成 [Batch_size, C, H, W] 的 Tensor
            reference_images_tensor = torch.stack(batch_ref_imgs).to(device)
            
            # 🔥 修复 4：调用正确的融合方法，同时获取联合 Query 特征和联合置信度
            mu_q, k_q = blip_model.extract_query_features(reference_images_tensor, captions)

            # 🔥 修复 5：严格传递 vMF 模型要求的 4 个参数 (特征 + 置信度)
            batch_distance = blip_model.inference(
                query_feat=mu_q, 
                k_query=k_q, 
                target_feats=target_features, 
                k_targets=target_kappas
            )
            
            distance.append(batch_distance.cpu())
            captions_all += captions

        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    distance = torch.vstack(distance)
    return distance, reference_names, group_members, pairs_id, captions_all, name_to_feat


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # argparse 导入已修复，这里不会再报 NameError
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = ArgumentParser()
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str)
    parser.add_argument("--model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
    parser.add_argument("--rerank", type=str2bool, default=False)
    args = parser.parse_args()
    
    blip_model, _, txt_processors = load_model_and_preprocess(name=args.blip_model_name, model_type=args.backbone, is_eval=False, device=device)
    
    checkpoint_path = args.model_path

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 明确告诉它去取 'model' 这个 key 里的权重字典
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    msg = blip_model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", len(msg.missing_keys))
    if len(msg.missing_keys) > 0:
        print("First 10 missing keys:", msg.missing_keys[:10])

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)

    generate_cirr_test_submissions(f'{args.blip_model_name}_2', blip_model, preprocess, txt_processors, args.rerank)

if __name__ == '__main__':
    main()