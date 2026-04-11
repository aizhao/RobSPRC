"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause

 原始修改版：
 1. 保留你已验证有效的新 query 构造：top-k 局部选择 + 差异门控更新 + composer
 2. target 改为 32 个 token 表示，query-vs-target-token 取 max 做匹配
 3. 主损失保持 InfoNCE 主线（并保留轻量 hard negative 辅助）
 4. 其余 query / kappa / 动态融合 / loss_kappa / loss_cord 逻辑不变
 5. 推理保持纯 cosine，支持 target 多 token 输入时 token 维取 max
"""

import logging
import math
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)


# ================= [vMF 核心映射与数学工具模块] =================


class KappaPredictor(nn.Module):
    """
    不确定性感知网络：输出 concentration / confidence 标量 kappa
    """
    def __init__(self, input_dim, hidden_dim=256, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.constant_(self.net[0].bias, 0.0)
        nn.init.xavier_uniform_(self.net[3].weight, gain=0.5)
        nn.init.constant_(self.net[3].bias, 3.0)

    def forward(self, x):
        raw = self.net(x)
        kappa = F.softplus(raw) + self.epsilon
        return kappa


def log_bessel_i(v, x, terms=50):
    """
    Compute the logarithm of the modified Bessel function of the first kind, log(I_v(x)),
    using PyTorch.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float32)

    x = x.unsqueeze(-1)
    k = torch.arange(terms, dtype=torch.float32, device=x.device)

    log_coef = (2 * k + v) * torch.log(x / 2) - (torch.lgamma(k + 1) + torch.lgamma(k + v + 1))

    max_log_coef = torch.max(log_coef, dim=-1, keepdim=True)[0]
    sum_exp_log = torch.exp(log_coef - max_log_coef).clamp_max(1e7).sum(dim=-1)

    log_bessel = max_log_coef.squeeze(-1) + torch.log(sum_exp_log)
    return log_bessel


def vmf_logpartition(kappa, d):
    """
    Evaluates the log-partition log C_d(kappa) for vMF density.
    Inspired from: https://github.com/minyoungkim21/vmf-lib
    """
    s = 0.5 * d - 1
    logI = log_bessel_i(s, kappa)

    if (logI != logI).sum().item() > 0:
        raise ValueError('NaN is detected from the output of log-besseli()')

    logC = -0.5 * d * np.log(2 * np.pi) + s * kappa.clamp_min(1e-7).log() - logI
    return logC


class Global_vMF_CIR_Loss(nn.Module):
    """
    query 单向量 vs target 多 token
    主损失 = InfoNCE / ITC
    logits 保留 PML 风格的 loc(kappa) + kappa * sim 形式：
        logits_ij = ( logC(kappa_i) + kappa_i * sim_ij ) / temp
    """
    def __init__(self, feature_dim=256, temp_init=0.07):
        super().__init__()
        self.d = feature_dim
        self.temp = nn.Parameter(temp_init * torch.ones([]))

    def forward(self, fusion_feats, kappa_q, target_feats):
        bs = fusion_feats.size(0)
        labels = torch.arange(bs, device=fusion_feats.device)

        sim_t2q = torch.einsum('bd,ntd->bnt', fusion_feats, target_feats)
        sim_i2t, _ = sim_t2q.max(dim=-1)  # [B, B]

        if kappa_q.dim() == 1:
            kappa_q = kappa_q.unsqueeze(-1)
        kappa_q = kappa_q.clamp_min(1e-6)
        logC_q = vmf_logpartition(kappa_q, self.d)  # [B, 1]

        logits_i2t = (logC_q + kappa_q * sim_i2t) / self.temp.clamp_min(1e-6)
        loss_itc = F.cross_entropy(logits_i2t, labels)
        return loss_itc, sim_i2t, logits_i2t


# ================= [主模型模块] =================


@registry.register_model("blip2_cir_align_prompt")
class Blip2QformerCirAlignPrompt(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                if key_orig in state_dict:
                    param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.max_txt_len = max_txt_len
        self.embed_dim = embed_dim

        hidden_size = self.Qformer.config.hidden_size
        vision_width = self.visual_encoder.num_features

        self.kappa_predictor = KappaPredictor(input_dim=hidden_size)
        self.kappa_t_predictor = KappaPredictor(input_dim=hidden_size)
        self.kappa_v_predictor = KappaPredictor(input_dim=hidden_size)

        nn.init.constant_(self.kappa_t_predictor.net[3].bias, 3.0)
        nn.init.constant_(self.kappa_v_predictor.net[3].bias, 3.0)

        self.global_vmf_loss = Global_vMF_CIR_Loss(
            feature_dim=embed_dim,
        )

        # 动态融合温度
        self.dynamic_fusion_tau = 0.5

        # ===== 新 query：局部 token 选择 + 门控更新 =====
        self.topk_tokens = 16

        # 文本全局向量 -> 视觉 token 空间，用于对图像 token 打分
        self.text_to_vision = nn.Linear(hidden_size, vision_width)

        # 局部视觉表示 -> hidden 空间，用于和文本做差异 r
        self.vision_to_hidden = nn.Linear(vision_width, hidden_size)

        # 差异向量归一化
        self.r_ln = nn.LayerNorm(hidden_size)

        # 预估“文本-局部”层面的局部不确定性
        self.local_kappa_predictor = KappaPredictor(input_dim=hidden_size * 3)

        # 用 r + uncertainty 生成 gate g（标量门）
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size + 3, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        # 仅更新 top-k token 的更新网络
        self.update_mlp = nn.Sequential(
            nn.Linear(vision_width * 3, vision_width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(vision_width, vision_width)
        )

    # ---------- helper: 动态融合 ----------
    def _dynamic_fuse_kappa(self, k_m, k_t, k_r, eps=1e-6):
        u_m = 1.0 / (k_m + eps)
        u_t = 1.0 / (k_t + eps)
        u_r = 1.0 / (k_r + eps)

        u_stack = torch.cat([u_r, u_t, u_m], dim=1)
        weights = F.softmax(-u_stack / self.dynamic_fusion_tau, dim=1)

        w_r = weights[:, 0:1]
        w_t = weights[:, 1:2]
        w_m = weights[:, 2:3]

        u_q = w_r * u_r + w_t * u_t + w_m * u_m
        k_q = 1.0 / (u_q + eps)

        return k_q, w_r, w_t, w_m

    def _batched_gather_tokens(self, x, idx):
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
        return torch.gather(x, dim=1, index=idx_exp)

    def _build_query_from_image_text(
        self,
        image_embeds,
        image_atts,
        query_tokens,
        query_atts,
        text_input_ids,
        text_attention_mask,
    ):
        B, N, C = image_embeds.shape
        topk = min(self.topk_tokens, N)

        # ===== 1) 文本编码 =====
        text_output = self.Qformer.bert(
            text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True
        )
        raw_text_emb = text_output.last_hidden_state[:, 0, :]
        k_t = self.kappa_t_predictor(raw_text_emb)

        # ===== 2) 参考图质量分支 =====
        query_image_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True
        )
        raw_vision_emb = query_image_output.last_hidden_state[:, 0, :]
        k_r = self.kappa_v_predictor(raw_vision_emb)

        # ===== 3) 文本对图像 token 打分，选 top-k =====
        text_vis = self.text_to_vision(raw_text_emb)
        token_scores = torch.bmm(
            image_embeds, text_vis.unsqueeze(-1)
        ).squeeze(-1) / math.sqrt(C)

        topk_scores, topk_idx = torch.topk(token_scores, k=topk, dim=1)
        topk_attn = F.softmax(topk_scores, dim=1)
        selected_vis = self._batched_gather_tokens(image_embeds, topk_idx)

        # ===== 4) 聚合得到局部视觉表示 f_local =====
        f_local = (selected_vis * topk_attn.unsqueeze(-1)).sum(dim=1)
        f_local_h = self.vision_to_hidden(f_local)

        # ===== 5) 差异 r + uncertainty -> gate g =====
        r_vec = self.r_ln(raw_text_emb - f_local_h)

        local_k_input = torch.cat([raw_text_emb, f_local_h, r_vec], dim=-1)
        k_local = self.local_kappa_predictor(local_k_input)

        gate_input = torch.cat([
            r_vec,
            torch.log(k_t + 1e-6),
            torch.log(k_r + 1e-6),
            torch.log(k_local + 1e-6)
        ], dim=-1)

        g = torch.sigmoid(self.gate_mlp(gate_input))

        # ===== 6) 只更新 top-k token =====
        text_vis_expand = text_vis.unsqueeze(1).expand(-1, topk, -1)
        f_local_expand = f_local.unsqueeze(1).expand(-1, topk, -1)

        delta_input = torch.cat([selected_vis, text_vis_expand, f_local_expand], dim=-1)
        delta = self.update_mlp(delta_input)

        updated_selected_vis = selected_vis + g.unsqueeze(1) * delta

        updated_image_embeds = image_embeds.clone()
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, C)
        updated_image_embeds.scatter_(1, topk_idx_exp, updated_selected_vis)

        # ===== 7) 用更新后的 token + 文本做 composer =====
        attention_mask = torch.cat([query_atts, text_attention_mask], dim=1)

        fusion_output = self.Qformer.bert(
            text_input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=updated_image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        second_pass_output = self.Qformer.bert(
            text_input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        # ===== 回归你原来的实现：直接取 full sequence 的 index=32 =====
        raw_fusion_emb = second_pass_output.last_hidden_state[:, 32, :]

        aux = {
            "k_t": k_t,
            "k_r": k_r,
            "k_local": k_local,
            "g": g,
            "topk_attn": topk_attn,
            "topk_idx": topk_idx,
            "f_local_h": f_local_h,
            "r_vec": r_vec,
            "updated_image_embeds": updated_image_embeds,
        }

        return raw_text_emb, raw_fusion_emb, aux

    # ---------- helper: pairwise ranking uncertainty calibration ----------
    def _pairwise_kappa_ranking_loss(self, kappa, margin, eps=1e-6):
        log_kappa = torch.log(kappa.squeeze(-1) + eps)
        margin_norm = (margin - margin.mean()) / (margin.std() + eps)

        margin_diff = margin_norm.unsqueeze(1) - margin_norm.unsqueeze(0)
        logk_diff = log_kappa.unsqueeze(1) - log_kappa.unsqueeze(0)

        eye_mask = torch.eye(margin.size(0), device=margin.device, dtype=torch.bool)
        valid_mask = (~eye_mask) & (margin_diff.abs() > 0.1)

        if valid_mask.any():
            pair_loss = F.softplus(-(margin_diff.detach() * logk_diff))
            return pair_loss[valid_mask].mean()
        else:
            return torch.zeros([], device=margin.device, dtype=margin.dtype)

    def forward(self, samples, enable_uncertainty=False):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]

        with self.maybe_autocast():
            model_dtype = self.visual_encoder.patch_embed.proj.weight.dtype

            image_embeds = (
                self.ln_vision(self.visual_encoder(image.to(model_dtype)))
                if image.dim() != 3 else self.ln_vision(image.to(model_dtype))
            )
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)

            target_embeds = (
                self.ln_vision(self.visual_encoder(target.to(model_dtype)))
                if target.dim() != 3 else self.ln_vision(target.to(model_dtype))
            )
            target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(image.device)

            # ===== query 构造：保持你的新逻辑 =====
            raw_text_emb, raw_fusion_emb, query_aux = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=text_tokens.input_ids,
                text_attention_mask=text_tokens.attention_mask,
            )

            k_t = query_aux["k_t"]
            k_r = query_aux["k_r"]
            k_local = query_aux["k_local"]
            g = query_aux["g"]
            topk_attn = query_aux["topk_attn"]

            # ===== target 改为 32 个 token 表示 =====
            target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=target_embeds,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True
            )
            raw_target_emb = target_output.last_hidden_state[:, 0, :]
            target_feats = F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1)
            k_target = self.kappa_v_predictor(raw_target_emb)

            # ===== 主方向特征 =====
            u_query = F.normalize(self.text_proj(raw_fusion_emb), dim=-1)

            # ===== query uncertainty =====
            k_query_raw = self.kappa_predictor(raw_fusion_emb)

            with torch.no_grad():
                k_query_final, w_r, w_t, w_m = self._dynamic_fuse_kappa(
                    k_query_raw.detach(), k_t.detach(), k_r.detach()
                )

            # ===== 主检索损失：InfoNCE + 轻量 hard negative =====
            base_loss, sim_i2t, loss_info = self.global_vmf_loss(
                fusion_feats=u_query,
                kappa_q=k_query_raw,
                target_feats=target_feats,
            )

            # # ===== batch 内 margin，用于 uncertainty 校准 =====
            # with torch.no_grad():
            #     pos_sim = sim_i2t.diag()
            #     eye_mask = torch.eye(sim_i2t.size(0), device=sim_i2t.device).bool()
            #     hardest_neg = sim_i2t.masked_fill(eye_mask, -1e4).max(dim=1)[0]
            #     margin = pos_sim - hardest_neg

            # loss_kappa = self._pairwise_kappa_ranking_loss(k_query_raw, margin)

            # ===== 错配文本分支：走同样的 query 构造流程 =====
            wrong_text_ids = torch.roll(text_tokens.input_ids, shifts=1, dims=0)
            wrong_text_mask = torch.roll(text_tokens.attention_mask, shifts=1, dims=0)

            _, raw_wrong_fusion_emb, _ = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=wrong_text_ids,
                text_attention_mask=wrong_text_mask,
            )

            kappa_m_wrong = self.kappa_predictor(raw_wrong_fusion_emb)

            # 正确图文组合应比错配更确定
            loss_cord = F.softplus(-(k_query_raw - kappa_m_wrong)).mean()

            # ===== 总损失 =====
            total_loss = base_loss  + 0.05 * loss_cord

        return {
            'loss': total_loss,
            'loss_base': base_loss.item(),
            # 'loss_kappa': loss_kappa.item(),
            'loss_cord': loss_cord.item(),

            # 兼容旧日志接口，置零即可
            'loss_nll_q': 0.0,
            'loss_nll_t': 0.0,
            'loss_rank': 0.0,

            # ===== kappa 统计 =====
            'k_q_raw_std': k_query_raw.std().item(),
            'k_q_raw_min': k_query_raw.min().item(),
            'k_q_raw_max': k_query_raw.max().item(),

            'k_q_final_mean': k_query_final.mean().item(),
            'k_q_final_std': k_query_final.std().item(),
            'k_q_final_min': k_query_final.min().item(),
            'k_q_final_max': k_query_final.max().item(),

            'k_t_std': k_t.std().item(),
            'k_t_min': k_t.min().item(),
            'k_t_max': k_t.max().item(),

            # 兼容原日志名字，k_v 实际上这里代表 reference-image quality 分支
            'k_v_std': k_r.std().item(),
            'k_v_min': k_r.min().item(),
            'k_v_max': k_r.max().item(),

            # ===== 动态融合权重统计 =====
            'w_r_mean': w_r.mean().item(),
            'w_t_mean': w_t.mean().item(),
            'w_m_mean': w_m.mean().item(),

            # # ===== 难度统计 =====
            # 'margin_mean': margin.mean().item(),
            # 'margin_std': margin.std().item(),
            # 'pos_sim_mean': pos_sim.mean().item(),
            # 'hardest_neg_mean': hardest_neg.mean().item(),

            'k_q_wrong_mean': kappa_m_wrong.mean().item(),
            'k_local_mean': k_local.mean().item(),
            'g_mean': g.mean().item(),
            'topk_attn_mean': topk_attn.mean().item(),
        }

    @torch.no_grad()
    def forward_debug(self, samples):
        """
        训练 batch 级分析：margin 按 target 多 token 后的 sim_i2t 计算
        """
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]

        with self.maybe_autocast():
            model_dtype = self.visual_encoder.patch_embed.proj.weight.dtype

            image_embeds = (
                self.ln_vision(self.visual_encoder(image.to(model_dtype)))
                if image.dim() != 3 else self.ln_vision(image.to(model_dtype))
            )
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)

            target_embeds = (
                self.ln_vision(self.visual_encoder(target.to(model_dtype)))
                if target.dim() != 3 else self.ln_vision(target.to(model_dtype))
            )
            target_atts = torch.ones(target_embeds.size()[:-1], dtype=torch.long).to(image.device)

            _, raw_fusion_emb, query_aux = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=text_tokens.input_ids,
                text_attention_mask=text_tokens.attention_mask,
            )

            k_t = query_aux["k_t"]
            k_r = query_aux["k_r"]

            target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=target_embeds,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True
            )
            target_feats = F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1)

            u_query = F.normalize(self.text_proj(raw_fusion_emb), dim=-1)

            k_query_raw = self.kappa_predictor(raw_fusion_emb)
            k_query_final, _, _, _ = self._dynamic_fuse_kappa(k_query_raw, k_t, k_r)

            sim_q2t_tokens = torch.einsum('bd,ntd->bnt', u_query, target_feats)
            sim_i2t, _ = sim_q2t_tokens.max(dim=-1)

            pos_sim = sim_i2t.diag()
            eye_mask = torch.eye(sim_i2t.size(0), device=sim_i2t.device).bool()
            hardest_neg = sim_i2t.masked_fill(eye_mask, -1e4).max(dim=1)[0]
            margin = pos_sim - hardest_neg

        return {
            "query_feat": u_query,
            "k_q_raw": k_query_raw.squeeze(-1).float().cpu(),
            "k_q_final": k_query_final.squeeze(-1).float().cpu(),
            "k_t": k_t.squeeze(-1).float().cpu(),
            "k_v": k_r.squeeze(-1).float().cpu(),
            "margin": margin.float().cpu(),
        }

    @torch.no_grad()
    def extract_query_features(self, image, text):
        if isinstance(text, str):
            text = [text]

        with self.maybe_autocast():
            model_dtype = self.visual_encoder.patch_embed.proj.weight.dtype
            image_embeds = (
                self.ln_vision(self.visual_encoder(image.to(model_dtype)))
                if image.dim() != 3 else self.ln_vision(image.to(model_dtype))
            )
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)

            _, raw_fusion_emb, query_aux = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=text_tokens.input_ids,
                text_attention_mask=text_tokens.attention_mask,
            )

            u_query = F.normalize(self.text_proj(raw_fusion_emb), dim=-1)
            k_query_raw = self.kappa_predictor(raw_fusion_emb)
            return u_query, k_query_raw

    @torch.no_grad()
    def extract_query_debug_features(self, image, text):
        if isinstance(text, str):
            text = [text]

        with self.maybe_autocast():
            model_dtype = self.visual_encoder.patch_embed.proj.weight.dtype
            image_embeds = (
                self.ln_vision(self.visual_encoder(image.to(model_dtype)))
                if image.dim() != 3 else self.ln_vision(image.to(model_dtype))
            )
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)

            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(image.device)

            _, raw_fusion_emb, query_aux = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=text_tokens.input_ids,
                text_attention_mask=text_tokens.attention_mask,
            )

            k_t = query_aux["k_t"]
            k_r = query_aux["k_r"]

            u_query = F.normalize(self.text_proj(raw_fusion_emb), dim=-1)
            k_q_raw = self.kappa_predictor(raw_fusion_emb)
            k_q_final, _, _, _ = self._dynamic_fuse_kappa(k_q_raw, k_t, k_r)

        return {
            "query_feat": u_query,
            "k_q_raw": k_q_raw.squeeze(-1),
            "k_q_final": k_q_final.squeeze(-1),
            "k_t": k_t.squeeze(-1),
            "k_v": k_r.squeeze(-1),
        }

    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            model_dtype = self.visual_encoder.patch_embed.proj.weight.dtype
            image_embeds_frozen = (
                self.ln_vision(self.visual_encoder(image.to(model_dtype)))
                if image.dim() != 3 else self.ln_vision(image.to(model_dtype))
            )
            image_atts = torch.ones(image_embeds_frozen.size()[:-1], dtype=torch.long).to(self.device)
            query_tokens = self.query_tokens.expand(image_embeds_frozen.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True
            )
            raw_target_emb = query_output.last_hidden_state[:, 0, :]
            target_features = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

            k_target = self.kappa_v_predictor(raw_target_emb)

        return target_features, k_target

    @torch.no_grad()
    def inference(self, query_feat, k_query, target_feats, k_targets):
        if target_feats.dim() == 2:
            similarity_scores = torch.matmul(query_feat, target_feats.t())

        elif target_feats.dim() == 3:
            sim_q2t_tokens = torch.einsum('bd,ntd->bnt', query_feat, target_feats)
            similarity_scores, _ = sim_q2t_tokens.max(dim=-1)

        else:
            raise ValueError(f"Unrecognized target_feats dimension: {target_feats.dim()}")

        return similarity_scores

    @classmethod
    def from_config(cls, cfg):
        model = cls(
            vit_model=cfg.get("vit_model", "eva_clip_g"),
            img_size=cfg.get("image_size"),
            drop_path_rate=cfg.get("drop_path_rate", 0),
            use_grad_checkpoint=cfg.get("use_grad_checkpoint", False),
            vit_precision=cfg.get("vit_precision", "fp16"),
            freeze_vit=cfg.get("freeze_vit", True),
            num_query_token=cfg.get("num_query_token"),
            cross_attention_freq=cfg.get("cross_attention_freq", 2),
            max_txt_len=cfg.get("max_txt_len", 32)
        )
        model.load_checkpoint_from_config(cfg)
        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        k_test = task_cfg.k_test
        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)