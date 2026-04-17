"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause

 vMF-HUG style CIR
 1. 保留局部选择 + 差异门控更新 + composer
 2. Query / Target 均采用 32 token dense vMF representation
 3. 主匹配：标准单向 KL(q || t) + MaxSim + 熵加权聚合
 4. 新增独立 coordination uncertainty branch: kappa_m
 5. 新增 dynamic fusion，使 query uncertainty 真正参与主损失
 6. 新增 fine-grained vMF contrast
 7. 全部带开关，便于消融实验
"""

import logging
import math
import numpy as np
import torch
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


# ================= [基础模块] =================


class KappaPredictor(nn.Module):
    """
    输出 concentration / confidence 标量 kappa
    支持 2D: [B, D]
    支持 3D: [B, T, D]
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
    计算 log(I_v(x))
    支持广播输入。
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(v):
        v = torch.tensor(v, dtype=torch.float32, device=x.device)

    x = x.clamp_min(1e-8)
    x_unsq = x.unsqueeze(-1)

    k = torch.arange(terms, dtype=x.dtype, device=x.device)
    log_coef = (2 * k + v) * torch.log(x_unsq / 2.0) - (
        torch.lgamma(k + 1.0) + torch.lgamma(k + v + 1.0)
    )

    max_log_coef = torch.max(log_coef, dim=-1, keepdim=True)[0]
    sum_exp_log = torch.exp(log_coef - max_log_coef).sum(dim=-1).clamp_min(1e-30)

    return max_log_coef.squeeze(-1) + torch.log(sum_exp_log)


def vmf_logpartition(kappa, d):
    """
    log C_d(kappa)
    vMF density:
        f(x; mu, kappa) = C_d(kappa) exp(kappa * mu^T x)
    """
    kappa = kappa.clamp_min(1e-8)
    nu = 0.5 * d - 1.0
    logI = log_bessel_i(nu, kappa)
    logC = nu * torch.log(kappa) - 0.5 * d * math.log(2.0 * math.pi) - logI
    return logC


def vmf_A(kappa, d):
    """
    A_d(kappa) = I_{d/2}(kappa) / I_{d/2 - 1}(kappa)
    """
    kappa = kappa.clamp_min(1e-8)
    nu = 0.5 * d - 1.0

    logI_nu = log_bessel_i(nu, kappa)
    logI_nu1 = log_bessel_i(nu + 1.0, kappa)

    A = torch.exp((logI_nu1 - logI_nu).clamp(min=-60.0, max=0.0))
    return A.clamp(min=1e-8, max=1.0 - 1e-6)


def vmf_entropy(kappa, d):
    """
    标准 vMF 熵：
        H = -log C_d(kappa) - kappa * A_d(kappa)
    """
    kappa = kappa.clamp_min(1e-8)
    return -vmf_logpartition(kappa, d) - kappa * vmf_A(kappa, d)


def vmf_kl(kappa_p, mu_dot, kappa_q, d):
    """
    标准单向 KL:
        KL( vMF(mu_p, kappa_p) || vMF(mu_q, kappa_q) )
        = log C(kappa_p) - log C(kappa_q)
          + A_d(kappa_p) * (kappa_p - kappa_q * <mu_p, mu_q>)
    """
    kappa_p = kappa_p.clamp_min(1e-8)
    kappa_q = kappa_q.clamp_min(1e-8)

    logC_p = vmf_logpartition(kappa_p, d)
    logC_q = vmf_logpartition(kappa_q, d)
    A_p = vmf_A(kappa_p, d)

    return logC_p - logC_q + A_p * (kappa_p - kappa_q * mu_dot)


class TokenWise_vMF_KL_CIR_Loss(nn.Module):
    """
    Query 多 Token vs Target 多 Token
    - token-pair 度量：标准单向 KL(q || t)
    - target token 维：MaxSim (最大负 KL，也就是最小 KL)
    - query token 聚合：用 vMF 熵做权重
    """
    def __init__(self, feature_dim=256, temp_init=0.07, token_weight_temp=0.5):
        super().__init__()
        self.d = feature_dim
        self.temp = nn.Parameter(temp_init * torch.ones([]))
        self.token_weight_temp = token_weight_temp

    def compute_score(self, query_tokens, kappa_q_tokens, target_tokens, kappa_t_tokens):
        """
        返回：
            score_q2t: [B, N, T_t, T_q] = -KL(q || t)
            entropy_q: [B, T_q]
        """
        sim_q2t = torch.einsum('bqd,ntd->bntq', query_tokens, target_tokens)

        kappa_q_tokens = kappa_q_tokens.clamp_min(1e-6)
        kappa_t_tokens = kappa_t_tokens.clamp_min(1e-6)

        kq = kappa_q_tokens.unsqueeze(1).unsqueeze(1)   # [B,1,1,T_q]
        kt = kappa_t_tokens.unsqueeze(0).unsqueeze(-1)  # [1,N,T_t,1]

        kl_q_t = vmf_kl(
            kappa_p=kq,
            mu_dot=sim_q2t,
            kappa_q=kt,
            d=self.d
        )  # [B,N,T_t,T_q]

        score_q2t = -kl_q_t
        entropy_q = vmf_entropy(kappa_q_tokens, self.d)   # [B,T_q]
        return score_q2t, entropy_q

    def forward(self, query_tokens, kappa_q_tokens, target_tokens, kappa_t_tokens):
        bs, T_q, _ = query_tokens.size()
        labels = torch.arange(bs, device=query_tokens.device)

        score_q2t, entropy_q = self.compute_score(
            query_tokens=query_tokens,
            kappa_q_tokens=kappa_q_tokens,
            target_tokens=target_tokens,
            kappa_t_tokens=kappa_t_tokens,
        )

        # MaxSim over target tokens
        max_score_i2t, _ = score_q2t.max(dim=2)   # [B,N,T_q]

        token_weights = F.softmax(
            -entropy_q.detach() / self.token_weight_temp, dim=-1
        )  # [B,T_q]

        score_global = (max_score_i2t * token_weights.unsqueeze(1)).sum(dim=-1)  # [B,N]
        logits_global = score_global / self.temp.clamp_min(1e-6)

        loss_each = F.cross_entropy(logits_global, labels, reduction="none")
        loss_itc = loss_each.mean()

        return loss_itc, logits_global, score_global, entropy_q, score_q2t


# ================= [主模型] =================


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
        max_txt_len=32,

        # ===== ablation switches =====
        use_coord_branch=True,
        use_dynamic_fusion=True,
        use_fine_grained_loss=True,

        fuse_coord_unc=True,
        fuse_text_unc=True,
        fuse_image_unc=True,

        lambda_cord=0.1,
        lambda_fine=0.40,
        lambda_rtc=0.40,
        lambda_align=0.40,

        fine_temp=0.07,
        fine_token_weight_temp=0.5,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        if freeze_vit:
            for _, param in self.visual_encoder.named_parameters():
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

        # ===== ablation switches =====
        self.use_coord_branch = use_coord_branch
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_fine_grained_loss = use_fine_grained_loss

        self.fuse_coord_unc = fuse_coord_unc
        self.fuse_text_unc = fuse_text_unc
        self.fuse_image_unc = fuse_image_unc

        self.lambda_cord = lambda_cord
        self.lambda_fine = lambda_fine
        self.lambda_rtc = lambda_rtc
        self.lambda_align = lambda_align

        self.fine_temp = fine_temp
        self.fine_token_weight_temp = fine_token_weight_temp

        hidden_size = self.Qformer.config.hidden_size
        vision_width = self.visual_encoder.num_features

        # ===== uncertainty heads =====
        self.kappa_predictor = KappaPredictor(input_dim=hidden_size)      # raw query token uncertainty
        self.kappa_t_predictor = KappaPredictor(input_dim=hidden_size)    # text/global uncertainty
        self.kappa_v_predictor = KappaPredictor(input_dim=hidden_size)    # image/global or target token uncertainty

        nn.init.constant_(self.kappa_t_predictor.net[3].bias, 3.0)
        nn.init.constant_(self.kappa_v_predictor.net[3].bias, 3.0)

        # ===== holistic retrieval loss =====
        self.token_vmf_loss = TokenWise_vMF_KL_CIR_Loss(
            feature_dim=embed_dim,
            temp_init=0.07,
            token_weight_temp=0.6,
        )

        self.dynamic_fusion_tau = 0.4

        # ===== 局部 token 选择 + 门控更新 =====
        self.topk_tokens = 8

        self.text_to_vision = nn.Linear(hidden_size, vision_width)
        self.vision_to_hidden = nn.Linear(vision_width, hidden_size)
        self.r_ln = nn.LayerNorm(hidden_size)

        self.local_kappa_predictor = KappaPredictor(input_dim=hidden_size * 3)

        # ===== 独立 coordination uncertainty branch =====
        self.kappa_m_predictor = KappaPredictor(input_dim=hidden_size * 3)

        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size + 3, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(vision_width * 3, vision_width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(vision_width, vision_width)
        )

        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(
            mean=0.0, std=self.Qformer.config.initializer_range
        )

    # ---------- helper ----------
    def _expand_like(self, kappa_scalar_or_tokens, ref_tokens):
        """
        把 [B,1] / [B] 扩成 [B,T]，或直接返回 [B,T]
        """
        if kappa_scalar_or_tokens is None:
            return None

        if kappa_scalar_or_tokens.dim() == 1:
            kappa_scalar_or_tokens = kappa_scalar_or_tokens.unsqueeze(-1)

        if kappa_scalar_or_tokens.dim() == 2 and kappa_scalar_or_tokens.size(1) == 1:
            return kappa_scalar_or_tokens.expand(-1, ref_tokens.size(1))

        return kappa_scalar_or_tokens

    def _dynamic_fuse_kappa(self, k_q_raw, k_m=None, k_t=None, k_r=None, eps=1e-6):
        """
        动态融合 query uncertainty
        第一版默认最推荐：
          - 一定包含 k_q_raw
          - 可选融合 k_m / k_t / k_r
        """
        sources = [("q", k_q_raw)]

        if self.use_coord_branch and self.fuse_coord_unc and (k_m is not None):
            sources.append(("m", self._expand_like(k_m, k_q_raw)))

        if self.fuse_text_unc and (k_t is not None):
            sources.append(("t", self._expand_like(k_t, k_q_raw)))

        if self.fuse_image_unc and (k_r is not None):
            sources.append(("r", self._expand_like(k_r, k_q_raw)))

        u_stack = torch.stack([1.0 / (v + eps) for _, v in sources], dim=-1)  # [B,T,S]
        weights = F.softmax(-u_stack / self.dynamic_fusion_tau, dim=-1)

        u_q = (weights * u_stack).sum(dim=-1)
        k_q = 1.0 / (u_q + eps)

        # for logging
        w_q = torch.zeros_like(k_q)
        w_t = torch.zeros_like(k_q)
        w_r = torch.zeros_like(k_q)
        w_m = torch.zeros_like(k_q)

        for idx, (name, _) in enumerate(sources):
            if name == "q":
                w_q = weights[..., idx]
            elif name == "t":
                w_t = weights[..., idx]
            elif name == "r":
                w_r = weights[..., idx]
            elif name == "m":
                w_m = weights[..., idx]

        return k_q, w_q, w_t, w_r, w_m

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

        # text global
        text_output = self.Qformer.bert(
            text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True
        )
        raw_text_emb = text_output.last_hidden_state[:, 0, :]
        k_t = self.kappa_t_predictor(raw_text_emb)  # [B,1]

        # image global
        query_image_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True
        )
        raw_vision_emb = query_image_output.last_hidden_state[:, 0, :]
        k_r = self.kappa_v_predictor(raw_vision_emb)  # [B,1]

        # text-guided top-k selection
        text_vis = self.text_to_vision(raw_text_emb)
        token_scores = torch.bmm(
            image_embeds, text_vis.unsqueeze(-1)
        ).squeeze(-1) / math.sqrt(C)

        topk_scores, topk_idx = torch.topk(token_scores, k=topk, dim=1)
        topk_attn = F.softmax(topk_scores, dim=1)
        selected_vis = self._batched_gather_tokens(image_embeds, topk_idx)

        f_local = (selected_vis * topk_attn.unsqueeze(-1)).sum(dim=1)
        f_local_h = self.vision_to_hidden(f_local)

        r_vec = self.r_ln(raw_text_emb - f_local_h)

        local_k_input = torch.cat([raw_text_emb, f_local_h, r_vec], dim=-1)
        k_local = self.local_kappa_predictor(local_k_input)

        # ===== 独立 coordination uncertainty =====
        coord_k_input = torch.cat([raw_text_emb, f_local_h, r_vec], dim=-1)
        k_m = self.kappa_m_predictor(coord_k_input)

        gate_input = torch.cat([
            r_vec,
            torch.log(k_t + 1e-6),
            torch.log(k_r + 1e-6),
            torch.log(k_local + 1e-6)
        ], dim=-1)

        g = torch.sigmoid(self.gate_mlp(gate_input))

        text_vis_expand = text_vis.unsqueeze(1).expand(-1, topk, -1)
        f_local_expand = f_local.unsqueeze(1).expand(-1, topk, -1)

        delta_input = torch.cat([selected_vis, text_vis_expand, f_local_expand], dim=-1)
        delta = self.update_mlp(delta_input)

        updated_selected_vis = selected_vis + g.unsqueeze(1) * delta

        updated_image_embeds = image_embeds.clone()
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, C)
        updated_image_embeds.scatter_(1, topk_idx_exp, updated_selected_vis)

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

        raw_fusion_tokens = second_pass_output.last_hidden_state[:, : query_tokens.size(1), :]

        aux = {
            "k_t": k_t,
            "k_r": k_r,
            "k_local": k_local,
            "k_m": k_m,
            "g": g,
            "topk_attn": topk_attn,
            "topk_idx": topk_idx,
            "f_local_h": f_local_h,
            "r_vec": r_vec,
            "updated_image_embeds": updated_image_embeds,
        }

        return raw_text_emb, raw_fusion_tokens, aux, fusion_output

    def _compute_fine_grained_vmf_loss(self, score_q2t, entropy_q):
        """
        score_q2t: [B, N, T_t, T_q] = token-pair score
        entropy_q: [B, T_q]

        component-level multi-positive InfoNCE:
        - positives: diagonal target instance 内所有 target components
        - negatives: 其它 instance 的所有 target components
        """
        B, N, T_t, T_q = score_q2t.shape
        assert B == N, "当前实现默认 batch 内 query 和 target 一一对应"

        # [B, T_q, N*T_t]
        logits = score_q2t.permute(0, 3, 1, 2).reshape(B, T_q, N * T_t)
        logits = logits / max(self.fine_temp, 1e-6)

        # positives: 对于第 b 个 query，正样本是第 b 个 target 的所有 token
        pos_mask = F.one_hot(
            torch.arange(B, device=score_q2t.device), num_classes=N
        ).unsqueeze(-1).expand(B, N, T_t).reshape(B, N * T_t).bool()   # [B, N*T_t]

        pos_logits = logits.masked_fill(~pos_mask.unsqueeze(1), -1e4)

        numerator = torch.logsumexp(pos_logits, dim=-1)   # [B, T_q]
        denominator = torch.logsumexp(logits, dim=-1)     # [B, T_q]

        loss_tok = -(numerator - denominator)             # [B, T_q]

        token_weights = F.softmax(
            -entropy_q.detach() / self.fine_token_weight_temp, dim=-1
        )  # [B, T_q]

        loss_fine = (loss_tok * token_weights).sum(dim=-1).mean()
        return loss_fine

    # ---------- forward ----------
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

            # ===== query branch =====
            raw_text_emb, raw_fusion_tokens, query_aux, fusion_output = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=text_tokens.input_ids,
                text_attention_mask=text_tokens.attention_mask,
            )

            k_t = query_aux["k_t"]          # [B,1]
            k_r = query_aux["k_r"]          # [B,1]
            k_local = query_aux["k_local"]  # [B,1]
            k_m = query_aux["k_m"]          # [B,1]
            g = query_aux["g"]
            topk_attn = query_aux["topk_attn"]

            # ===== target branch =====
            target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=target_embeds,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True
            )

            target_feats = F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1)   # [B,32,D]
            k_target_tokens = self.kappa_v_predictor(target_output.last_hidden_state).squeeze(-1)    # [B,32]

            # ===== query token components =====
            u_query_tokens = F.normalize(self.text_proj(raw_fusion_tokens), dim=-1)   # [B,32,D]
            k_query_tokens_raw = self.kappa_predictor(raw_fusion_tokens).squeeze(-1)   # [B,32]

            # ===== dynamic fusion =====
            if self.use_dynamic_fusion:
                k_query_final, w_q, w_t, w_r, w_m = self._dynamic_fuse_kappa(
                    k_q_raw=k_query_tokens_raw,
                    k_m=k_m,
                    k_t=k_t,
                    k_r=k_r,
                )
            else:
                k_query_final = k_query_tokens_raw
                w_q = torch.ones_like(k_query_final)
                w_t = torch.zeros_like(k_query_final)
                w_r = torch.zeros_like(k_query_final)
                w_m = torch.zeros_like(k_query_final)

            # ===== holistic retrieval loss =====
            base_loss, logits_global, score_global, entropy_q, score_q2t = self.token_vmf_loss(
                query_tokens=u_query_tokens,
                kappa_q_tokens=k_query_final,
                target_tokens=target_feats,
                kappa_t_tokens=k_target_tokens,
            )

            # ===== wrong-text branch for coordination uncertainty =====
            wrong_text_ids = torch.roll(text_tokens.input_ids, shifts=1, dims=0)
            wrong_text_mask = torch.roll(text_tokens.attention_mask, shifts=1, dims=0)

            _, raw_wrong_fusion_tokens, wrong_aux, _ = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=wrong_text_ids,
                text_attention_mask=wrong_text_mask,
            )

            k_m_wrong = wrong_aux["k_m"]   # [B,1]

            if self.use_coord_branch:
                # matched coordination confidence > wrong coordination confidence
                loss_cord = F.softplus(-(k_m - k_m_wrong)).mean()
            else:
                loss_cord = torch.zeros([], device=image.device, dtype=u_query_tokens.dtype)

            # ===== text-only / rtc branch =====
            prompt_tokens = self.prompt_tokens.expand(image_embeds.shape[0], -1, -1)
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

            text_only_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=prompt_tokens,
                attention_mask=attention_mask,
                return_dict=True,
                no_img=True
            )

            raw_text_only_tokens = text_only_output.last_hidden_state[:, : query_tokens.size(1), :]
            u_text_only_tokens = F.normalize(self.text_proj(raw_text_only_tokens), dim=-1)
            k_text_only = self.kappa_predictor(raw_text_only_tokens).squeeze(-1)

            loss_rtc, _, _, _, _ = self.token_vmf_loss(
                query_tokens=u_text_only_tokens,
                kappa_q_tokens=k_text_only,
                target_tokens=target_feats,
                kappa_t_tokens=k_target_tokens,
            )

            # ===== prompt-anchor align =====
            loss_align = F.mse_loss(
                fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
                prompt_tokens.detach()
            )

            # ===== fine-grained vMF contrast =====
            if self.use_fine_grained_loss:
                loss_fine = self._compute_fine_grained_vmf_loss(
                    score_q2t=score_q2t,
                    entropy_q=entropy_q
                )
            else:
                loss_fine = torch.zeros([], device=image.device, dtype=u_query_tokens.dtype)

            # ===== total loss =====
            total_loss = (
                base_loss
                + self.lambda_fine * loss_fine
                + self.lambda_cord * loss_cord
                + self.lambda_rtc * loss_rtc
                + self.lambda_align * loss_align
            )

        return {
            'loss': total_loss,
            'loss_base': base_loss.item(),
            'loss_fine': loss_fine.item(),
            'loss_cord': loss_cord.item(),
            'loss_align': loss_align.item(),
            'loss_rtc': loss_rtc.item(),

            # compatibility
            'loss_nll_q': 0.0,
            'loss_nll_t': 0.0,
            'loss_rank': 0.0,

            'k_q_raw_mean': k_query_tokens_raw.mean().item(),
            'k_q_raw_std': k_query_tokens_raw.std().item(),
            'k_q_raw_min': k_query_tokens_raw.min().item(),
            'k_q_raw_max': k_query_tokens_raw.max().item(),

            'k_q_final_mean': k_query_final.mean().item(),
            'k_t_std': k_t.std().item(),
            'k_v_std': k_r.std().item(),

            'w_q_mean': w_q.mean().item(),
            'w_r_mean': w_r.mean().item(),
            'w_t_mean': w_t.mean().item(),
            'w_m_mean': w_m.mean().item(),

            'k_local_mean': k_local.mean().item(),
            'k_m_mean': k_m.mean().item(),
            'k_m_wrong_mean': k_m_wrong.mean().item(),

            'g_mean': g.mean().item(),
            'topk_attn_mean': topk_attn.mean().item(),

            'entropy_q_mean': entropy_q.mean().item(),
            'entropy_q_std': entropy_q.std().item(),
        }

    @torch.no_grad()
    def forward_debug(self, samples):
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

            _, raw_fusion_tokens, query_aux, _ = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=text_tokens.input_ids,
                text_attention_mask=text_tokens.attention_mask,
            )

            k_t = query_aux["k_t"]
            k_r = query_aux["k_r"]
            k_m = query_aux["k_m"]

            target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=target_embeds,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True
            )
            target_feats = F.normalize(self.vision_proj(target_output.last_hidden_state), dim=-1)
            k_target_tokens = self.kappa_v_predictor(target_output.last_hidden_state).squeeze(-1)

            u_query_tokens = F.normalize(self.text_proj(raw_fusion_tokens), dim=-1)
            k_query_tokens_raw = self.kappa_predictor(raw_fusion_tokens).squeeze(-1)

            if self.use_dynamic_fusion:
                k_query_final, _, _, _, _ = self._dynamic_fuse_kappa(
                    k_q_raw=k_query_tokens_raw,
                    k_m=k_m,
                    k_t=k_t,
                    k_r=k_r,
                )
            else:
                k_query_final = k_query_tokens_raw

            score_q2t, entropy_q = self.token_vmf_loss.compute_score(
                query_tokens=u_query_tokens,
                kappa_q_tokens=k_query_final,
                target_tokens=target_feats,
                kappa_t_tokens=k_target_tokens,
            )

            max_score_i2t, _ = score_q2t.max(dim=2)   # [B,N,T_q]
            token_weights = F.softmax(-entropy_q / self.token_vmf_loss.token_weight_temp, dim=-1)
            sim_i2t = (max_score_i2t * token_weights.unsqueeze(1)).sum(dim=-1)

            pos_sim = sim_i2t.diag()
            eye_mask = torch.eye(sim_i2t.size(0), device=sim_i2t.device).bool()
            hardest_neg = sim_i2t.masked_fill(eye_mask, -1e4).max(dim=1)[0]
            margin = pos_sim - hardest_neg

        return {
            "query_feat": u_query_tokens,
            "k_q_raw": k_query_tokens_raw.mean(dim=-1).float().cpu(),
            "k_q_final": k_query_final.mean(dim=-1).float().cpu(),
            "k_t": k_t.squeeze(-1).float().cpu(),
            "k_v": k_r.squeeze(-1).float().cpu(),
            "k_m": k_m.squeeze(-1).float().cpu(),
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

            _, raw_fusion_tokens, query_aux, _ = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=text_tokens.input_ids,
                text_attention_mask=text_tokens.attention_mask,
            )

            u_query_tokens = F.normalize(self.text_proj(raw_fusion_tokens), dim=-1)
            k_query_tokens_raw = self.kappa_predictor(raw_fusion_tokens).squeeze(-1)

            if self.use_dynamic_fusion:
                k_query_final, _, _, _, _ = self._dynamic_fuse_kappa(
                    k_q_raw=k_query_tokens_raw,
                    k_m=query_aux["k_m"],
                    k_t=query_aux["k_t"],
                    k_r=query_aux["k_r"],
                )
            else:
                k_query_final = k_query_tokens_raw

            return u_query_tokens, k_query_final

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

            _, raw_fusion_tokens, query_aux, _ = self._build_query_from_image_text(
                image_embeds=image_embeds,
                image_atts=image_atts,
                query_tokens=query_tokens,
                query_atts=query_atts,
                text_input_ids=text_tokens.input_ids,
                text_attention_mask=text_tokens.attention_mask,
            )

            k_t = query_aux["k_t"]
            k_r = query_aux["k_r"]
            k_m = query_aux["k_m"]

            u_query_tokens = F.normalize(self.text_proj(raw_fusion_tokens), dim=-1)
            k_q_raw = self.kappa_predictor(raw_fusion_tokens).squeeze(-1)

            if self.use_dynamic_fusion:
                k_q_final, _, _, _, _ = self._dynamic_fuse_kappa(
                    k_q_raw=k_q_raw,
                    k_m=k_m,
                    k_t=k_t,
                    k_r=k_r,
                )
            else:
                k_q_final = k_q_raw

        return {
            "query_feat": u_query_tokens,
            "k_q_raw": k_q_raw.mean(dim=-1),
            "k_q_final_mean": k_q_final.mean(dim=-1),  # 用于统计相关性
            "k_q_final_tokens": k_q_final,             # 🔥 新增：原汁原味的 [B, 32] 维度！用于推理！
            "k_t": k_t.squeeze(-1),
            "k_v": k_r.squeeze(-1),
            "k_m": k_m.squeeze(-1),
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

            target_features = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)
            k_target_tokens = self.kappa_v_predictor(query_output.last_hidden_state).squeeze(-1)

        return target_features, k_target_tokens

    @torch.no_grad()
    def inference(self, query_feat, k_query, target_feats, k_targets):
        """
        Query 和 Target 均为序列表示
        - token-pair 度量：标准单向 KL(q || t)
        - target token 维：MaxSim
        - query token 聚合：熵加权
        """
        score_q2t, entropy_q = self.token_vmf_loss.compute_score(
            query_tokens=query_feat,
            kappa_q_tokens=k_query,
            target_tokens=target_feats,
            kappa_t_tokens=k_targets,
        )

        max_score_i2t, _ = score_q2t.max(dim=2)   # [B,N,T_q]
        token_weights = F.softmax(-entropy_q / self.token_vmf_loss.token_weight_temp, dim=-1)

        similarity_scores = (max_score_i2t * token_weights.unsqueeze(1)).sum(dim=-1)
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
            embed_dim=cfg.get("embed_dim", 256),
            max_txt_len=cfg.get("max_txt_len", 32),

            use_coord_branch=cfg.get("use_coord_branch", True),
            use_dynamic_fusion=cfg.get("use_dynamic_fusion", True),
            use_fine_grained_loss=cfg.get("use_fine_grained_loss", True),

            fuse_coord_unc=cfg.get("fuse_coord_unc", True),
            fuse_text_unc=cfg.get("fuse_text_unc", True),
            fuse_image_unc=cfg.get("fuse_image_unc", True),

            lambda_cord=cfg.get("lambda_cord", 0.05),
            lambda_fine=cfg.get("lambda_fine", 0.20),
            lambda_rtc=cfg.get("lambda_rtc", 0.20),
            lambda_align=cfg.get("lambda_align", 0.10),

            fine_temp=cfg.get("fine_temp", 0.07),
            fine_token_weight_temp=cfg.get("fine_token_weight_temp", 0.5),
        )
        model.load_checkpoint_from_config(cfg)
        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        k_test = task_cfg.k_test
        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)