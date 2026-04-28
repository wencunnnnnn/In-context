"""
SGCAFE: Support-Guided Cross-Attention Feature Enhancement

在 CLIP 特征空间中，用 cross-attention + mask bias 显式计算
support→query 的空间对应关系，输出增强后的 query 特征。
LLM 不再看到 support 图，只看到增强后的 576 个 query token。
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGCAFEModule(nn.Module):
    """Support-Guided Cross-Attention Feature Enhancement."""

    def __init__(self, clip_dim=1024, inner_dim=256, num_heads=4, mask_bias_alpha=5.0, support_dropout=0.3):
        super().__init__()
        assert inner_dim % num_heads == 0

        self.clip_dim = clip_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self._trained = False  # 推理时用于判断是否跳过未训练的 SGCAFE
        self.head_dim = inner_dim // num_heads
        self.mask_bias_alpha = mask_bias_alpha
        self.support_dropout = support_dropout

        # Layer norms
        self.layer_norm_q = nn.LayerNorm(clip_dim)
        self.layer_norm_s = nn.LayerNorm(clip_dim)

        # Cross-attention projections
        self.q_proj = nn.Linear(clip_dim, inner_dim)
        self.k_proj = nn.Linear(clip_dim, inner_dim)
        self.v_proj = nn.Linear(clip_dim, inner_dim)
        self.o_proj = nn.Linear(inner_dim, clip_dim)

        # Gated residual
        self.gate_proj = nn.Linear(clip_dim, clip_dim)

        # Auxiliary weak segmentation head (训练时用，推理时丢弃)
        self.aux_head = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # gate 初始化: sigmoid(-1)≈0.27，让 SGCAFE 起步就有足够信号通过
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -1.0)
        # aux_head 默认初始化即可

    def forward(self, query_features, support_features, support_mask_weights):
        """
        Args:
            query_features:       (B, N, clip_dim) - CLIP features of query image
            support_features:     (B, N, clip_dim) - CLIP features of support image
            support_mask_weights: (B, N)           - [0,1] mask at CLIP spatial resolution

        Returns:
            enhanced:        (B, N, clip_dim) - enhanced query features
            attn:            (B, num_heads, N, N) - attention map
            support_dropped: bool - 是否触发了 support dropout
        """
        B, N, C = query_features.shape
        orig_dtype = query_features.dtype

        # --- Support Dropout (training trick) ---
        support_dropped = False
        if self.training and random.random() < self.support_dropout:
            support_features = torch.zeros_like(support_features)
            support_mask_weights = torch.zeros_like(support_mask_weights)
            support_dropped = True

        # 全程 float32 计算，避免 bf16 下 CLIP 大范围值导致 NaN
        query_f32 = query_features.float()
        support_f32 = support_features.float()

        # Layer norm - 手动用 F.layer_norm 绕过子模块的 hooks
        q = F.layer_norm(query_f32, [C],
                         self.layer_norm_q.weight.float(),
                         self.layer_norm_q.bias.float(),
                         self.layer_norm_q.eps)
        s = F.layer_norm(support_f32, [C],
                         self.layer_norm_s.weight.float(),
                         self.layer_norm_s.bias.float(),
                         self.layer_norm_s.eps)

        # Project to inner_dim - 手动用 F.linear 绕过子模块
        Q = F.linear(q, self.q_proj.weight.float(), self.q_proj.bias.float())
        K = F.linear(s, self.k_proj.weight.float(), self.k_proj.bias.float())
        V = F.linear(s, self.v_proj.weight.float(), self.v_proj.bias.float())

        # Reshape for multi-head: (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention logits: (B, num_heads, N_q, N_s)
        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Mask bias: 强制关注 support 中 mask 区域
        mask_bias = self.mask_bias_alpha * support_mask_weights.float()
        attn_logits = attn_logits + mask_bias.unsqueeze(1).unsqueeze(2)

        attn = F.softmax(attn_logits, dim=-1, dtype=torch.float32)

        # Weighted sum: (B, num_heads, N, head_dim)
        context = torch.matmul(attn, V)

        # Reshape back: (B, N, inner_dim)
        context = context.transpose(1, 2).contiguous().view(B, N, self.inner_dim)

        # Output projection - 手动 F.linear
        cross_attn_output = F.linear(context, self.o_proj.weight.float(), self.o_proj.bias.float())

        # Gated residual connection
        gate = torch.sigmoid(F.linear(query_f32, self.gate_proj.weight.float(), self.gate_proj.bias.float()))
        enhanced = query_f32 + gate * cross_attn_output

        # 转回原始 dtype
        enhanced = enhanced.to(orig_dtype)
        attn = attn.to(orig_dtype)

        return enhanced, attn, support_dropped

    def aux_forward(self, enhanced_features):
        """
        弱分割头：将增强特征映射为粗略 mask 预测。
        仅训练时调用。

        Args:
            enhanced_features: (B, N, clip_dim) - SGCAFE 增强后的特征

        Returns:
            aux_logits: (B, 24, 24) - 粗略 mask logits（未经 sigmoid）
        """
        B, N, C = enhanced_features.shape
        # 转 float32 计算，与 aux_head 权重 dtype 一致
        logits = self.aux_head(enhanced_features.float()).squeeze(-1)  # (B, N)
        # N 应该是 576 = 24*24
        H = W = int(math.sqrt(N))
        return logits.view(B, H, W)
