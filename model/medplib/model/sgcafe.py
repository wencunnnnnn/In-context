"""
SGCAFE: Support-Guided Cross-Attention Feature Enhancement

在 CLIP 特征空间中，用 cross-attention + mask bias 显式计算
support→query 的空间对应关系，输出增强后的 query 特征。
LLM 不再看到 support 图，只看到增强后的 575 个 query token。
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

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # gate 初始化偏置为负值，使初始 gate ≈ 0，训练初期不干扰原始特征
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, query_features, support_features, support_mask_weights):
        """
        Args:
            query_features:       (B, N, clip_dim) - CLIP features of query image
            support_features:     (B, N, clip_dim) - CLIP features of support image
            support_mask_weights: (B, N)           - [0,1] mask at CLIP spatial resolution

        Returns:
            enhanced: (B, N, clip_dim) - enhanced query features
            attn:     (B, num_heads, N, N) - attention map (for optional alignment loss)
        """
        B, N, C = query_features.shape

        # --- Support Dropout (training trick) ---
        if self.training and random.random() < self.support_dropout:
            support_features = torch.zeros_like(support_features)
            support_mask_weights = torch.zeros_like(support_mask_weights)

        # Layer norm
        q = self.layer_norm_q(query_features)
        s = self.layer_norm_s(support_features)

        # Project to inner_dim
        Q = self.q_proj(q)  # (B, N, inner_dim)
        K = self.k_proj(s)
        V = self.v_proj(s)

        # Reshape for multi-head: (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention logits: (B, num_heads, N_q, N_s)
        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Mask bias: 强制关注 support 中 mask 区域
        # support_mask_weights: (B, N) -> (B, 1, 1, N)
        mask_bias = self.mask_bias_alpha * support_mask_weights.float()
        attn_logits = attn_logits + mask_bias.unsqueeze(1).unsqueeze(2)

        attn = F.softmax(attn_logits, dim=-1)

        # Weighted sum: (B, num_heads, N, head_dim)
        context = torch.matmul(attn, V)

        # Reshape back: (B, N, inner_dim)
        context = context.transpose(1, 2).contiguous().view(B, N, self.inner_dim)

        # Output projection: (B, N, clip_dim)
        cross_attn_output = self.o_proj(context)

        # Gated residual connection
        gate = torch.sigmoid(self.gate_proj(query_features))
        enhanced = query_features + gate * cross_attn_output

        return enhanced, attn
