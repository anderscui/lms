# coding=utf-8
import math

import torch
import torch.nn as nn


def rope(v):
    return v


class AttentionHead(nn.Module):
    def __init__(self, emb_dim, d_h):
        super().__init__()
        self.W_Q = nn.Parameter(torch.empty(emb_dim, d_h))
        self.W_K = nn.Parameter(torch.empty(emb_dim, d_h))
        self.W_V = nn.Parameter(torch.empty(emb_dim, d_h))
        self.d_h = d_h

    def forward(self, x, mask):
        # x: (batch_size, seq_len, emb_dim)
        # Q, K, V: (batch_size, seq_len, d_h)
        # d_h: dim of q, k, v
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # apply rotary positional encoding
        Q, K = rope(Q), rope(K)

        # K.transpose(-2, -1):  (batch_size, d_h, seq_len)
        # Q @ K.transpose(-2, -1): (batch_size, seq_len, seq_len)：每一个 query 与 每一个 k 的点积
        # sqrt: for numerical stability
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)
        # causal mask
        masked_scores = scores.masked_fill(mask==0, float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)
        # output: (batch_size, seq_len, d_h)
        # head 的输出本应为 (batch_size, seq_len, emb_dim)，但此处为 multi-head，故 d_h 为其中一个的维度，应为 emb_dim 的因子
        return attention_weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        # 此处相当于对单个 head 分解，故应为因子
        d_h = emb_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(emb_dim, d_h) for _ in range(num_heads)])
        # projection matrix：(emb_dim, emb_dim)，将 multi-head 的权重组合起来
        self.W_O = nn.Parameter(torch.empty(emb_dim, emb_dim))

    def forward(self, x, mask):
        head_outputs = [head(x, mask) for head in self.heads]
        # 拼接所有 head 的数值：(batch_size, seq_len, emb_dim)
        x = torch.concat(head_outputs, dim=-1)
        # project：此时维度恢复为 (batch_size, seq_len, emb_dim)
        return x @ self.W_O


class MLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(emb_dim, emb_dim * 4))
        self.B_1 = nn.Parameter(torch.empty(emb_dim * 4))

        self.W_2 = nn.Parameter(torch.empty(emb_dim * 4, emb_dim))
        self.B_2 = nn.Parameter(torch.empty(emb_dim))

    def forward(self, x):
        # after this: (batch_size, seq_len, emb_dim * 4)
        x = x @ self.W_1 + self.B_1
        x = torch.relu(x)
        # after this：(batch_size, seq_len, emb_dim)
        x = x @ self.W_2 + self.B_2
        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm2 = RMSNorm(emb_dim)
        self.mlp = MLP(emb_dim)

    def forward(self, x, mask):
        # 计算 attention weights 先将输入 RMSNorm 1
        attn_out = self.attn(self.norm1(x), mask)
        # Residual connection 1
        x = x + attn_out

        # 计算 mlp 先将输入 RMSNorm 2
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x

class DecoderLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, num_blocks, pad_idx):
        super().__init__()
        # padding tokens 会被 map 到零向量
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        # 一组 decoder blocks
        self.layers = nn.ModuleList([DecoderBlock(emb_dim, num_heads) for _ in range(num_blocks)])
        # 从 block 的output 到词汇的 logits
        self.output = nn.Parameter(torch.rand(emb_dim, vocab_size))

    def forward(self, x):
        # after this：(batch_size, seq_len, emb_dim)
        x = self.embedding(x)
        _, seq_len, _ = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        for layer in self.layers:
            x = layer(x, mask)

        # final output: (batch_size, seq_len, vocab_size)
        return x @ self.output
