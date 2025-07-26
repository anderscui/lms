# coding=utf-8
import re

import torch
import torch.nn as nn


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.token2ids = vocab
        self.id2tokens = {tid: token for token, tid in vocab.items()}

        self.unk_token = '<|unk|>'
        self.endoftext_token = '<|endoftext|>'

    def tokenize(self, t: str):
        parts = re.split(r'([,.:;?_!"()\']|--|\s)', t)
        parts = [token.strip() for token in parts if token.strip()]
        return parts

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token2ids.get(token, self.token2ids[self.unk_token]) for token in tokens]

    def decode(self, ids):
        text = " ".join([self.id2tokens[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query # [6, 2]
        keys = x @ self.W_key
        values = x @ self.W_value
        # [6, 2] @ [2, 6] -> [6, 6]: 每一个 position 上的 scores 构成一行
        attn_scores = queries @ keys.T
        # normalize
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 取出 values 的加权和作为 context vector
        context_vec = attn_weights @ values
        return context_vec


class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        # [6, 2] @ [2, 6] -> [6, 6]: 每一个 position 上的 scores 构成一行
        attn_scores = queries @ keys.T
        # normalize
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 取出 values 的加权和作为 context vector
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape

        queries = self.W_query(x) # [2, 6, 2]
        keys = self.W_key(x)
        values = self.W_value(x)

        # [2, 6, 2] @ [2, 2, 6] -> [2, 6, 6]: 每个 batch 的每一个 position 上的 scores 构成一行
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill(self.mask.bool()[:seq_len, :seq_len], -torch.inf)

        # normalize
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 取出 values 的加权和作为 context vector
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            ) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), 'd_out must be divisible by num_heads'

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # [batch_size, num_tokens, d_out]
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 对于每个 position 的q、k、v，将 d_out 展开为“矩阵”
        # [batch_size, num_tokens, num_heads, head_dim]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # [batch_size, num_heads, num_tokens, head_dim]
        # 这样 transpose 之后，就跟之前的 single-head 很像了，只是多了一维
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # [batch_size, num_heads, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(2, 3)
        # masks truncated to num_tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # shape?
        attn_scores.masked_fill(mask_bool, -torch.inf)

        # [batch_size, num_heads, num_tokens, num_tokens]
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # [batch_size, num_heads, num_tokens, head_dim] ->
        # [batch_size, num_tokens, num_heads, head_dim]
        context_vec = (attn_weights @ values).transpose(1, 2)
        # [batch_size, num_tokens, d_out]
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        # [batch_size, num_tokens, d_out]
        context_vec = self.out_proj(context_vec)
        return context_vec


