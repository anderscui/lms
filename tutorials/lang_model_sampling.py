# coding=utf-8
import numpy as np


def sample_token(logits, vocab, temperature=0.7, top_k=50):
    if len(logits) != len(vocab):
        raise ValueError('Mismatch between the logits and vocab sizes.')

    if temperature <= 0:
        raise ValueError('Temperature must be positive.')
    if top_k < 1:
        raise ValueError('top_k must be at least 1.')
    if top_k > len(logits):
        raise ValueError('top_k must be at most len(logits).')

    # 添加 temperature
    logits = logits / temperature
    cutoff = np.sort(logits)[-top_k]
    # 仅保留 top k 个 logits（如果有 tie 的情况，此方法会有k+1的可能 tokens）
    # np.ext(-inf) = 0.0
    logits[logits < cutoff] = float('-inf')

    # 所有 logits 都变为非正值，避免出现 overflow 的情况
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()

    return np.random.choice(vocab, p=probs)
