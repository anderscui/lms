# coding=utf-8
import torch
import torch.nn as nn


class ElmanRnnUnit(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.Uh = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.Wh = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.b = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x, h):
        # x and h both have shape: (batch_size, embedding_dim)
        return torch.tanh(x @ self.Wh + h @ self.Uh + self.b)


class ElmanRnn(nn.Module):
    def __init__(self, embedding_dim, num_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        # train 之后得到的：每一个 layer 有一组 W、U、b
        self.rnn_units = nn.ModuleList([ElmanRnnUnit(embedding_dim) for _ in range(num_layers)])

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        # 每一 layer 一个 prev hidden state, init state 为 0.
        h_prev = [torch.zeros(batch_size, embedding_dim, device=x.device)
                  for _ in range(self.num_layers)]
        outputs = []
        for t in range(seq_len):
            # 遍历 seq 的每一个位置, layer 1 的 input_t 为当前位置的 x_t
            input_t = x[:, t]
            for l, rnn_unit in enumerate(self.rnn_units):
                # 在 layer 1，用 x_t 和 h_prev 计算当前隐藏状态，该状态会传到 layer 1 的下一个位置，也传到 layer 2
                # 在 layer 2，input_t 是 上一层的 hidden state，包含了上一层迄今整个序列的信息。
                # 在更上面的层以此类推
                h_new = rnn_unit(input_t, h_prev[l])
                h_prev[l] = h_new
                input_t = h_new
            # 此时的 input_t 是最上面 layer 的 hidden state
            outputs.append(input_t)

        # result shape: (batch_size, seq_len, embedding_dim)
        return torch.stack(outputs, dim=1)


class RecurrentLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = ElmanRnn(embedding_dim, num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len): seq -> seq if token ids
        embeddings = self.embedding(x)
        # embeddings: (batch_size, seq_len, embedding_dim)
        rnn_output = self.rnn(embeddings)
        # rnn_output: (batch_size, seq_len, embedding_dim)
        logits = self.fc(rnn_output)
        # output logits: (batch_size, seq_len, vocab_size)
        return logits


