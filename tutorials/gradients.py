# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim

inputs = torch.tensor([
    [22, 25], [25, 35], [47, 80], [52, 95], [46, 82], [56, 90],
    [23, 27], [30, 50], [40, 60], [39, 57], [53, 95], [48, 88]
], dtype=torch.float32)
print(inputs.shape)

labels = torch.tensor([
    [0], [0], [1], [1], [1], [1], [0], [1], [1], [0], [1], [1]
], dtype=torch.float32)
print(labels.shape)

n_inputs, n_outputs = inputs.shape[1], 1

model = nn.Sequential(
    nn.Linear(n_inputs, n_outputs),
    nn.Sigmoid()
)
print(model)

optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for step in range(500):
    optimizer.zero_grad()
    # 预测并计算当前 loss, forward loss
    loss = criterion(model(inputs), labels)
    # use backpropagation to calc gradient, backward pass
    loss.backward()
    optimizer.step()
