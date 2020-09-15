import torch
import torch.nn as nn
import numpy as np


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.pow((x - y), 2))



'''
x_train = np.array([[3.0], [4.0], [6.0]], dtype=np.float32)
y_train = np.array([[1.0], [1.0], [3.0]], dtype=np.float32)

inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)

criterion = My_loss()
loss = criterion(inputs,targets)
print(loss)
'''





