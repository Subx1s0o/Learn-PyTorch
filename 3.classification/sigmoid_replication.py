import torch
from matplotlib import pyplot as plt

x = torch.arange(-10,11,dtype=torch.float)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

plt.plot(sigmoid(x))
plt.show()