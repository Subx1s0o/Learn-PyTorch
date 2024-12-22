from matplotlib import pyplot as plt
import torch

x = torch.arange(-10,11,dtype=torch.float)

plt.plot(x)
plt.show()

def relu(x):
    return torch.max(torch.zeros_like(x), x)

plt.plot(relu(x))
plt.show()