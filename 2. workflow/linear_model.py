import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Ініціалізуємо параметр "weights" (вага) випадковим значенням
        # nn.Parameter дозволяє PyTorch відслідковувати цей параметр та оптимізувати його під час тренування
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        # - torch.randn(1) генерує випадкове значення з нормального розподілу
        # - requires_grad=True означає, що PyTorch має обчислювати градієнт цього параметра під час тренування
        # - dtype=torch.float вказує на тип даних 

        # Ініціалізуємо параметр "bias" (зсув) випадковим значенням
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        # - Аналогічно до self.weights, цей параметр має бути оптимізований під час тренування
        # - Використовуємо torch.randn(1) для випадкової ініціалізації
        # - Параметр має requires_grad=True для обчислення градієнта


    def forward(self,x: torch.Tensor) -> torch.Tensor: #функції форвард пропагації
            return self.weight * x + self.bias # формула лінійної регресії
    