import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Задаємо ВІДОМЕ значення для ваги (weight) та зсуву (bias)
weight = 0.7   # Вага, яку модель буде навчатися знаходити
bias = 0.3     # Зсув, який модель також буде навчатися визначати

# Визначаємо межі для створення нашого масиву вхідних даних
start = 0      # Початкова точка (0)
end = 1        # Кінцева точка (1) - не включно
step = 0.02    # Крок для створення значень, що буде на кожному кроці

# Створюємо масив X з значень від 0 до 1 з кроком 0.02
# `unsqueeze(dim=1)` додає новий вимір, щоб отримати масив стовпців (для сумісності з моделями)
X = torch.arange(start, end, step).unsqueeze(dim=1)  # Це наш вхідний масив
y = weight * X + bias   # Вихідні дані обчислюються за формулою y = w * x + b

# Визначаємо індекс, де розділяємо дані на тренувальну та тестову частину
train_split = int(0.8 * len(X))  # 80% від загальної кількості даних підуть на тренування


# Розділяємо дані на тренувальну та тестову частину
# X_train і y_train — це тренувальні дані
X_train, y_train = X[:train_split], y[:train_split]  # Вибираємо перші 80% для тренування

# X_test і y_test — це тестові дані
X_test, y_test = X[train_split:], y[train_split:]  # Останні 20% — для тестування моделі

# Описуємо функцію для візуалізації передбачень
def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    # Створюємо нову фігуру для побудови графіка
    plt.figure(figsize=(10, 7))  # Визначаємо розміри графіка (ширина: 10, висота: 7 дюймів)

    # Побудова графіка для тренувальних даних
    # Всі точки тренувальних даних будуть синього кольору ('b'), розмір точок 4, а підпис 'Training Data'
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

    # Побудова графіка для тестових даних
    # Точки тестових даних будуть зеленого кольору ('g'), розмір точок 4, а підпис 'Testing Data'
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    # Якщо є передбачення, додаємо їх на графік
    # Червоні точки ('r') будуть для передбачених значень, розмір точок 4, підпис 'Predictions'
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Додаємо легенду для розрізнення різних точок на графіку
    plt.legend(prop={"size": 14})  # Розмір шрифта для легенди

    # Відображаємо графік
    plt.show()  # Виводимо графік на екран


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

        def forward(self,x: torch.Tensor) -> torch.Tensor:
            return self.weight * x + self.bias # формула лінійної регресії