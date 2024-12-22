import torch
from matplotlib import pyplot as plt

# 1. Створення тензора x з послідовністю чисел від -10 до 10
x = torch.arange(-10, 11, dtype=torch.float)

# 2. Реалізація сигмоїдної функції
# Сигмоїдна функція відображає будь-яке значення в діапазоні (0, 1)
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# 3. Побудова графіка сигмоїдної функції
plt.plot(x, sigmoid(x))  # Відображення значень x і результатів sigmoid(x)
plt.title("Sigmoid Function")  # Заголовок графіка
plt.xlabel("x")  # Підпис осі x
plt.ylabel("sigmoid(x)")  # Підпис осі y
plt.grid()  # Додавання сітки для кращої читабельності
plt.show()  # Відображення графіка
