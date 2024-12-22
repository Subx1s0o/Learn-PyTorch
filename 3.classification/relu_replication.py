from matplotlib import pyplot as plt
import torch

# 1. Створення тензора x, що містить послідовність чисел від -10 до 10 з кроком 1.
x = torch.arange(-10, 11, dtype=torch.float)

# 2. Побудова першого графіка — візуалізація лінійної функції x.
plt.plot(x)  # Підготовка графіка значень x.
plt.show()   # Відображення графіка.

# 3. Реалізація функції активації ReLU (Rectified Linear Unit).
# Вона повертає 0 для від'ємних чисел і значення числа для додатних.
def relu(x):
    return torch.max(torch.zeros_like(x), x)  # torch.zeros_like(x) створює тензор нулів тієї ж форми, що й x.

# 4. Побудова другого графіка — візуалізація значень після застосування ReLU до x.
plt.plot(relu(x))  # Підготовка графіка значень після застосування ReLU.
plt.title("ReLU Function")  # Заголовок графіка
plt.show()         # Відображення графіка.
і