
from sklearn.datasets import make_circles  # Для створення датасету з кругами
import matplotlib.pyplot as plt  # Для побудови графіків
import torch  # Для роботи з PyTorch
from sklearn.model_selection import train_test_split  # Для розподілу даних на тренувальний та тестовий набори

# Генерація даних з кругами
n_samples = 1000  # Кількість зразків (точок) у датасеті
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)  # Створення датасету: X - ознаки, y - мітки класів

# Коментар: Для візуалізації даних (якщо потрібно)
# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)  # Візуалізація точок з класами (розфарбовані)
# plt.xlabel("Feature 1")  # Підпис осі X
# plt.ylabel("Feature 2")  # Підпис осі Y
# plt.title("Scatter plot of make_circles dataset")  # Назва графіка
# plt.show()  # Відображення графіка

# Перетворюємо дані на тензори PyTorch
X = torch.from_numpy(X).type(torch.float)  # Перетворення ознак X з NumPy в PyTorch тензор типу float
y = torch.from_numpy(y).type(torch.float)  # Перетворення міток y з NumPy в PyTorch тензор типу float

# Розподіляємо дані на тренувальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
# test_size=0.2: 20% даних йде на тестування, решта - на тренування
# random_state=42: використовуємо однаковий випадковий стан для відтворюваності результатів

# Функція для обчислення точності
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # Підрахунок правильних прогнозів
    acc = (correct / len(y_pred)) * 100  # Обчислення точності у відсотках
    return acc
