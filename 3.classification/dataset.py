
from sklearn.datasets import make_circles, make_blobs, make_moons  # Для створення датасету
import matplotlib.pyplot as plt  # Для побудови графіків
import torch  # Для роботи з PyTorch
from sklearn.model_selection import train_test_split  # Для розподілу даних на тренувальний та тестовий набори
from device import device

# Генерація даних з кругами
n_samples = 1000  # Кількість зразків (точок) у датасеті
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)  # Створення датасету: X - ознаки, y - мітки класів

# Коментар: Для візуалізації даних (якщо потрібно)
# plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)  # Візуалізація точок з класами (розфарбовані)
# plt.xlabel("Feature 1")  # Підпис осі X
# plt.ylabel("Feature 2")  # Підпис осі Y
# plt.title("Circles dataset")  # Назва графіка
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


#-----------------------------------------------------------------------------------
# Параметри
NUM_CLASSES = 4  # Кількість класів (кластерів) для генерації даних
NUM_FEATURES = 2  # Кількість ознак (координат) у кожній точці
RANDOM_SEED = 42  # Фіксований генератор випадкових чисел для відтворюваності результатів

# Генерація даних за допомогою make_blobs
# n_samples=1000: загальна кількість точок
# n_features=NUM_FEATURES: двовимірні координати точок
# centers=NUM_CLASSES: кількість центрів кластерів
# cluster_std=1.2: дисперсія точок навколо центрів
X_blob, y_blob = make_blobs(
    n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES,
    random_state=RANDOM_SEED, cluster_std=1.2
)

# Перетворення даних із NumPy в тензори PyTorch
X_blob = torch.from_numpy(X_blob).type(torch.float)  # Координати точок
y_blob = torch.from_numpy(y_blob).type(torch.long)  # Мітки класів (цільові значення)

# Розділення даних на тренувальну і тестову вибірки
# test_size=0.2: 20% даних будуть використані для тестування
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED
)

# Код для візуалізації даних (закоментований)
# Відображення точок із кольорами, що відповідають класам
# plt.figure(figsize=(10,7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

#-------------------------------------------------------------------------

# Генерація 2D даних у вигляді двох півмісяців
X, y = make_moons(n_samples, random_state=RANDOM_SEED, noise=0.03)


# Перетворення даних із NumPy в тензори PyTorch
X_moons = torch.from_numpy(X).type(torch.float)  # Координати точок
y_moons = torch.from_numpy(y).type(torch.long)  # Мітки класів (цільові значення)

# Розділення даних на тренувальну і тестову вибірки
# test_size=0.2: 20% даних будуть використані для тестування
X_moons_train, X_moons_test, y_moons_train, y_moons_test = train_test_split(
    X_moons, y_moons, test_size=0.2, random_state=RANDOM_SEED
)


plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y, cmap='viridis')
plt.title("Moons Dataset")
plt.show()

