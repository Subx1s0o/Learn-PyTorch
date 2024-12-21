import torch
from torch import nn
from dataset import X_test, X_train, y_test, y_train  # Імпорт даних для тренування та тестування
import requests
from pathlib import Path  # Для роботи з шляхами до файлів
from matplotlib import pyplot as plt  # Для візуалізації результатів

# Визначення пристрою (GPU або CPU)
device = "mps" if torch.mps.is_available() else "cpu"  # Якщо є підтримка Metal API для GPU на Mac, то використовуємо GPU, інакше CPU

# Опис моделі (нейронна мережа з двома шарами)
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),  # Перший шар з 2 вхідними та 5 вихідними нейронами
    nn.Linear(in_features=5, out_features=1)   # Другий шар з 5 вхідними та 1 вихідним нейроном
).to(device)  # Переміщення моделі на вибраний пристрій (GPU або CPU)

# Функція для обчислення точності
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # Підрахунок правильних прогнозів
    acc = (correct / len(y_pred)) * 100  # Обчислення точності у відсотках
    return acc

# Встановлюємо фіксовані значення для генераторів випадкових чисел (для відтворюваності)
torch.manual_seed(42)
torch.mps.manual_seed(42)

# Кількість епох для тренування
epochs = 1000

#Ініціалізуємо оптимайзер Градієнтний спуск
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Визначаємо головну функцію обчислення лосу
loss_fn = nn.BCEWithLogitsLoss()

# Переміщуємо дані на вибраний пристрій
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Основний цикл тренування
for epoch in range(epochs):

    # Тренувальний етап
    model_0.train()  # Переводимо модель в режим тренування

    # Передбачення для тренувальних даних
    y_logits = model_0(X_train).squeeze()  # Отримуємо логіти (неперетворені виходи)
    y_pred = torch.round(torch.sigmoid(y_logits))  # Перетворюємо логіти в бінарні передбачення

    # Обчислюємо функцію втрат
    loss = loss_fn(y_logits, y_train)  # Обчислюємо лосс для тренувальних даних
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)  # Обчислюємо точність

    # Оновлюємо параметри моделі (backpropagation)
    optimizer.zero_grad()  # Очищаємо старі градієнти
    loss.backward()  # Обчислюємо нові градієнти
    optimizer.step()  # Оновлюємо параметри моделі

    # Тестовий етап
    model_0.eval()  # Переводимо модель в режим оцінки (eval)

    with torch.inference_mode():  # Вимикаємо обчислення градієнтів для тестування
        test_logits = model_0(X_test).squeeze()  # Отримуємо логіти для тестових даних
        test_pred = torch.round(torch.sigmoid(test_logits))  # Перетворюємо логіти в бінарні передбачення

        test_loss = loss_fn(test_logits, y_test)  # Обчислюємо лосс для тестових даних
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)  # Обчислюємо точність для тестових даних

    # Виводимо метрики кожні 10 епох
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

# Завантаження допоміжного файлу, якщо він ще не існує
if Path("helper_function.py").is_file():  # Перевіряємо, чи існує файл
    print("helper function is already exist")  # Якщо файл існує
else:
    print("Download helper function...")  # Якщо файлу немає, завантажуємо
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")  # Завантажуємо файл
    with open("helper_function.py", "wb") as f:  # Записуємо файл на диск
        f.write(request.content)

# Імпортуємо функцію з допоміжного файлу для візуалізації
from helper_function import  plot_decision_boundary

# Візуалізація межі рішення для тренувальних та тестових даних
plt.figure(figsize=(12, 6))  # Встановлюємо розмір графіка

# Графік для тренувальних даних
plt.subplot(1, 2, 1)  # Розташування на графіку (1 рядок, 2 стовпці, перший)
plt.title("Train")  # Заголовок для графіка
plot_decision_boundary(model=model_0, X=X_train, y=y_train)  # Будуємо межу рішення для тренувальних даних

# Графік для тестових даних
plt.subplot(1, 2, 2)  # Розташування на графіку (1 рядок, 2 стовпці, другий)
plt.title("Test")  # Заголовок для графіка
plot_decision_boundary(model=model_0, X=X_test, y=y_test)  # Будуємо межу рішення для тестових даних

# Відображаємо графік
plt.show()
