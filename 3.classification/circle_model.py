import torch
from torch import nn
from dataset import X_test, X_train, y_test, y_train, accuracy_fn  # Імпорт даних та функції для обчислення точності
import requests
from pathlib import Path  # Для роботи з шляхами до файлів
from matplotlib import pyplot as plt  # Для візуалізації результатів
from device import device  # Імпорт пристрою для обчислень (CPU або GPU)

# === Модель 1: Початкова модель для класифікації ===
# nn.Sequential дозволяє створити послідовну модель
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),  # Перший лінійний шар: перетворення з 2 вхідних ознак у 5 нейронів
    nn.Linear(in_features=5, out_features=1)   # Другий лінійний шар: перетворення з 5 нейронів у 1 вихід (ймовірність)
).to(device)  # Переміщуємо модель на вибраний пристрій (CPU/GPU)

# === Встановлення фіксованих значень для відтворюваності результатів ===
torch.manual_seed(42)  # Фіксуємо випадковий генератор для CPU
torch.mps.manual_seed(42)  # Фіксуємо випадковий генератор для GPU (якщо використовується Metal API)

# === Налаштування параметрів навчання ===
loss_fn = nn.BCEWithLogitsLoss()  # Функція втрат для бінарної класифікації
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)  # Стохастичний градієнтний спуск (SGD)

# === Перенесення даних на пристрій для обчислень ===
X_train, y_train = X_train.to(device), y_train.to(device)  # Тренувальні дані
X_test, y_test = X_test.to(device), y_test.to(device)      # Тестові дані

# === Тренування моделі 1 ===
epochs = 100  # Кількість епох (ітерацій) тренування
for epoch in range(epochs):
    # Режим навчання (вмикає обчислення градієнтів)
    model_0.train()
    
    # Прогноз для тренувальних даних
    y_logits = model_0(X_train).squeeze()  # Логіти (сировинні значення без активації)
    y_pred = torch.round(torch.sigmoid(y_logits))  # Ймовірність перетворюється у прогноз (0 або 1)

    # Обчислення втрат та точності на тренувальних даних
    loss = loss_fn(y_logits, y_train)  # Обчислення функції втрат
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)  # Точність класифікації

    # Оновлення ваг моделі
    optimizer.zero_grad()  # Очищення градієнтів
    loss.backward()  # Зворотне розповсюдження (обчислення градієнтів)
    optimizer.step()  # Оновлення параметрів моделі

    # Режим оцінки (вимикає обчислення градієнтів)
    model_0.eval()
    with torch.inference_mode():
        # Прогноз для тестових даних
        test_logits = model_0(X_test).squeeze()  # Логіти для тестових даних
        test_pred = torch.round(torch.sigmoid(test_logits))  # Ймовірність перетворюється у прогноз

        # Обчислення втрат і точності на тестових даних
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # Виведення результатів кожні 10 епох
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

# === Завантаження допоміжного файлу для візуалізації ===
# Якщо файл helper_function.py ще не завантажений, завантажити його з GitHub
if not Path("helper_function.py").is_file():
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_function.py", "wb") as f:
        f.write(request.content)

from helper_function import plot_decision_boundary  # Імпорт функції для візуалізації

# === Візуалізація межі рішення для моделі 1 ===
plt.figure(figsize=(12, 6))  # Розмір графіка
plt.subplot(1, 2, 1)  # Перший підграфік
plt.title("Train")  # Назва графіка для тренувальних даних
plot_decision_boundary(model=model_0, X=X_train, y=y_train)  # Побудова межі рішення
plt.subplot(1, 2, 2)  # Другий підграфік
plt.title("Test")  # Назва графіка для тестових даних
plot_decision_boundary(model=model_0, X=X_test, y=y_test)  # Побудова межі рішення
plt.show()  # Відображення графіка


# # === Висновки ===
# Під час навчання функція втрат трималася на рівні ~50%, що дивно, адже вона мала зменшуватись. 
# Точність теж залишалася майже незмінною. Модель створила просту пряму лінію для класифікації, 
# але цього мабуть недостатньо для роботи зі складними даними.