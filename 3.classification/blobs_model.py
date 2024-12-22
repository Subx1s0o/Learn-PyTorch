import torch
from torch import nn
from dataset import X_blob_test, X_blob_train, y_blob_test, y_blob_train, accuracy_fn
from device import device
from matplotlib import pyplot as plt
from helper_function import plot_decision_boundary

# === Клас моделі для класифікації ===
class BlobModel(nn.Module):
    def __init__(self, input_features, out_features, hidden_units=8):
        super().__init__()
        # Побудова моделі за допомогою Sequential
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),  # Перший лінійний шар
            nn.ReLU(),  # Активаційна функція ReLU після першого шару
            nn.Linear(in_features=hidden_units, out_features=hidden_units),  # Другий лінійний шар
            nn.ReLU(),  # ReLU після другого шару
            nn.Linear(in_features=hidden_units, out_features=out_features)  # Вихідний лінійний шар
        )
    
    def forward(self, x):
        # Передача даних через всі шари моделі
        return self.linear_layer_stack(x)

# Ініціалізація моделі
model_blob = BlobModel(input_features=2, out_features=4).to(device)  # Модель приймає 2 ознаки і видає 4 класи

# === Налаштування тренування ===
loss_fn = nn.CrossEntropyLoss()  # Функція втрат для багатокласової класифікації
optim = torch.optim.Adam(model_blob.parameters(), lr=0.1)  # Оптимізатор Adam з початковою швидкістю навчання

# Ініціалізація генератора випадкових чисел для повторюваності результатів
torch.manual_seed(42)
torch.mps.manual_seed(42)

# Кількість епох тренування
epochs = 1000

# Переносимо дані на пристрій
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

# === Цикл тренування ===
for epoch in range(epochs):
    model_blob.train()  # Встановлюємо модель в режим тренування

    # Прогноз для тренувальних даних
    y_logits = model_blob(X_blob_train)  # Логіти (сирий вихід перед softmax)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)  # Прогноз після softmax

    # Розрахунок функції втрат
    loss = loss_fn(y_logits, y_blob_train)

    # Оцінка точності для тренувальних даних
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_preds)

    # Оновлення ваг
    optim.zero_grad()  # Обнулення попередніх градієнтів
    loss.backward()  # Зворотне поширення помилки
    optim.step()  # Оновлення параметрів моделі

    # Оцінка моделі на тестових даних
    model_blob.eval()  # Встановлюємо модель у режим оцінки
    with torch.inference_mode():  # Вимикаємо обчислення градієнтів для тесту
        test_logits = model_blob(X_blob_test)  # Логіти для тестових даних
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)  # Прогноз для тесту

        # Розрахунок функції втрат і точності для тестових даних
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_blob_test, test_pred)

    # Вивід метрик кожні 10 епох
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

# === Візуалізація межі рішення ===
model_blob.eval()  # Модель у режимі оцінки
with torch.inference_mode():
    y_logits = model_blob(X_blob_test)  # Логіти для візуалізації

# Візуалізація межі рішення
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")  # Межа рішення для тренувальних даних
plot_decision_boundary(model_blob, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")  # Межа рішення для тестових даних
plot_decision_boundary(model_blob, X_blob_test, y_blob_test)
plt.show()
