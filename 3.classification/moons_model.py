from torch import nn
import torch
from matplotlib import pyplot as plt
from helper_function import plot_decision_boundary
from dataset import X_moons_train, X_moons_test, y_moons_train, y_moons_test, accuracy_fn
from device import device

class MoonsModel(nn.Module):
    def __init__(self, input_features, out_features ,hidden_units=8):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),  # Перший лінійний шар
            nn.ReLU(),  # Активаційна функція ReLU після першого шару
            nn.Linear(in_features=hidden_units, out_features=hidden_units),  # Другий лінійний шар
            nn.ReLU(),  # ReLU після другого шару
            nn.Linear(in_features=hidden_units, out_features=out_features)  # Вихідний лінійний шар
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
model_moons = MoonsModel(input_features=2 , out_features=2).to(device)

# === Налаштування тренування ===
loss_fn = nn.CrossEntropyLoss()  # Функція втрат для багатокласової класифікації
optim = torch.optim.Adam(model_moons.parameters(), lr=0.1)  # Оптимізатор Adam з початковою швидкістю навчання

# Ініціалізація генератора випадкових чисел для повторюваності результатів
torch.manual_seed(42)
torch.mps.manual_seed(42)

# Кількість епох тренування
epochs = 1000

# Переносимо дані на пристрій
X_moons_train, y_moons_train = X_moons_train.to(device), y_moons_train.to(device)
X_moons_test, y_moons_test = X_moons_test.to(device), y_moons_test.to(device)

for epoch in range(epochs):
    model_moons.train()
    y_logits = model_moons(X_moons_train)
    y_preds = y_logits.argmax(dim=1)

    loss = loss_fn(y_logits,y_moons_train)
    acc = accuracy_fn(y_true=y_moons_train, y_pred=y_preds)


    # Оновлення ваг
    optim.zero_grad()  # Обнулення попередніх градієнтів
    loss.backward()  # Зворотне поширення помилки
    optim.step()  # Оновлення параметрів моделі

        # Оцінка моделі на тестових даних
    model_moons.eval()  # Встановлюємо модель у режим оцінки
    with torch.inference_mode():  # Вимикаємо обчислення градієнтів для тесту
        test_logits = model_moons(X_moons_test)  # Логіти для тестових даних
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)  # Прогноз для тесту

        # Розрахунок функції втрат і точності для тестових даних
        test_loss = loss_fn(test_logits, y_moons_test)
        test_acc = accuracy_fn(y_moons_test, test_pred)

    # Вивід метрик кожні 10 епох
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

# === Візуалізація межі рішення ===
model_moons.eval()  # Модель у режимі оцінки
with torch.inference_mode():
    y_logits = model_moons(X_moons_test)  # Логіти для візуалізації

# Візуалізація межі рішення
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")  # Межа рішення для тренувальних даних
plot_decision_boundary(model_moons, X_moons_train, y_moons_train)
plt.subplot(1, 2, 2)
plt.title("Test")  # Межа рішення для тестових даних
plot_decision_boundary(model_moons, X_moons_test, y_moons_test)
plt.show()
