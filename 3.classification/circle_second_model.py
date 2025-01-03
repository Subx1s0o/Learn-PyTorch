import torch
from torch import nn
from device import device
from dataset import X_test, X_train, y_test, y_train, accuracy_fn
from matplotlib import pyplot as plt
from helper_function import plot_decision_boundary

# === Модель 2: Покращена класифікаційна модель ===
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)  # Перший лінійний шар
        self.layer_2 = nn.Linear(in_features=10, out_features=10)  # Другий лінійний шар
        self.layer_3 = nn.Linear(in_features=10, out_features=1)  # Третій лінійний шар

    def forward(self, x):
        # Прямий прохід через шари моделі
        return self.layer_3(self.layer_2(self.layer_1(x)))

# Ініціалізація моделі та оптимізатора
model = CircleModelV1().to(device)
loss_fn = nn.BCEWithLogitsLoss()  # Функція втрат для класифікації
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)  # Оптимізатор


# Переносимо дані на правильний пристрій
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# Фіксуємо значення для відтворюваності
torch.manual_seed(42)
torch.mps.manual_seed(42)

# Тренування моделі 2
epochs = 1000
for epoch in range(epochs):
    model.train()  # Режим тренування
    y_logits = model(X_train).squeeze()  # Логіти (сирі виходи моделі)
    y_pred = torch.round(torch.sigmoid(y_logits))  # Прогнози після сигмоїди

    loss = loss_fn(y_logits, y_train)  # Розрахунок функції втрат
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)  # Точність тренувальних даних

    optimizer.zero_grad()  # Обнулення градієнтів
    loss.backward()  # Зворотне поширення помилки
    optimizer.step()  # Оновлення ваг

    model.eval()  # Режим оцінки
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()  # Логіти для тестових даних
        test_pred = torch.round(torch.sigmoid(test_logits))  # Прогнози для тесту

        test_loss = loss_fn(test_logits, y_test)  # Функція втрат для тестових даних
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)  # Точність для тестових даних

    # Друк результатів кожні 100 епох
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

# Візуалізація межі рішення для моделі 2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")  # Візуалізація на тренувальних даних
plot_decision_boundary(model=model, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")  # Візуалізація на тестових даних
plot_decision_boundary(model=model, X=X_test, y=y_test)
plt.show()


# === Висновок ===
# Покращена класифікаційна модель із трьома лінійними шарами була протестована на задачі класифікації.
# Незважаючи на збільшення кількості параметрів моделі, якість класифікації суттєво не покращилася.
