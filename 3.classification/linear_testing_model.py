import torch
from torch import nn
from device import device
from matplotlib import pyplot as plt
from helper_function import plot_predictions

# === Лінійна регресія для порівняння ===
# Генерація даних для регресії
w, b = 0.7, 0.3  # Параметри лінійної залежності
X_regression = torch.arange(0, 1, 0.01).unsqueeze(1)  # Вхідні дані
y_regression = w * X_regression + b  # Вихідні дані

# Розділення даних на тренувальну та тестову частини
train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Побудова моделі для регресії
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),  # Перший лінійний шар
    nn.Linear(in_features=10, out_features=10),  # Другий лінійний шар
    nn.Linear(in_features=10, out_features=1)  # Третій лінійний шар
).to(device)

loss_fn_2 = nn.L1Loss()  # Функція втрат для регресії
optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01)  # Оптимізатор

# Перенесення даних на обране пристрій
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

# Тренування моделі регресії
epochs = 1000
for epoch in range(epochs):
    model_2.train()  # Режим тренування
    y_pred = model_2(X_train_regression)  # Прогнози для тренувальних даних
    loss = loss_fn_2(y_pred, y_train_regression)  # Розрахунок втрат

    optimizer_2.zero_grad()  # Обнулення градієнтів
    loss.backward()  # Зворотне поширення помилки
    optimizer_2.step()  # Оновлення ваг

    model_2.eval()  # Режим оцінки
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)  # Прогнози для тестових даних
        test_loss = loss_fn_2(test_pred, y_test_regression)  # Функція втрат для тесту

    # Друк результатів кожні 100 епох
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# Візуалізація результатів регресії
with torch.inference_mode():
    y_pred = model_2(X_test_regression)  # Прогнози на тестових даних

# Побудова графіків для регресії
plot_predictions(
    X_train_regression.cpu(),
    y_train_regression.cpu(),
    X_test_regression.cpu(),
    y_test_regression.cpu(),
    y_pred.cpu()
)
plt.show()


# Я тестував дві моделі — для класифікації та для регресії. Класифікаційні моделі, хоч і тренуються, 
# давали великий лосс (50%), тобто вони не вміють добре передбачати. А ось звичайна регресійна модель 
# показала кращий результат, лосс зменшувався, що означає, що вона навчалася і робила прогнози краще.
# Це показує, що класифікаційні моделі не працюють так, як треба, але я поки не знаю точно чому.