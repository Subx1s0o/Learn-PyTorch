import torch
from torch import nn 
from device import device
from dataset import X_test, X_train, y_test, y_train , accuracy_fn
from matplotlib import pyplot as plt
from helper_function import plot_predictions, plot_decision_boundary

# === Модель 2: Покращена класифікаційна модель ===
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

model = CircleModelV1().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

# Фіксуємо значення для відтворюваності
torch.manual_seed(42)
torch.mps.manual_seed(42)

# Тренування моделі 2
epochs = 1000
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

# Візуалізація межі рішення для моделі 2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model=model, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model=model, X=X_test, y=y_test)
plt.show()



#  === Лінійна регресія для порівняння ===
# Генерація даних для регресії
w, b = 0.7, 0.3
X_regression = torch.arange(0, 1, 0.01).unsqueeze(1)
y_regression = w * X_regression + b

# Розділення даних
train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Побудова моделі для регресії
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

loss_fn_2 = nn.L1Loss()
optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01)

X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

# Тренування моделі регресії
for epoch in range(epochs):
    model_2.train()
    y_pred = model_2(X_train_regression)
    loss = loss_fn_2(y_pred, y_train_regression)

    optimizer_2.zero_grad()
    loss.backward()
    optimizer_2.step()

    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn_2(test_pred, y_test_regression)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# Візуалізація результатів регресії
with torch.inference_mode():
    y_pred = model_2(X_test_regression)

plot_predictions(
    X_train_regression.cpu(),
    y_train_regression.cpu(),
    X_test_regression.cpu(),
    y_test_regression.cpu(),
    y_pred.cpu()
)
plt.show()

# === Висновки ===
# 1. Початкова класифікаційна модель демонструє слабку здатність навчатися (loss ~50%) через недостатню складність.
# 2. Покращена модель з додатковими шарами не суттєво покращила результати через відсутність активаційних функцій.
# 3. У задачі регресії модель ефективно навчалася, зменшуючи loss, завдяки прямолінійній природі задачі.
