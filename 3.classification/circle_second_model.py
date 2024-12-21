import torch
from torch import nn 
from device import device
from dataset import X_test, X_train, y_test, y_train , accuracy_fn
from matplotlib import pyplot as plt
from helper_function import plot_decision_boundary

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        # return z

        return self.layer_3(self.layer_2(self.layer_1(x)))

model = CircleModelV1().to(device=device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

torch.manual_seed(42)
torch.mps.manual_seed(42)

epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


for epoch in range(epochs):
    model.train()

    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true=y_train,y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()

    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,y_test)

        test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")


# Візуалізація межі рішення для тренувальних та тестових даних
plt.figure(figsize=(12, 6))  # Встановлюємо розмір графіка

# Графік для тренувальних даних
plt.subplot(1, 2, 1)  # Розташування на графіку (1 рядок, 2 стовпці, перший)
plt.title("Train")  # Заголовок для графіка
plot_decision_boundary(model=model, X=X_train, y=y_train)  # Будуємо межу рішення для тренувальних даних

# Графік для тестових даних
plt.subplot(1, 2, 2)  # Розташування на графіку (1 рядок, 2 стовпці, другий)
plt.title("Test")  # Заголовок для графіка
plot_decision_boundary(model=model, X=X_test, y=y_test)  # Будуємо межу рішення для тестових даних

# Відображаємо графік
plt.show()
