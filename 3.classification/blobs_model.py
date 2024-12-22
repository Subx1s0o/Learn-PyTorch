import torch
from torch import nn
from dataset import X_blob_test,X_blob_train,y_blob_test,y_blob_train, accuracy_fn
from device import device
from matplotlib import pyplot as plt
from helper_function import plot_decision_boundary

class BlobModel(nn.Module):
    def __init__(self, input_features, out_features, hidden_units=8):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),  # 1-й шар
            nn.ReLU(),  # Активаційна функція
            nn.Linear(in_features=hidden_units, out_features=hidden_units),  # 2-й шар
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features)  # Останній шар
        )
    
    def forward(self,x):
        return self.linear_layer_stack(x)

model_blob = BlobModel(input_features=2, out_features=4).to(device)


loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model_blob.parameters(), lr=0.1)

torch.manual_seed(42)
torch.mps.manual_seed(42)

epochs = 1000

X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model_blob.train()

    # Прогноз для тренувальних даних
    y_logits = model_blob(X_blob_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1) 

    loss = loss_fn(y_logits, y_blob_train)

    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_preds)

    optim.zero_grad()

    loss.backward()

    optim.step()

    model_blob.eval()
    with torch.inference_mode():
        test_logits = model_blob(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1) 

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_blob_test, test_pred)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")

model_blob.eval()
with torch.inference_mode():
    y_logits = model_blob(X_blob_test)


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_blob,X_blob_train,y_blob_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_blob,X_blob_test,y_blob_test)

plt.show()