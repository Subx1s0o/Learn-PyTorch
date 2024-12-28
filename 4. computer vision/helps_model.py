import torch
import torch.nn as nn
from tqdm.auto import tqdm
from helper_function import accuracy_fn

# Оцінка моделі
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn=accuracy_fn,
               device="cpu"):
    loss, acc = 0, 0

    model.eval()
    model.to(device)

    with torch.inference_mode():
        for X, y in data_loader:
            # Переносимо дані на пристрій
            X, y = X.to(device), y.to(device)

            # Прогноз
            y_pred = model(X)

            # Обчислення втрат і точності
            loss += loss_fn(y_pred, y).item()
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

        # Усереднюємо втрати та точність
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss,
        "model_acc": acc
    }


def train_and_test_model(model: nn.Module, 
                         train_dataloader, 
                         test_dataloader, 
                         loss_fn: nn.Module, 
                         optimizer: torch.optim.Optimizer, 
                         epochs: int,
                         device="cpu"):
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        total_samples = 0  # Лічильник для загальної кількості вибірок

        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_samples += len(X)  # Рахуємо вибірки в цьому батчі

            if batch % 400 == 0:
                print(f"Looked at {total_samples} samples so far.")

        train_loss /= len(train_dataloader)

        # Тестування моделі
        result = eval_model(model, test_dataloader, loss_fn, accuracy_fn, device=device)

        print(f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {result['model_loss']:.4f} | Test Acc: {result['model_acc']:.4f}")


# Функція для прогнозів
def make_predictions(model, data, device="cpu"):
    model.eval()
    model.to(device)
    pred_probs = []

    with torch.inference_mode():
        for sample in data:
            sample = sample.to(device)  # Переносимо зразок на пристрій
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit, dim=1)

            # Додаємо до списку результатів
            pred_probs.append(pred_prob.cpu())

    return torch.cat(pred_probs)  # Об'єднуємо всі результати в один тензор
