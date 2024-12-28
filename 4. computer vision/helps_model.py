import torch
import torch.nn as nn
from tqdm.auto import tqdm
from helper_function import accuracy_fn

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn=accuracy_fn,
               device="cpu"):
    loss, acc = 0, 0

    model.eval()  # Перехід моделі в режим оцінки (відключення Dropout та BatchNorm)
    model.to(device)  # Переміщуємо модель на заданий пристрій (CPU чи GPU)

    with torch.inference_mode():  # Вимикаємо обчислення градієнтів, що прискорює процес тестування
        for X, y in data_loader:
            # Переносимо батч з даними на пристрій
            X, y = X.to(device), y.to(device)

            # Прогноз від моделі для поточного батчу
            y_pred = model(X)

            # Обчислюємо втрати для поточного батчу
            loss += loss_fn(y_pred, y).item()

            # Обчислюємо точність на основі правильних прогнозів
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

        # Усереднюємо втрати та точність
        loss /= len(data_loader)
        acc /= len(data_loader)

    # Повертаємо словник з результатами
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
    model.to(device)  # Переміщуємо модель на заданий пристрій (CPU чи GPU)

    for epoch in tqdm(range(epochs)):  # Ітерація по епохах тренування
        train_loss = 0
        total_samples = 0  # Лічильник для загальної кількості вибірок

        model.train()  # Перехід моделі в режим тренування (включає Dropout та інші специфічні операції для тренування)
        for batch, (X, y) in enumerate(train_dataloader):  # Ітерація по батчах тренувальних даних
            X, y = X.to(device), y.to(device)  # Переносимо дані на пристрій

            # Прогноз від моделі
            y_pred = model(X)

            # Обчислюємо втрати для поточного батчу
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # Очищуємо градієнти
            optimizer.zero_grad()

            # Зворотне поширення помилки (backpropagation)
            loss.backward()

            # Оновлюємо ваги моделі
            optimizer.step()

            total_samples += len(X)  # Рахуємо кількість оброблених зразків

            # Кожні 400 батчів виводимо прогрес
            if batch % 400 == 0:
                print(f"Looked at {total_samples} samples so far.")

        # Усереднюємо втрати для тренування по всіх батчах
        train_loss /= len(train_dataloader)

        # Оцінка моделі на тестових даних
        result = eval_model(model, test_dataloader, loss_fn, accuracy_fn, device=device)

        # Виводимо результати для поточної епохи
        print(f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {result['model_loss']:.4f} | Test Acc: {result['model_acc']:.4f}")

# Робимо передбачення для зразків
def make_predictions(model, data, device="cpu"):
    model.eval()  # Перехід моделі в режим оцінки
    model.to(device)  # Переміщуємо модель на заданий пристрій
    pred_probs = []

    with torch.inference_mode():  # Вимикаємо обчислення градієнтів
        for sample in data:
            sample = sample.to(device)  # Переносимо зразок на пристрій
            pred_logit = model(sample)  # Отримуємо прогноз від моделі
            pred_prob = torch.softmax(pred_logit, dim=1)  # Застосовуємо softmax для отримання ймовірностей

            # Додаємо прогноз до списку
            pred_probs.append(pred_prob.cpu())  # Переносимо результат на CPU, щоб зберегти його в пам'яті

    return torch.cat(pred_probs)  # Об'єднуємо всі результати в один тензор
