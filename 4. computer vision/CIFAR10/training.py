from pathlib import Path
from torch import nn
from CIFAR_cnn_Model import CIFARModel
import torch
from helps_function import train_and_test_model, eval_model
from data import device, train_dataloader, test_dataloader

# Ініціалізація моделі
model = CIFARModel(
    input_shape=3,  # Кількість каналів у зображенні
    hidden_units=96,  # Кількість фільтрів у першому конволюційному шарі
    output_shape=10  # Кількість класів для класифікації
).to(device)


loss_fn = nn.CrossEntropyLoss()  # Функція втрат
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Оптимізатор

def prepare_dataloader(dataloader, device):
    for batch in dataloader:
        inputs, targets = batch
        yield inputs.to(device), targets.to(device)


# Навчання
train_and_test_model(
    model=model, 
    train_dataloader=train_dataloader, 
    test_dataloader=test_dataloader, 
    loss_fn=loss_fn, 
    optimizer=optimizer, 
    epochs=10,
    device=device
)

# Оцінка моделі
model_results = eval_model(
    model=model, 
    data_loader=test_dataloader, 
    loss_fn=loss_fn,
    device=device
)



# Створюємо шлях для збереження моделі
MODEL_PATH = Path("models")  

# Перевіряємо, чи існує шлях, і якщо ні — створюємо директорію
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Назва файлу для збереження моделі
MODEL_NAME = "full_cifar_model.pth"

# Повний шлях до файлу, де буде збережена модель
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Збереження моделі 
torch.save(model.state_dict(), f=MODEL_SAVE_PATH)
