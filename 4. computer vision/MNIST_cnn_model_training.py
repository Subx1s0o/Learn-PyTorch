import torch
from pathlib import Path
from data import  class_names, train_dataloader, test_dataloader
import torch.nn as nn
from helps_model import train_and_test_model, eval_model
from MNIST_cnn_model import MNISTModelV2

# Вибір пристрою для навчання

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Ініціалізація моделі
torch.manual_seed(42)
model = MNISTModelV2(
    input_shape=1,
    hidden_units=10,
    output_shape=len(class_names)
).to(device)  # Переносимо модель на GPU (MPS) або CPU

# Функція втрат та оптимізатор
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Переносимо дані на пристрій (MPS або CPU)
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
    epochs=5,
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
MODEL_NAME = "full_mnist_model.pth"

# Повний шлях до файлу, де буде збережена модель
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Збереження моделі 
torch.save(model.state_dict(), f=MODEL_SAVE_PATH)
