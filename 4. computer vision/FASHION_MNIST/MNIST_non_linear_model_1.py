import torch
import torch.nn as nn
from data import class_names, train_dataloader, test_dataloader  # Імпортуємо необхідні дані та завантажувачі даних.
from helps_model import train_and_test_model  # Імпортуємо функцію для тренування моделі.

# Створення класу моделі для класифікації зображень MNIST
class MNISTModelV1(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        # Описуємо шари моделі.
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # Перетворюємо 2D зображення в 1D вектор (28x28 = 784)
            nn.Linear(in_features=input_shape, out_features=hidden_units),  # Лінійний шар між входом і прихованим шаром
            nn.ReLU(),  # Нелінійна активація (ReLU) для прихованого шару
            nn.Linear(in_features=hidden_units, out_features=output_shape),  # Лінійний шар між прихованим шаром і виходом
            nn.ReLU()  # Ще одна нелінійна активація для виходу
        )
    
    # Функція forward описує, як дані проходять через модель.
    def forward(self, x):
        return self.layer_stack(x)  # Пропускаємо через всі шари, описані вище.

# Встановлюємо початкове значення для випадкових чисел для відтворюваності.
torch.manual_seed(42)

# Створюємо модель з конкретними параметрами: розміри входу, кількість одиниць у прихованому шарі та кількість класів на виході.
model = MNISTModelV1(
    input_shape=28*28,  # Вхідний розмір (784 для MNIST)
    hidden_units=10,  # Кількість одиниць у прихованому шарі
    output_shape=len(class_names)  # Кількість класів (10 для MNIST)
)

# Визначаємо функцію втрат (крос-ентропія для задачі класифікації).
loss_fn = nn.CrossEntropyLoss()

# Визначаємо оптимізатор (SGD - стохастичний градієнтний спуск).
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Тренуємо модель на 20 епохах, використовуючи train_model для тренування.
train_and_test_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=3)


'''У цьому коді я створив модель для класифікації зображень з набору даних MNIST. Вона складається 
з двох лінійних шарів, які перетворюють вхідні зображення в 10 класів. Я також додали функцію активації 
ReLU після кожного з лінійних шарів для додавання нелінійності в модель. Проте, можливо, саме через ці додаткові 
функції активації модель почала класифікувати гірше на 10%, порівняно з попередньою версією без них. '''