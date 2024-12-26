import torch
import torch.nn as nn
from data import class_names, train_dataloader, test_dataloader
from helps_model import train_and_test_model, eval_model

class MNISTModelV2(nn.Module):
    """
    Конволюційна нейронна мережа (CNN) для набору даних MNIST.
    Ця модель складається з двох конволюційних блоків, після яких йде повнозв'язний шар для класифікації.
    Атрибути:
        conf_blok_1 (nn.Sequential): Перший конволюційний блок.
        conf_blok_2 (nn.Sequential): Другий конволюційний блок.
        classifier (nn.Sequential): Повнозв'язний шар для класифікації.
    """
    def __init__(self, input_shape, hidden_units, output_shape):
        """
        Ініціалізує MNISTModelV2.
        Аргументи:
            input_shape (int): Кількість вхідних каналів.
            hidden_units (int): Кількість прихованих одиниць у конволюційних шарах.
            output_shape (int): Кількість вихідних одиниць (кількість класів).
        """
        super().__init__()

        self.conf_blok_1 = nn.Sequential(
           nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2)
        )

        self.conf_blok_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
           nn.Flatten(),
           nn.Linear(in_features=hidden_units*7*7, out_features=output_shape),  
        )
    
    def forward(self, x):
        """
        Визначає прямий прохід моделі.
        Аргументи:
            x (torch.Tensor): Вхідний тензор.
        Повертає:
            torch.Tensor: Вихідний тензор після проходження через модель.
        """
        x = self.conf_blok_1(x)
        # print(f"Shape after conf_blok_1: {x.shape}")
        x = self.conf_blok_2(x)
        # print(f"Shape after conf_blok_2: {x.shape}")
        x = self.classifier(x)
        # print(f"Shape after classifier: {x.shape}")
        return x

torch.manual_seed(42)
model = MNISTModelV2(
    input_shape=1,
    hidden_units=10,
    output_shape=len(class_names)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

train_and_test_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=3)

model_results = eval_model(model, test_dataloader, loss_fn)
