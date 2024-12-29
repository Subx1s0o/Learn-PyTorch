from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import torch

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Завантажуємо датасет з трансформаціями
train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transforms.ToTensor())

# Список класів (10 класів для CIFAR10)
class_names = train_data.classes  # Використовуємо клас зі змінної train_data

# Створюємо DataLoader
batch_size = 32  # Задайте бажаний розмір батча
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Виводимо приклад зображення
image, label = next(iter(train_dataloader))
plt.imshow(image[0].permute(1, 2, 0))  # permute для зміни порядку каналів (C, H, W) -> (H, W, C)
plt.title(class_names[label[0].item()])  # Виводимо мітку класу
plt.axis("off")
plt.show()
