from torch.utils.data import random_split
from torchvision import datasets, transforms

# Завантажуємо датасет
dataset = datasets.OxfordIIITPet(
    root="data",
    download=True,
    transform=transforms.ToTensor()
)

# Визначаємо пропорції
dataset_size = len(dataset)  # Загальний розмір датасету
train_size = int(0.8 * dataset_size)  # 80% для тренування
test_size = dataset_size - train_size  # Решта (20%) для тесту

# Використовуємо random_split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Розмір тренувального набору: {len(train_dataset)}")
print(f"Розмір тестового набору: {len(test_dataset)}")
