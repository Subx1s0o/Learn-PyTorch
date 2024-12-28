from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch 
from matplotlib import pyplot as plt
from torch import nn 

# Завантаження та підготовка даних FashionMNIST:
# 1. Використовуємо torchvision.datasets для завантаження набору даних FashionMNIST.
# 2. # Використовуємо `transforms.ToTensor()` для перетворення зображень у тензори, щоб модель могла працювати з числовими даними.
#      0. Зображення в FashionMNIST - це 28x28 пікселів у відтінках сірого (1 канал).
#      1. `ToTensor()` виконує перетворення:
#          - Піксельні значення, представлені цілими числами [0, 255] (1 канал, 0 чорний, 255 білий), нормалізуються 
#            до діапазону [0.0, 1.0] шляхом поділу на 255.
#          - Результат представлений як тензор розмірності [канал, висота, ширина]. 
# 3. Розділяємо набір на навчальні та тестові дані.
train_data = datasets.FashionMNIST(root="data", train=True, download=True,transform=transforms.ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False, download=True,transform=transforms.ToTensor(), target_transform=None)

# Визначення розміру батчу для DataLoader
BATCH = 32

# Створення DataLoader:
# 1. Перетворює дані в ітераційний формат для обробки в модель.
# 2. Навчальний DataLoader перемішує дані (shuffle=True), щоб уникнути залежності від порядку зразків.
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH, shuffle=False)

# Отримання першого батчу даних:
# Використовуємо next() для отримання батчу з features (зображення) та labels (мітки класів).
train_features_batch, train_labels_batch = next(iter(train_dataloader))
test_features_batch, test_labels_batch = next(iter(test_dataloader))

# Список назв класів FashionMNIST:
# Класи представляють 10 різних типів одягу (наприклад, футболка, штани тощо).
class_names = train_data.classes

# Вибір випадкового зображення та його мітки з батчу:
# 1. torch.manual_seed(42): Встановлюємо фіксоване зерно для відтворюваності результатів.
# 2. torch.randint(): Генеруємо випадковий індекс.
# 3. Вибираємо зображення та мітку класу за цим індексом.
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item() 

# Вибір випадкового індексу для зображення та лейбла з тренувальних даних
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()

# Вибір зображення та відповідного лейбла за одним індексом
# train_features_batch містить зображення, а train_labels_batch містить лейбли для цих зображень
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]  

# Перетворення кольорового зображення в чорно-біле:
# Для цього беремо середнє значення пікселів по всіх трьох каналах (RGB)
img_gray = torch.mean(img, dim=0)  # Середнє значення по каналах (RGB), отримуємо зображення в градаціях сірого

# Виведення зображення за допомогою matplotlib:
# Використовуємо cmap='gray' для відображення зображення в чорно-білих тонах
plt.imshow(img_gray, cmap='gray')

# Встановлюємо заголовок для зображення, використовуючи клас (мітку) для цього зображення
# class_names - це список або словник, що містить назви класів, до яких належать зображення
plt.title(f"Label: {class_names[label.item()]}")

# Вимикаємо осі для чистого вигляду зображення
plt.axis('off')

# Виводимо зображення
plt.show()

# Flatten (перетворення зображення у вектор):
# 1. nn.Flatten(): Перетворює 2D зображення (наприклад, 28x28) у 1D вектор (розміром 784).
# 2. Використовується для передавання даних у повнозв’язний шар моделі.
flatten_model = nn.Flatten()

# Перетворення першого зображення батчу через Flatten:
x = train_features_batch[0]
output = flatten_model(x)

# Висновок:
# У цьому файлі я виконав завантаження даних FashionMNIST, їхню підготовку до роботи з моделлю,
# вивчив, як працюють DataLoader та трансформації. Також я побачив, як зображення перетворюються у тензори 
# і як функція Flatten готує їх для використання у моделях. 
