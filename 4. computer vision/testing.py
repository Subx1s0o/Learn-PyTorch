import torch
from pathlib import Path
from data import test_data, class_names
from matplotlib import pyplot as plt
from helps_model import make_predictions
from MNIST_cnn_model import MNISTModelV2

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Шлях до файлу з моделлю
MODEL_PATH = Path("models/full_mnist_model.pth")

# Завантажуємо повну модель
model = MNISTModelV2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

model.load_state_dict(torch.load(MODEL_PATH))

# Переводимо модель у режим оцінки
model.eval()

# Вибираємо випадкові 9 зразків із тестових даних
import random
random.seed(42)
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), 9):
    test_samples.append(sample)
    test_labels.append(label)

# Перетворюємо список зразків на батч з розміром [batch_size, channels, height, width]
test_samples = torch.stack(test_samples).to(device)  # Перетворюємо в тензор без додавання нового виміру batch_size

# Візуалізуємо перший зразок

# Перекладаємо зображення на CPU перед відображенням
# plt.imshow(test_samples[0].cpu().squeeze(), cmap='gray')  # .squeeze() для усунення зайвого виміру
# plt.title(class_names[test_labels[0]])
# plt.show()



# Робимо передбачення для обраних зразків
pred_probs = make_predictions(model, test_samples.unsqueeze(dim=1), device=device)

# Виводимо результати для перших двох зразків
print(pred_probs[:2])

pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)

print(test_labels)
