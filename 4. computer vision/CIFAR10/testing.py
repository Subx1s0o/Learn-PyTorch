import torch
from pathlib import Path
from data import test_data, class_names, test_dataloader
from matplotlib import pyplot as plt
from helps_function import make_predictions
from CIFAR_cnn_Model import CIFARModel
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Шлях до файлу з моделлю
MODEL_PATH = Path("models/full_cifar_model.pth")

# Завантажуємо повну модель
model = CIFARModel(input_shape=3, hidden_units=96, output_shape=len(class_names)).to(device)

model.load_state_dict(torch.load(MODEL_PATH))

# Переводимо модель у режим оцінки
model.eval()

# Вибираємо випадкові 9 зразків із тестових даних
import random
# random.seed(42)
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

# Отримуємо передбачені класи
pred_classes = pred_probs.argmax(dim=1)

# Візуалізуємо результати
plt.figure(figsize=(9, 9))

nrows, ncols = 3, 3
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i + 1)
    # Виводимо кожний зразок
    plt.imshow(sample.cpu().permute(1, 2, 0))  # permute для зміни порядку каналів (C, H, W) -> (H, W, C)
    color = 'green' if pred_classes[i] == test_labels[i] else 'red'
    plt.title(f"Pred: {class_names[pred_classes[i]]} | Truth: {class_names[test_labels[i]]}", color=color, fontsize=8)
    plt.axis('off')

plt.show()



# Перетворюємо прогнози та мітки на тензори
y_preds = []
model.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        y_pred = torch.softmax(y_logit.squeeze(), dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())

# Об'єднуємо всі передбачення в один тензор
y_pred_tensor = torch.cat(y_preds)

# Переконвертуємо test_data.targets в тензор, якщо це потрібно
test_labels_tensor = torch.tensor(test_data.targets)

# Створення матриці плутанини
confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(y_pred_tensor, test_labels_tensor)

# Візуалізація матриці плутанини
fig, ax = plot_confusion_matrix(confmat_tensor.numpy(), figsize=(10, 7), class_names=class_names)
plt.show()
