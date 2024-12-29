import torch
import matplotlib.pyplot as plt

# Створюємо батч випадкових зображень
images = torch.rand(32, 1, 28, 28)
test_image = images[0]
print(f"Shape of the batch of images: {images.shape}")
print(f"Shape of the single image: {test_image.shape}")
print(f"Single image: {test_image}")

# Створюємо конволюційний шар
conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)

# Пропускаємо зображення через конволюційний шар
output = conv_layer(test_image.unsqueeze(0))  # додамо batch dimension
print(f"Shape of the output: {output.shape}")
print(f"Output: {output}")

# Візуалізація результатів
output = output.squeeze(0)  # Видаляємо batch dimension для зручності візуалізації

# Виводимо всі 10 фільтрованих карт
fig, axes = plt.subplots(2, 5, figsize=(15, 10))
for i in range(10):
    axes[i // 5, i % 5].imshow(output[i].detach().numpy(), cmap='gray')
    axes[i // 5, i % 5].set_title(f"Feature Map {i+1}")
    axes[i // 5, i % 5].axis('off')  # Вимикаємо осі

plt.show()
