import torch

device = ""
# Перевіряємо, чи є підтримка MPS (для Mac)
if torch.mps.is_available():
    device = "mps"
# Перевіряємо, чи є підтримка CUDA (для NVIDIA GPU)
elif torch.cuda.is_available():
    device = "cuda"
else:
    # Якщо немає підтримки MPS або CUDA, використовуємо CPU
    device = "cpu"

