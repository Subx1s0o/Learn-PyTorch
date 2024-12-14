import torch


#Перевірка на наявність MPS для apple
if torch.mps.is_available():
    print("Metal GPU доступний!")
else:
    print("GPU не доступний або не підтримується.")

#Перевірка на наявність Cuda для nvidia
if torch.cuda.is_available():
    print("CUDA GPU доступний!")
else:
    print("CUDA не доступний або не підтримується.")

# Налаштовуємо глобальну змінну девайсу
device = "mps" if torch.mps.is_available() else "cpu"


# Створюємо тензор на CPU
tensor_on_cpu = torch.ones(7)

# Створюємо тензор на GPU (Metal) або CPU
tensor_on_gpu = torch.ones(7, device=device)

# Переміщуємо тензор з CPU на GPU
tensor_tranformed_from_cpu_to_gpu = tensor_on_cpu.to(device)


# Переформатувати тензор на GPU в NumPy масив неможна оскільки NumPy використовує тільки CPU, 
# тому першочергового треба повернути тип девайсу CPU тензору.

# Переміщаємо тензор з GPU на CPU, а потім в масив NumPy
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
