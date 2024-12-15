import torch

a = torch.rand(3,4)

b = torch.rand(3,4)

# Воно порівнює кожне число в обох тензорах. буде False
print(a == b)

# torch.manual_seed() встановлює початкове значення для генератора випадкових чисел, що забезпечує 
# відтворюваність результатів при виконанні випадкових операцій. 
# Дозволяє отримати однакові результати при кожному запуску коду.
# Працює одноразово для кожного виклику ранд
torch.manual_seed(42)
c = torch.rand(3,4)

torch.manual_seed(42)
d = torch.rand(3,4)


print(c == d )