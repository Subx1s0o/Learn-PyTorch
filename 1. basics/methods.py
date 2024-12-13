import torch

# Метод Тензор може містити тільки однакову кількість даних у масивах
# На прикладі показаний 2D тензор
# 0D  це одиничне число, називається Скаляр
# 1D масив називається вектором (1 масив усередині)
# 2D масив називається матрицею (масив масивів векторів)
# 3D  масив називається Тензор (масив масивів масивів векторів,
#  або масив матриць)
# Також приймає тип даних
t = torch.tensor([[1,2,3], [1,2,3]])


# дозволяє дізнатися виміри тензора, чим більше вкладеність
# тим більше вимір
t2 = t.ndim

#визначає виміри тензора, тобто кількість елементів по кожному виміру.
# Це можна інтерпретувати як кількість векторів (або "рядків" і "стовпців")
#  та кількість елементів в кожному з них.
t3 = t.size()

# Створює рандомний тензор, приймає розмір, в даному прикладу створює тензор
# розміром 3з4. Також опціонально приймає тип даних через dtype, але рандом 
# може генерувати тільки числа з плаваючою комою. Також приймає device, 
# воно вирішує чи буде тензор згенеровано на CPU чи на GPU.
random_tensor = torch.rand(3,4, dtype=torch.float32,device="cpu")

# Дозволяє створити тензор заданих розмірів з нулями
# Zeros like дозволяє створити тензор з нулями наслідуючись від розміру
zeros = torch.zeros(2,3)
zeros_like = torch.zeros_like(random_tensor)

# Дозволяє створити тензор заданих розмірів з одиницями
# Ones like дозволяє створити тензор з одиницями наслідуючись від розміру
ones = torch.ones(2,3)
zeros_like = torch.ones_like(random_tensor)

# Визначає який тип даних міститься у тензора
ones.dtype

# Створює масив від якогось до якогось числа з вказаними кроками через step
one_to_ten = torch.arange(start=1,end=10,step=1)

# Тензори можна додавати,віднімати,множити та ділити
sum = t+ones
multiply = t * torch.tensor([[2,4,6],[4,2,1]])
divide = t / torch.tensor([[2,4,6],[4,2,1]])
subtraction = t - torch.tensor([[2,4,6],[4,2,1]])


# Дозволяє перемножати матриці, https://www.mathsisfun.com/algebra/matrix-multiplying.html
torch.matmul(multiply,torch.tensor([[2,2],[2,4],[7,5]]))

# @ це оператор для множення матриць
# (3,2) @ (3,2) # НЕ буде працювати
# (2,3) @ (3,2) # буде працювати
# (3,5) @ (5,4) # буде працювати
multiply @ torch.tensor([[2,2],[2,4],[7,5]])

# Коротка версія matmul
torch.mm(multiply,torch.tensor([[2,2],[2,4],[7,5]]))

# обчислює мінімальне значення всіх елементів тензора.
torch.min(t)
t.min()

# обчислює максимальне значення всіх елементів тензора.
torch.max(t)
t.max()

# обчислює середнє значення всіх елементів тензора. Обовʼязковий float тип
torch.mean(torch.tensor([[90,232],[90,43]],dtype=torch.float32))

# Обчислює суму усіх елементів тензора
torch.sum(t)
t.sum()

# Знаходять індекс мінімального чи максимального числа у тензорі 
t.argmin()
t.argmax()