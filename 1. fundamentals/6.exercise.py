# https://www.learnpytorch.io/00_pytorch_fundamentals/#extra-curriculum

# Create a random tensor with shape (7, 7).

# Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7) 
# (hint: you may have to transpose the second tensor).

# Set the random seed to 0 and do exercises 2 & 3 over again.

# Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? 
# (hint: you'll need to look into the documentation for torch.cuda for this one). If there is, set the GPU random seed to 1234.

# Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this). 

# Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).

# Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of one of the tensors).

# Find the maximum and minimum values of the output of 7.

# Find the maximum and minimum index values of the output of 7.

# Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed 
# to be left with a tensor of shape (10).

import torch

device = "mps" if torch.mps.is_available() else "cpu"

# torch.manual_seed(0) для CPU
# tensor = torch.rand(7,7) для CPU
torch.mps.manual_seed(1234)
tensor = torch.rand(7,7,device=device)

# torch.manual_seed(0) для CPU
# transposed_tensor = torch.rand(1,7).transpose(1,0) для CPU
torch.mps.manual_seed(1234)
transposed_tensor = torch.rand(1,7,device=device).transpose(1,0)

matmul_result = tensor @ transposed_tensor # або torch.mm або torch.matmul

print(matmul_result)


#2 Створення двух рандомних тензорів розмірок 2,3 на CPU та відправка їх обох на GPU девайс
torch.manual_seed(1234)
first_tensor = torch.rand(2,3).to(device)

torch.manual_seed(1234)
second_tensor = torch.rand(2,3).to(device)

# Перевернення тензору на 3,2 розмір та перемноження матриці
second_matmul_result = first_tensor @ second_tensor.transpose(1,0)

matmul_max = second_matmul_result.max()
matmul_min = second_matmul_result.min()

matmul_index_max = second_matmul_result.argmax()
matmul_index_min = second_matmul_result.argmin()


#3

random_tensor = torch.rand(1,1,1,10)
squeezed_tensor = random_tensor.squeeze()

print(squeezed_tensor)