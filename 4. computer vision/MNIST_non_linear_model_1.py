import torch
import torch.nn as nn
from data import class_names,train_dataloader, test_dataloader
from helps_model import  train_model

class MNISTModelV1(nn.Module):
    def __init__(self, input_shape, hidden_units,output_shape):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x):
       return self.layer_stack(x)

torch.manual_seed(42)
model = MNISTModelV1(
    input_shape=28*28,
    hidden_units=10,
    output_shape=len(class_names)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=20)
