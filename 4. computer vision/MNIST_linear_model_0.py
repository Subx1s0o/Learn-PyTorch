from torch import nn
import torch
from data import class_names,train_dataloader, test_dataloader
from helper_function import accuracy_fn
from tqdm.auto import tqdm


class MNISTModelV0(nn.Module):
    def __init__(self, input_shape, hidden_units,output_shape):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
       return self.layer_stack(x)

torch.manual_seed(42)

model = MNISTModelV0(
    input_shape=28*28,
    hidden_units=10,
    output_shape=len(class_names)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

epochs = 3

for epoch in tqdm(range(epochs)):
   
    train_loss = 0

    for batch, (X,y) in enumerate(train_dataloader):
       model.train()

       y_pred = model(X)

       loss = loss_fn(y_pred,y)

       train_loss += loss 

       optimizer.zero_grad()

       loss.backward()

       optimizer.step()

       if batch % 400 == 0:
           print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples. ")

    train_loss /= len(train_dataloader)   

    test_loss, test_acc = 0,0
    model.eval()

    with torch.inference_mode():

        for X_test,y_test in test_dataloader:
            test_pred = model(X_test)

            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)

        test_acc /= len(test_dataloader)
    
    print(f"\nTrain Loss: {train_loss:.4f} | Test Loss {test_loss:.4f}, Test acc: {test_acc:.4f}")


