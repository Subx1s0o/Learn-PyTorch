import torch
import torch.nn as nn
from tqdm.auto import tqdm
from helper_function import accuracy_fn

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn=accuracy_fn):
    loss,acc = 0,0

    model.eval()
    
    with torch.inference_mode():
        for X,y in data_loader:
            y_pred = model(X)

            loss += loss_fn(y_pred,y)
            acc +=accuracy_fn(y, y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {

        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc

        }


def train_and_test_model(model: nn.Module, 
                train_dataloader, 
                test_dataloader, 
                loss_fn: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                epochs: int):

    for epoch in tqdm(range(epochs)):
        train_loss = 0

        for batch, (X, y) in enumerate(train_dataloader):
            model.train()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

        train_loss /= len(train_dataloader)

       
        result = eval_model(model, test_dataloader, loss_fn, accuracy_fn)

        
        print(f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {result['model_loss']:.4f} | Test Acc: {result['model_acc']:.4f}")


