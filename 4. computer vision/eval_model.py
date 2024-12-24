import torch

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
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