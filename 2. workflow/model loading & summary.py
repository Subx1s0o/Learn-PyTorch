import torch
from linear_model import LinearRegressionModel  # Імпортуємо клас


#Що б завантажити збережені дані тренерованої моделі треба створити новий інстанс цієї моделі і 
# завантажити в неї збережені дані
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load("models/01_workflow_model.pth",weights_only=True))

print(loaded_model.state_dict())