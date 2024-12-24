from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch 
from matplotlib import pyplot as plt
from torch import nn 

train_data = datasets.FashionMNIST(root="data", train=True, download=True,transform=transforms.ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False, download=True,transform=transforms.ToTensor(), target_transform=None)

BATCH = 32

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH, shuffle=False)

train_features_batch , train_labels_batch = next(iter(train_dataloader))
test_features_batch , test_labels_batch = next(iter(test_dataloader))



class_names = train_data.classes

torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()

img, label = train_features_batch[random_idx], test_labels_batch[random_idx]

flatten_model = nn.Flatten()

x = train_features_batch[0]
output = flatten_model(x)

