from time import sleep
from model import Model, trainLoop, CoolDataLoader
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import keyboard
import torch

batch_size = 256
train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



print("Starting pretraining")
model = Model(num_outputs=4)
data = CoolDataLoader(train_loader, "rotate", num_classes=4)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
trainLoop(data, 1, optimizer, model, criterion)
print("Trainloop finished or stopped\n\n")
sleep(0.5)



print("Starting downstream digit classification")
digitClassifier = torch.nn.Linear(32, 10)
model.replace_output(digitClassifier)
data = CoolDataLoader(train_loader)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
trainLoop(data, 1, optimizer, model, criterion)
print("Trainloop finished or stopped\n\n")
