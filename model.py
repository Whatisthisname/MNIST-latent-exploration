import numpy as np
import torch.nn as nn
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import keyboard
import torch.nn.functional as F
from pipetools import pipe
import torch


class PrintLayer________(nn.Module):
    def __init__(self, message=""):
        super(PrintLayer________, self).__init__()
        self.message = message
    def forward(self, x):
        print(self.message, x.shape)
        return x


class Model(nn.Module):
    def __init__(self, num_outputs : int):
        super().__init__()
        self.conv = nn.Sequential( #out:  1  28  28
            # PrintLayer________("Conv input:\n"),
            nn.Conv2d(1, 10, 3),   #out: 10  26  26
            # PrintLayer________(),
            nn.ELU(),              #out: 10  26  26
            # PrintLayer________(),
            nn.MaxPool2d(2),       #out: 10  13  13
            # PrintLayer________(),
            nn.BatchNorm2d(num_features=10), # out: 10 13 13
            # PrintLayer________(),
            nn.Conv2d(10, 20, 3),  #out: 20  11  11
            # PrintLayer________(),
            nn.ELU(),              #out: 20  11  11
            # PrintLayer________(),
            nn.MaxPool2d(2),       #out: 20  5   5
            # PrintLayer________(),
            nn.BatchNorm2d(num_features=20), #out: 20  5  5
            # PrintLayer________("Conv output:\n")
        )

        self.fc = nn.Sequential(
            # PrintLayer________("Fully connected input:\n"),
            nn.Linear(in_features=20*5*5, out_features=32),
            # PrintLayer________(), 
            nn.ELU(1),
            # PrintLayer________(), 
            nn.Linear(in_features=32,out_features=32),
            # PrintLayer________(), 
            nn.ELU(1),
            # PrintLayer________(), 
            nn.Linear(in_features=32,out_features=32),
            # PrintLayer________(), 
            nn.ELU(1),
            # PrintLayer________("Fully connected output:\n") 
        )

        self.final = nn.Linear(32, num_outputs)

    def forward(self, input):
        return input > pipe | self.conv | nn.Flatten() | self.fc | self.final


    def replace_output(self, newLayer):
        # map 32 inputs to custom amount
        self.final = newLayer


def trainLoop(dataloader, epo, optimizer, model, criterion):
    for e in range(epo):
        print(f"epoch {e+1}/{epo}")
        for idx, (data, labels) in enumerate(dataloader.generate()):
            optimizer.zero_grad()
            predict_y = model(data.float())
            loss = criterion(predict_y, labels)
            if idx % 10 == 0:
                print('\rbatch: {}, loss: {}'.format(idx, loss.sum().item()), end="")
            loss.backward()
            optimizer.step()
            if keyboard.is_pressed('s'):
                print("")
                return

class CoolDataLoader():
    def __init__(self, dataLoader, transformationMode = None, batch_size = 256, num_classes=10):
        self.transform = transformationMode
        self.dataLoader = dataLoader
        self.batch_size = 256
        self.num_classes = num_classes

    def generate(self):
        for train_x, train_y in self.dataLoader:
            data = []
            targets = []
            if self.transform is None: 
                yield train_x, train_y

            elif self.transform == "rotate":
                for image in train_x:
                    amount = np.random.randint(0, 4)
                    data.append(torch.rot90(image, amount, dims=(1,2)))
                    targets.append(torch.tensor(amount))

                data = torch.stack(data)
                # targets = F.one_hot(torch.stack(targets), self.num_classes)
                targets = torch.stack(targets)
                yield data, targets