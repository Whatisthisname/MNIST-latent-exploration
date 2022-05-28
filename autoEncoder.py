from msilib.schema import RadioButton
from time import sleep
from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
import keyboard
from matplotlib.widgets import Slider, Button, RadioButtons

import warnings
warnings.filterwarnings("ignore")

# get the data
train_dataset = mnist.MNIST(root='./ass3/train', train=True, transform=ToTensor(), download=True)

# put data into dataloader:
train_loader = DataLoader(train_dataset, batch_size=256)

# make a custom dataloader
class SelfSupervisionDataLoader(DataLoader):
    
    def __init__(self, loader : DataLoader, transform=None, num_classes=10):
        """If transform is not specified, the data is returned as is (for supervised learning).
Otherwise, the data is transformed, and the labels are the original data."""
        self.loader = loader
        self.transform = transform
        self.num_classes = num_classes
    
    def __iter__(self):
        for batch in self.loader:
            data, labels = batch
            # labels = torch.eye(self.num_classes)[labels]
            if self.transform is not None:
                labels = data.clone().double()
                data = self.transform(data)
            yield data, labels

# function to add noise to the data
def addNoise(data, noise_level = 0.3):
    noise = np.random.normal(0, noise_level, size=data.shape)
    # clamp values of data to be between 0 and 1
    data = np.clip(data + noise, 0, 1)
    return data

class PrintShape(nn.Module):
    def __init__(self, message):
        super().__init__()
        self.mes = message
    def forward(self, x):
        print(self.mes, x.shape)
        return x

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x : torch.Tensor):
        return x.view((-1, *self.shape))

class Downsample(nn.Module):
    def __init__(self, ratio):
        super(Downsample, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        x = torch.nn.functional.interpolate(input = x, scale_factor=self.ratio, mode='bilinear', recompute_scale_factor=True)
        return x

class Encoder(nn.Module):
    def __init__(self, latent_size:int):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            Downsample(0.5),
            # PrintShape("after downsample:"),
            nn.Dropout2d(0.2),
            nn.Conv2d(1, 20, kernel_size=5), # input: 5x14x14 output: 5x10x10
            nn.ELU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, 2),              # input: 5x10x10 output: 5x5x5 
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=20*5*5, out_features=64),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, latent_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_size:int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 20*5*5),
            nn.ELU(),
            nn.BatchNorm1d(20*5*5),
            nn.Dropout(0.5),
            nn.Linear(20*5*5, 10*5*5),
            nn.ELU(),
            nn.BatchNorm1d(10*5*5),
            nn.Dropout(0.5),
            nn.Linear(10*5*5, 3*14*14),
            nn.ELU(),
            nn.BatchNorm1d(3*14*14),
            nn.Linear(3*14*14, 1*20*20),
            nn.ELU(),
            nn.BatchNorm1d(1*20*20),
            View((1, 20, 20)),
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_latent_space(self, x):
        return self.encoder(x)

model = Autoencoder(4).double()

train = False


if not train:
    # load model to new AutoEncoder class
    model.load_state_dict(torch.load("./ass3/autoencoder.pt"))

if train:
    if True:
        model.load_state_dict(torch.load("./ass3/autoencoder.pt"))

    epochs = 10
    learning_rate = 0.01
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    customLoader = SelfSupervisionDataLoader(train_loader, addNoise)


    stop_training = False
    for epoch in range(epochs):
        if stop_training: break
        print(f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (data, labels) in enumerate(customLoader):
            
            reconstruction = model(data)
            loss = criterion(reconstruction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Loss: {loss}")

            # if keyboard s then stop training:
            if keyboard.is_pressed('s'):
                print("Stopping training")
                sleep(0.5)
                stop_training = True
                break

        # save model to this directory after each epoch
        torch.save(model.state_dict(), "./ass3/autoencoder.pt")
        

# get digits from test set:
test_dataset = mnist.MNIST(root='./ass3/test', train=False, transform=ToTensor(), download=True)
# put it into dataloader
test_loader = DataLoader(test_dataset, batch_size=256)
# put it into supervised dataloader
test_loader = SelfSupervisionDataLoader(test_loader, transform=addNoise)


showcase_encoder_decoder = False
if showcase_encoder_decoder:
    # get first batch of data
    data, labels = next(iter(test_loader))
    # put model into test mode
    model.eval()
    # select 8 random indices from 0 to 255:
    indices = np.random.choice(range(256), 8, replace=False)
    # reconstruction = model(data[indices])
    reconstruction = model(labels[indices])
    # show pairwise originals (labels) and reconstructions
    for i in range(8):
        plt.subplot(3, 8, i+1)
        plt.imshow(data[indices][i].detach().numpy().reshape(28, 28), cmap='gray')
        plt.title(f"Noise")
        plt.subplot(3, 8, i+8+1)
        plt.imshow(reconstruction[i].detach().numpy().reshape(28, 28), cmap='gray')
        plt.title(f"Reconstructed")
        plt.subplot(3, 8, i+8+8+1)
        plt.imshow(labels[indices][i].detach().numpy().reshape(28, 28), cmap='gray')
        plt.title(f"True")
    plt.show()


do_pca_exploration = True
if do_pca_exploration:

    test_loader = SelfSupervisionDataLoader(test_loader, transform=None)

    model.eval()
    encodings = []
    print("Encoding digits...")
    for idx, (data, labels) in enumerate(test_loader):
        encodings.append(model.get_latent_space(data))

    encodings = torch.cat(encodings).detach().numpy()

    pca = PCA(n_components=4, whiten=True).fit(encodings)
    transformed = pca.transform(encodings)
    
    print("Variance explained by new axes:")
    print(pca.explained_variance_ratio_)


    fig, ((genAxes, mapAxes), (_1, slider1Axes), (_2, slider2Axes)) = plt.subplots(3, 2, gridspec_kw={'height_ratios': [1, 0.1, 0.1]})
    _1.axis("off")
    _2.axis("off")

    # make a pyplot with a slider
    def update_plot(_):

        decoderInput = np.array([xslider.val, yslider.val, 0, 0]).astype(np.float64).reshape(-1, 4)
        decoderInput = pca.inverse_transform(decoderInput)
        decoderInput = torch.tensor(decoderInput).double()

        # print(decoderInput)
        image = model.decoder(decoderInput).detach().numpy()
        image = image.reshape(28, 28)
        genAxes.clear()
        genAxes.imshow(image, cmap='gray')
        genAxes.set_title("Decoded image")
        
        mapAxes.clear()
        mapAxes.scatter(transformed[:, 0], transformed[:, 1], c=test_dataset.targets.numpy(), cmap = "tab10", s=0.3)
        mapAxes.scatter(xslider.val, yslider.val, c = "red", s=50, label="Your position")
        mapAxes.set_title(f"PCA decomposition")
        mapAxes.set_xlabel("First principal axis")
        mapAxes.set_ylabel("Seconds principal axis")
        mapAxes.axis("equal")
        mapAxes.legend()
        plt.show()
        plt.close(fig)

        

    # make a slider
    xslider = Slider(slider1Axes, 'X slider', -3, 3, valinit=0)
    xslider.on_changed(update_plot)
    yslider = Slider(slider2Axes, 'Y slider', -3, 3, valinit=0)
    yslider.on_changed(update_plot)
    update_plot(0)
    plt.show()
