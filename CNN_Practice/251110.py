# 251110.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo 
import torchvision.transforms as transforms
from torchviz import make_dot
from tqdm.auto import tqdm

# # Drop out
# torch.manual_seed(123)
# inputs = torch.randn(1, 10)
# print("Original Inputs:")
# print(inputs)

# # Definition Drop Out Function
# dropout = nn.Dropout(0.5)

# # Training Phase Motion
# dropout.train() # training mode
# print(f"Is in training mode? : {dropout.training}")
# outputs = dropout(inputs)
# print("Outputs in Training mode: ")
# print(outputs)

# # Prediction Phase Motion
# dropout.eval() # evaluation mode # DropOut Inactivation
# print(f"Is in training mode : {dropout.training}")
# outputs = dropout(inputs)
# print("Outputs in Eval Mode:")
# print(outputs) 

print("="*100)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Data Augmentation
# Activating Data Augmentation using Transforms
# Training Data : Normalization, Transpos, RandomErasing
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio = (0.3, 3.3), value=0, inplace=True)])
# print(transform_train)

from pythonlibs.torch_lib1 import *
# print(README)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# Data Loading Function 

data_root = './data'

train_set = datasets.CIFAR10(root = data_root, train = True, download=True, transform=transform)
test_set = datasets.CIFAR10(root = data_root, train = False, download=True, transform=transform)

batch_size = 100

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

classes = train_set.classes
# show_images_labels(test_loader, classes, None, None) # None, None : model, device

n_output = len(list(set(classes))) # the total number of classes the model needs to classify

class CNN_v2(nn.Module):
    def __init__(self, num_classes):    
        super().__init__()
        # Convolution, ReLU, MaxPool layer definition
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))

        # Fully Connected Layer Defition
        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, num_classes)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool,

            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.maxpool,

            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.maxpool
        )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )
    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3

# device = "cuda" if torch.cuda.is_available() else "cpu"
# net = CNN_v2(n_output).to(device)
# criterion = nn.CrossEntropyLoss()
# loss = eval_loss(test_loader, device, net, criterion)
# g = make_dot(loss, params=dict(net.named_parameters()))
# g.render("cnn_graph", format="png", cleanup=True)

torch_seed()
lr = 0.01
net = CNN_v2(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
history = np.zeros((0,5))

num_epochs = 50
history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)