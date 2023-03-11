import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from LeNet import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)

val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
val_loader = DataLoader(val_set, batch_size=5000, shuffle=False, num_workers=0)

LeNet = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5):
    epoch_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data

        outputs = LeNet(inputs)

        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

print('Finished Training')

save_path = './ckpt.pth'
torch.save(net.state_dict(), save_path)
