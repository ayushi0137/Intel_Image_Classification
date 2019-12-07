from __future__ import print_function, division
import os
import torch
import torchvision
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


def get_train_data(datadir):
    train_transforms = transforms.Compose([transforms.Resize((224, 224), interpolation=2),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ])
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    # num_train = len(train_data)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    return trainloader


def get_test_data(datadir):
    test_transforms = transforms.Compose([transforms.Resize((224, 224), interpolation=2),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ])
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    print(test_data)
    num_test = len(test_data)
    print(num_test)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2)
    return testloader


trainloader = get_train_data('D:/Downloads/seg_train/seg_train')
# print(trainloader)

classes = ('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street')
print(trainloader.dataset.classes)

testloader = get_test_data('D:/Downloads/seg_test/seg_test')


# print(testloader)
# print(testloader.dataset.classes)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(1)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


class Net(nn.Module):  # custom architecture
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        # self.conv5 = nn.Conv2d(128, 64, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 6)

        # self.fc4 = nn.Linear(900, 200)
        # self.fc5 = nn.Linear(200, 60)
        # self.fc6 = nn.Linear(60, 6)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv4(x)))
        # print(x.shape)
        # x = self.pool(F.relu(self.conv5(x)))
        # print(x.shape)
        x = x.view(-1, 128 * 4 * 4)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = self.fc6(x)
        return x


net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # print(inputs.shape)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# Now training on resnet18 model
resnet18 = models.resnet18(pretrained=True)

for param in resnet18.parameters():
    param.requires_grad = False

resnet18.fc = nn.Sequential(nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(256, 6),
                            nn.LogSoftmax(dim=1))

resnet18.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(resnet18.parameters(), lr = 0.001, momentum = 0.9)
optimizer = optim.Adam(resnet18.parameters())

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = resnet18.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = 'C:/Users/tunes/Aandom/ayushaa/intel_resnet18_net.pth'
torch.save(resnet18.state_dict(), PATH)

# Loading the trained resnet18 model
resnet18.load_state_dict(torch.load(PATH))

correct = 0
total = 0

with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs_test, labels_test = data[0].to(device), data[1].to(device)
        output_test = resnet18(inputs_test)
        _, predicted = torch.max(output_test.data, 1)
        total += labels_test.size(0)
        correct += (predicted == labels_test).sum().item()

print(len(inputs_test))
print(len(output_test))
print(len(predicted))
print(total)
print(correct)
print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
