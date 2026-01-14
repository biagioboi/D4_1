import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset   
torch.backends.cudnn.benchmark=True

batch_size = 32
epochs = 5

# Image augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Loading CIFAR10 using torchvision.datasets
traindata = datasets.CIFAR10('./data', train=True, download=True,
                       transform= transform_train)

# Dividing the training data into num_clients, with each client having equal number of images
traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / 20) for _ in range(20)])

# Creating a pytorch loader for a Deep Learning model
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

# Normalizing the test images
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Loading the test iamges and thus converting them into a test_loader
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        ), batch_size=batch_size, shuffle=True)

#################################
##### Neural Network model #####
#################################

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, vgg_name):
      super(VGG, self).__init__()
      self.features = self._make_layers(cfg[vgg_name])
      self.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, 10)
      )      

    def forward(self, x):
      out = self.features(x)
      out = out.view(out.size(0), -1)
      out = self.classifier(out)
      output = F.log_softmax(out, dim=1)
      return output

    def _make_layers(self, cfg):
      layers = []
      in_channels = 3
      for x in cfg:
        if x == 'M':
          layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
          layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                     nn.BatchNorm2d(x),
                     nn.ReLU(inplace=True)]
          in_channels = x
      layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
      return nn.Sequential(*layers)

def client_update(client_model, optimizer, train_loader, epoch=5):
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def aggregate(global_model, client_models):
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
      global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
      model.load_state_dict(global_model.state_dict())


def test(global_model, test_loader):
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

import torch
print(torch.cuda.is_available())
# Output would be True if Pytorch is using GPU otherwise it would be False.

global_model =  VGG('VGG19').cuda()

client_model = VGG('VGG19').cuda()
opt = optim.SGD(client_model.parameters(), lr=0.1)

###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []
acc_rounds = np.empty(0)

# Runnining FL
client_idx = np.random.permutation(20)[:6]
loss = 0
for i in tqdm(range(6)):
    loss += client_update(client_model, opt, train_loader[client_idx[i]], epoch=epochs)
    print('%d-th round' % i)
    print("loss: ",loss)
    losses_train.append(loss)
    test_loss, acc = test(client_model, test_loader)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / 1, test_loss, acc))

folder_name = 'models_dict'
os.makedirs('models_dict', exist_ok=True)

file_name = "client_model_3_VGG19"
file_path: str = os.path.join(folder_name, file_name)
torch.save(client_model.state_dict(), file_path)

client_models = [ VGG('VGG19').cuda() for _ in range(len((os.listdir(folder_name))))]
for file_name, model in zip(os.listdir(folder_name),client_models):
    model.load_state_dict(torch.load(os.path.join(folder_name,file_name)))

