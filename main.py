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
from devp.fabric_integration import bc_interaction_digibank as digibank
torch.backends.cudnn.benchmark=True

num_clients = 20
num_selected = 6
num_rounds = 1
epochs = 5
batch_size = 32

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
traindata_split = torch.utils.data.random_split(traindata, [int(traindata.data.shape[0] / num_clients) for _ in range(num_clients)])

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

def client_update(model, optimizer, train_loader, epochs, device):
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
    return loss.item()

def server_aggregate(global_model, client_models):
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
      global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
      model.load_state_dict(global_model.state_dict())

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for data, target in loader:
            data= data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

import torch
torch.cuda.is_available()
# Output would be True if Pytorch is using GPU otherwise it would be False.

def generate_random_indices(num_clients, num_selected, num_iterations):
    """
    Genera un array con gli indici dei clienti selezionati casualmente per un numero specificato di volte.
    
    Args:
        num_clients (int): il numero totale di clienti.
        num_selected (int): il numero di clienti da selezionare casualmente.
        num_iterations (int): il numero di volte che si vuole selezionare casualmente un insieme di clienti.
    
    Returns:
        numpy.ndarray: un array con gli indici dei clienti selezionati casualmente per un numero specificato di volte.
    """
    indices = []
    for _ in range(num_iterations):
        indices.append(np.random.permutation(num_clients)[:num_selected])
    return np.array(indices)
client_idx = generate_random_indices(num_clients, num_selected, num_rounds)

global_model =  VGG('VGG19')
import torch

device = torch.device("cpu")   # forzato
torch.set_default_device("cpu")  # opzionale (PyTorch >=2.0), aiuta a non “scappare” su cuda
global_model = global_model.to(device)
client_models = [ VGG('VGG19') for _ in range(num_clients)]
client_models = [m.to(device) for m in client_models]

folder = 'models_dict'
os.makedirs('models_dict', exist_ok=True)

###### List containing info about learning #########
losses_train = []
losses_test = []
###
acc_train = []
acc_test = []
###
acc_rounds = np.empty(0)

###### List containing info about learning with attacked and handled model#########
a_losses_train = []
a_losses_test = []
###
a_acc_train = []
a_acc_test = []
###
a_acc_rounds = np.empty(0)

no_rollback_losses_train = []
no_rollback_losses_test = []
###
no_rollback_acc_train = []
no_rollback_acc_test = []
###
no_rollback_acc = np.empty(0)

############## client models ##############
#Questo fa si che ogni client model abbia gli stessi pesi dell'attuale global model (in questo caso vuoto)

for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model 

############### optimizers ################
opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]
torch.save(global_model.state_dict(),os.path.join(folder,"global_model"))
cids_list = digibank.addModel("bia1",os.path.join(folder, 'global_model'))
import devp.ipfs_files_util as ipfs
from devp.fabric_integration import bc_interaction_digibank as digibank
loss = 0
cids_list = []
dest_folder = "client_models_retrived"
dest_folder_global_model = "global_model_retrived"
#addestramento del modello gestito con delta = 0.08
delta = 0.08
copy_loss = 0
copy_acc = 0
nome_modello = "modello_gestito"
global_copy =  VGG('VGG19')
for r in range(num_rounds):
    cids_list = []
    cids_list_retrived = []
    loss = 0
    
    for i in tqdm(range(num_selected)):
        loss += client_update(client_models[i], opt[i], train_loader[client_idx[r][i]], epochs, device)

        #print("loss: ", client_loss)
        #torch.save(list_models[i].state_dict(), os.path.join(folder, 'client_model_%d' % (i+1)))
    #save all client models
    number_models = 0
    for model in client_models:
        torch.save(model.state_dict(), os.path.join(folder, 'client_model_%d' % (number_models+1)))
        number_models += 1


    cids_list = digibank.addModels("bia1",folder)
    cids_list_retrived = digibank.getModelsByType("bia1", "client")
    ipfs.retriveModels(cids_list_retrived, dest_folder)
    digibank.deleteAssetsFromCidsList("bia1",cids_list_retrived)
    for model, filename in zip(client_models, os.listdir(dest_folder)):
        state = torch.load(os.path.join(dest_folder_global_model, "global_model"), map_location=device)
        global_copy.load_state_dict(state)
        global_copy.to(device)
        model.load_state_dict(state)
        model.to(device)
        os.remove(os.path.join(dest_folder, filename))

    
    #prendo il modello globale precedente dalla blockchain lo salvo in una cartella
    # e lo carico
    
    global_cid = digibank.getModelsByType("bia1", "global")
    ipfs.retrieve_file(global_cid[0], dest_folder_global_model+"/global_model")
    global_copy.load_state_dict(torch.load(os.path.join(dest_folder_global_model, "global_model")))



    server_aggregate(global_model, client_models)

    #valuto i due modelli
    test_loss, acc = test(global_model, test_loader, device)
    copy_loss, copy_acc = test(global_copy, test_loader, device)

    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))
    print('copy model test loss %0.3g | test acc: %0.3f' % (copy_loss, copy_acc))

    # FOR ROLLBACK
    # if(acc < copy_acc and  copy_acc - acc > delta):
    #   print(" \n #### ROLLBACK #### \n")
    #   print("acc : ",acc)
    #   print("copy_acc : ",copy_acc)
    #   model_rollback(global_copy,global_model,client_models)


    test_loss, acc = test(global_model, test_loader)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))
    
    a_losses_test.append(test_loss)
    a_acc_test.append(acc)

    #cancello il precedente global model
    digibank.deleteAsset("bia1", global_cid)

    #salvo il modello globale
    torch.save(global_model.state_dict(), os.path.join(folder, 'global_model'))
    #lo carico in ipfs
    cids_list = digibank.addModel("bia1",os.path.join(folder, 'global_model'),accuracy=acc)


    a_acc_rounds = np.append(a_acc_rounds, 100.*acc)
