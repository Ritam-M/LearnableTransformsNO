import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import scipy
import time
import sys

np.random.seed(0)
torch.random.seed(0)

class NeuralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dims, modes1, modes2):
        super(NeuralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = dims
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        
        self.M1 = nn.Linear(self.dims[0],self.modes1)
        self.M2 = nn.Linear(self.dims[1],self.modes2)
        
        self.N1 = nn.Linear(self.modes1,self.dims[0])
        self.N2 = nn.Linear(self.modes2,self.dims[1])
        
    def R(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        
        batchsize = x.shape[0]

        x_ft = self.M1(x.permute(0,1,3,2)).permute(0,1,2,3)
        x_ft = self.M2(x_ft.permute(0,1,3,2))
        
        out_ft = self.R(x_ft, self.weights1)
        out_ft = self.R(out_ft, self.weights2)

        x = self.N1(out_ft.permute(0,1,3,2)).permute(0,1,2,3)
        x = self.N2(x.permute(0,1,3,2))
        
        return x
    
class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, dims, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.dims = dims
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = NeuralConv2d(self.width, self.width, self.dims, self.modes1, self.modes2)
        self.conv1 = NeuralConv2d(self.width, self.width, self.dims, self.modes1, self.modes2)
        self.conv2 = NeuralConv2d(self.width, self.width, self.dims, self.modes1, self.modes2)
        self.conv3 = NeuralConv2d(self.width, self.width, self.dims, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # x = F.avg_pool2d(x, kernel_size=2, stride=2)   ## Downsampling
        
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        # x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)    # Upsampling
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, dims, width):
        super(Net2d, self).__init__()
        self.conv1 = SimpleBlock2d(modes, modes, dims, width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

################################################################
# configs
################################################################
PATH = '/kaggle/input/darcy-triangular/Darcy_Triangular_FNO.mat'

ntrain = 1900
ntest = 100

batch_size = 10
learning_rate = 0.001

epochs = 800
step_size = 100
gamma = 0.5

r = 1
h = int(((100 - 1)/r) + 1)
s = h

modes = 8
width = 32

################################################################
# load data and data normalization
################################################################
reader = MatReader(PATH)
x_train = reader.read_field('boundCoeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]
grid_x_train = reader.read_field('coord_x')[:ntrain,::r,::r][:,:s,:s]
grid_y_train = reader.read_field('coord_y')[:ntrain,::r,::r][:,:s,:s]

reader.load_file(PATH)
x_test = reader.read_field('boundCoeff')[ntrain:ntrain+ntest,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[ntrain:ntrain+ntest,::r,::r][:,:s,:s]
grid_x_test = reader.read_field('coord_x')[ntrain:ntrain+ntest,::r,::r][:,:s,:s]
grid_y_test = reader.read_field('coord_y')[ntrain:ntrain+ntest,::r,::r][:,:s,:s]

grid_x_train = grid_x_train.reshape(ntrain, s, s, 1)
grid_y_train = grid_y_train.reshape(ntrain, s, s, 1)
x_train = x_train.reshape(ntrain, s, s, 1)
x_train = torch.cat([x_train, grid_x_train, grid_y_train], dim = -1)

grid_x_test = grid_x_test.reshape(ntest, s, s, 1)
grid_y_test = grid_y_test.reshape(ntest, s, s, 1)
x_test = x_test.reshape(ntest, s, s, 1)
x_test = torch.cat([x_test, grid_x_test, grid_y_test], dim = -1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

train_loader_L2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=ntrain, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = Net2d(modes, dims=[s,s], width).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

start_time = time.time()
myloss = LpLoss(size_average=False)
#y_normalizer.cuda()

train_loss = np.zeros((epochs, 1))
test_loss = np.zeros((epochs, 1))
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        loss.backward()
        optimizer.step()
        train_mse += loss.item()        

    scheduler.step()
    model.eval() 

    train_L2 = 0
    with torch.no_grad():
        for x, y in train_loader_L2:
            x, y = x.cuda(), y.cuda() 
            out = model(x)
            l2 = myloss(out.view(ntrain, -1), y.view(ntrain, -1)) 
            train_L2 += l2.item() 

    test_L2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            test_L2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_L2 /= ntrain
    test_L2 /= ntest
    train_loss[ep,0] = train_L2
    test_loss[ep,0] = test_L2

    t2 = default_timer()

    print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f" % ( ep, t2-t1, train_mse, train_L2, test_L2))

elapsed = time.time() - start_time
print("\n=============================")
print("Training done...")
print('Training time: %.3f'%(elapsed))
print("=============================\n")

# ====================================
# saving settings
# ====================================
current_directory = os.getcwd()
case = "Case_"
folder_index = str(save_index)

results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

np.savetxt(save_results_to+'/train_loss.txt', train_loss)
np.savetxt(save_results_to+'/test_loss.txt', test_loss)

save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)

torch.save(model, save_models_to+'Darcy')
