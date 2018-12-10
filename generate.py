import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from MLP_Layer import MLPLayer
from torch.autograd import Variable
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# import torchvision.datasets as dset
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataset import Dataset
from scipy.io import savemat

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
gen_dir = 'Gen_Data'
check_point_dir = 'check_point'
mat_path = 'save_mat'
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)
if not os.path.exists(mat_path):
    os.makedirs(mat_path)

# Hyper-parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST semi-supervised')
parser.add_argument('-batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=10000, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default= 1e-3, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=150, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=100, help='hidden dimension of z (default: 8)')
parser.add_argument('-sigma_prior', type=float, default=torch.tensor(np.exp(-3)).to(device))
parser.add_argument('-n_mc', type=int, default=5)
parser.add_argument('-n_input', type=int, default=600)

args = parser.parse_args()



class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.output = args.n_input

        self.label_emb = nn.Embedding(y_dim, args.n_z)

        self.dec1 = MLPLayer(self.n_z, self.dim_h * 2, args.sigma_prior)
        self.bn1 = nn.BatchNorm1d(self.dim_h * 2)
        self.dec1_act = nn.ReLU()
        self.dec2 = MLPLayer(self.dim_h * 2, self.dim_h * 2, args.sigma_prior)
        self.bn2 = nn.BatchNorm1d(self.dim_h * 2)
        self.dec2_act = nn.ReLU()
        self.dec3 = MLPLayer(self.dim_h * 2, self.dim_h * 2, args.sigma_prior)
        self.bn3 = nn.BatchNorm1d(self.dim_h * 2)
        self.dec3_act = nn.ReLU()
        self.dec3_1 = MLPLayer(self.dim_h * 2, self.dim_h * 2, args.sigma_prior)
        self.bn3_1 = nn.BatchNorm1d(self.dim_h * 2)
        self.dec3_1_act = nn.ReLU()
        self.dec4 = MLPLayer(self.dim_h * 2, self.output, args.sigma_prior)
        # self.bn4 = nn.BatchNorm1d(self.output)
        self.dec4_act = nn.Tanh()

    def decode(self, z, labels):
        z_emb = torch.mul(self.label_emb(labels), z)
        h = self.dec1_act(self.bn1(self.dec1(z_emb)))
        h = self.dec2_act(self.bn2(self.dec2(h)))
        h = self.dec3_act(self.bn3(self.dec3(h)))
        h = self.dec3_1_act(self.bn3_1(self.dec3_1(h)))
        return self.dec4(h)  # i-vector no need tanh

    def forward(self, z, labels):
        x = self.decode(z, labels)
        return x

    def get_lpw_lqw(self):
        lpw = self.dec1.lpw + self.dec2.lpw + self.dec3.lpw + self.dec3_1.lpw + self.dec4.lpw
        lqw = self.dec1.lqw + self.dec2.lqw + self.dec3.lqw + self.dec3_1.lqw + self.dec4.lqw
        return lpw, lqw





decoder = Decoder(args).to(device)




decoder.load_state_dict(torch.load('decoder_20.pkl'))


def sample_z(m, n):
    return np.random.normal(loc=0., scale=1., size=[m, n])
condit = []
speakers = []
for i in range(2, 11, 1):
    times = 1
    x = np.where(dt.bincout < i*times)[0]
    tmp1 = 'classes less than ' + str(i*times)
    tmp2 = x.shape[0]
    condit.append(tmp1)
    speakers.append(tmp2)
    print(tmp2, tmp1)
n = 5  #each calss at least have n
count = n - dt.bincout
idx = np.where(count > 0)[0]
total = count[idx]
gen_sum = np.sum(total)
y = []
for i in range(len(idx)):
    for j in range(total[i]):
        y.append(idx[i])
y = np.asarray(y).astype(np.int)

y_tensor = torch.from_numpy(y)

y_tensor = y_tensor.type(torch.LongTensor).to(device)  #not one-hot
# print(y[0:20])
# print(len(idx), len(total), gen_sum, len(y))

# one_hot = np.zeros([gen_sum, y_dim], dtype=np.int)
# one_hot[np.arange(len(y)), y] = 1


with torch.no_grad():
    decoder.eval()
    z = torch.randn(len(y_tensor), args.n_z).to(device)
    #gen_labels = torch.LongTensor(np.random.randint(0, y_dim, args.batch_size)).to(device)
    # one_hot = torch.from_numpy(one_hot)
    # one_hot = one_hot.type(torch.LongTensor).to(device)
    gen_x = decoder(z, y_tensor)
    np.save('Gen_Data/acgan_wc_x_' + str(n) + '.npy', gen_x.cpu().numpy())
    np.save('Gen_Data/acgan_wc_y_' + str(n) + '.npy', y_tensor)
    dt.add_training_data(gen_x, y)
    dt.sort()
    mat_x = dt.train_data
    mat_y = dt.train_label + 1
    print(mat_y[0:10])
    train_x = {'x': mat_x}
    train_y = {'y': mat_y}
    savemat(mat_path + '/acgan_wc_x_gd_deep' + str(n) + '.mat', train_x)
    savemat(mat_path + '/acgan_wc_y_gd_deep' + str(n) + '.mat', train_y)




