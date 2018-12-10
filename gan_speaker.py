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
parser.add_argument('-dim_h', type=int, default=250, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=100, help='hidden dimension of z (default: 8)')
parser.add_argument('-sigma_prior', type=float, default=torch.tensor(np.exp(-3)).to(device))
parser.add_argument('-n_mc', type=int, default=5)
parser.add_argument('-n_input', type=int, default=600)

args = parser.parse_args()

dt = Dataset('./NIST_data_npy/', one_hot=True)
x_dim = dt.train_data.shape[1]
y_dim = dt._n_class
n_batch = dt._n_examples// args.batch_size
print("x_dim", x_dim)
print("y_dim", y_dim)
print("num_speakers",dt._n_examples)
print("num_batch", n_batch)



class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.output = args.n_input
        self.label_emb = nn.Embedding(y_dim, args.n_z)

        self.main = nn.Sequential(
            nn.Linear(self.n_z + dt._n_class, self.dim_h * 2),
            # nn.BatchNorm1d(self.dim_h * 2),
            nn.ReLU(),
            nn.Linear(self.dim_h * 2, self.dim_h * 2),
            # nn.BatchNorm1d(self.dim_h * 2),
            nn.ReLU(),
            nn.Linear(self.dim_h * 2, self.dim_h * 2),
            # nn.BatchNorm1d(self.dim_h * 2),
            nn.ReLU(),
            nn.Linear(self.dim_h * 2, self.dim_h * 2),
            nn.ReLU(),
            nn.Linear(self.dim_h * 2, self.output)
        )


    def forward(self, z, labels):
        z_emb = torch.cat([z, labels], 1)
        #z_emb = torch.mul(self.label_emb(labels), z)
        x = self.main(z_emb)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.input = args.n_input

        self.main = nn.Sequential(
            nn.Linear(self.input + dt._n_class , self.dim_h * 2),
            # nn.BatchNorm1d(self.dim_h * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h * 2, self.dim_h * 2),
            # nn.BatchNorm1d(self.dim_h * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h * 2, self.dim_h * 2),
            #nn.BatchNorm1d(self.dim_h * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h * 2, self.dim_h * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h * 2, 1),
            nn.Sigmoid()
        )


    def forward(self, x , label):
        h = torch.cat([x,label],1)
        h = self.main(h)
        return h



def reset_grad():
    dis_optimizer.zero_grad()
    dec_optimizer.zero_grad()


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False



decoder = Decoder(args).to(device)
discriminator = Discriminator(args).to(device)
dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.5 * args.lr, betas=(0.5, 0.999))

bcewl = nn.BCEWithLogitsLoss(reduction='sum')
bce = nn.BCELoss(reduction='sum')
mse = nn.MSELoss(reduction='sum')
ce = nn.CrossEntropyLoss(reduction='sum')

# Start training

decoder.train()
discriminator.train()


for epoch in range(args.epochs):
    x, y = dt.next_batch(args.batch_size)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y)
    #print(y.size())
    y = y.type(torch.FloatTensor).to(device)
    #x = x.to(device).view(-1, args.n_input)
    z = torch.randn(args.batch_size, args.n_z).to(device)  # z~N(0,1)
    #print(z.size())
    #gen_labels = torch.FloatTensor(np.random.randint(0, y_dim, args.batch_size)).view(args.batch_size,1).to(device)
    #gen_labels.view(args.batch_size,1)
    #print(gen_labels.size())

    real_labels = torch.Tensor(args.batch_size, 1).to(device).fill_(1.0)
    fake_labels = torch.Tensor(args.batch_size, 1).to(device).fill_(0.0)

    # ================================================================== #
    #                        Train the generator                         #
    # ================================================================== #
    free_params(decoder)
    frozen_params(discriminator)
    gen_imgs = decoder(z,y)

    fake_pred = discriminator(gen_imgs,y)
    dec_loss = bce(fake_pred, real_labels)


    reset_grad()
    dec_loss.backward()
    dec_optimizer.step()



    # ================================================================== #
    #                      Train the discriminator                       #
    # ================================================================== #
    frozen_params(decoder)
    free_params(discriminator)



    real_pred= discriminator(x, y)

    d_loss_real = bce(real_pred, real_labels)

    gen_imgs = decoder(z, y)

    fake_pred = discriminator(gen_imgs.detach(),y)

    d_loss_fake = bce(fake_pred, fake_labels)

    d_loss = d_loss_fake + d_loss_real

    reset_grad()
    d_loss.backward()
    dis_optimizer.step()
    # print("fake",fake_score.mean().item())
    # print("real",real_score.mean().item())

    if (epoch + 1) % 1 == 0:
        print("Epoch[{}/{}] ,dec_Loss: {:.4f}, d_Loss: {:.4f}"
              .format(epoch + 1, args.epochs, dec_loss.item(), d_loss.item()))

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
n = 2  #each calss at least have n
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

y_tensor = y_tensor.type(torch.FloatTensor).to(device)  #not one-hot
# print(y[0:20])
# print(len(idx), len(total), gen_sum, len(y))

one_hot = np.zeros([gen_sum, y_dim], dtype=np.int)
one_hot[np.arange(len(y)), y] = 1


with torch.no_grad():
    decoder.eval()
    z = torch.randn(len(y_tensor), args.n_z).to(device)
    #gen_labels = torch.LongTensor(np.random.randint(0, y_dim, args.batch_size)).to(device)
    one_hot= torch.from_numpy(one_hot)
    one_hot = one_hot.type(torch.FloatTensor).to(device)
    gen_x = decoder(z, one_hot)
    np.save('Gen_Data/gan_wc_x_' + str(n) + '.npy', gen_x.cpu().numpy())
    np.save('Gen_Data/gan_wc_y_' + str(n) + '.npy', y_tensor)
    #print(np.shape(y))
    dt = Dataset('./NIST_data_npy/', one_hot=False)

    dt.add_training_data(gen_x, y)
    dt.sort()
    mat_x = dt.train_data
    mat_y = dt.train_label + 1
    #print(mat_y[0:10])
    train_x = {'x': mat_x}
    train_y = {'y': mat_y}
    savemat(mat_path + '/gan_wc_x_gd' + str(n) + '.mat', train_x)
    savemat(mat_path + '/gan_wc_y_gd' + str(n) + '.mat', train_y)



        # # Save the model checkpoints
        # torch.save(encoder.state_dict(), './' + check_point_dir + '/encoder.ckpt')
        # torch.save(decoder.state_dict(), './' + check_point_dir + '/decoder.ckpt')
        # torch.save(discriminator.state_dict(), './' + check_point_dir + '/discriminator.ckpt')
