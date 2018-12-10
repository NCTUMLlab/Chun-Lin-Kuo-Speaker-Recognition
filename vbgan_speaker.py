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

dt = Dataset('./NIST_data_npy/', one_hot=False)
x_dim = dt.train_data.shape[1]
y_dim = dt._n_class
n_batch = dt._n_examples// args.batch_size
print("x_dim", x_dim)
print("y_dim", y_dim)
print("num_speakers",dt._n_examples)
print("num_batch", n_batch)

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.input = args.n_input

        self.enc1 = MLPLayer(self.input, self.dim_h * 2, args.sigma_prior)
        self.bn1 = nn.BatchNorm1d(self.dim_h * 2)
        self.enc1_act = nn.ReLU()
        self.enc2 = MLPLayer(self.dim_h * 2, self.dim_h * 2, args.sigma_prior)
        self.bn2 = nn.BatchNorm1d(self.dim_h * 2)
        self.enc2_act = nn.ReLU()
        self.enc3 = MLPLayer(self.dim_h * 2, self.dim_h * 2, args.sigma_prior)
        self.bn3 = nn.BatchNorm1d(self.dim_h * 2)
        self.enc3_act = nn.ReLU()
        self.enc3_1 = MLPLayer(self.dim_h * 2, self.dim_h * 2, args.sigma_prior)
        self.bn3_1 = nn.BatchNorm1d(self.dim_h * 2)
        self.enc3_1_act = nn.ReLU()
        self.enc4 = MLPLayer(self.dim_h * 2, self.n_z, args.sigma_prior)
        self.enc5 = MLPLayer(self.dim_h * 2, self.n_z, args.sigma_prior)

    def encode(self, x):
        h = self.enc1_act(self.bn1(self.enc1(x)))
        h = self.enc2_act(self.bn2(self.enc2(h)))
        h = self.enc3_act(self.bn3(self.enc3(h)))
        h = self.enc3_1_act(self.bn3_1(self.enc3_1(h)))

        return self.enc4(h), self.enc5(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def get_lpw_lqw(self):
        lpw = self.enc1.lpw + self.enc2.lpw + self.enc3.lpw + self.enc3_1.lpw + self.enc4.lpw + self.enc5.lpw
        lqw = self.enc1.lqw + self.enc2.lqw + self.enc3.lqw + self.enc3_1.lqw + self.enc4.lqw + self.enc5.lqw
        return lpw, lqw


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


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.input = args.n_input

        self.main = nn.Sequential(
            nn.Linear(self.input, self.dim_h * 2),
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
            nn.Linear(self.dim_h * 2, self.dim_h * 2)
        )

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(self.dim_h * 2, 1),
                                       nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(self.dim_h * 2, y_dim),
                                       nn.Softmax())

    def forward(self, x):
        h = self.main(x)
        validity = self.adv_layer(h)
        label = self.aux_layer(h)
        return validity, label


def forward_pass_samples(x, z, gen_labels, real_labels):
    enc_kl, dec_kl, rec_scores, sam_scores = torch.zeros(args.n_mc), torch.zeros(args.n_mc), torch.zeros(
        args.n_mc), torch.zeros(args.n_mc)
    enc_log_likelihoods, dec_log_likelihoods = torch.zeros(args.n_mc), torch.zeros(args.n_mc)
    for i in range(args.n_mc):
        # z = torch.randn(args.batch_size, args.n_z).to(device)  # z~N(0,1) , generate new one?
        # x_sam = decoder(z, gen_labels, 'unsupervised')
        z_enc, mu, log_var = encoder(x)
        x_rec = decoder(z_enc, gen_labels)
        # ==========================================#
        #                  BCE                      #
        # ==========================================#
        # assert ((x_rec >= 0.) & (x_rec <= 1.)).all()
        # reconst_loss = F.binary_cross_entropy(x_rec, x , reduction = 'sum')
        # ===========================================#
        #                  MSE                      #
        # ===========================================#
        reconst_loss = mse(x_rec, x)  #可以考慮cosine error
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # kl p(z) between q(z|x)


        gen_imgs = decoder(z, gen_labels)

        fake_pred, fake_aux = discriminator(gen_imgs)
        g_loss = bce_sum(fake_pred, real_labels) + ce_sum(fake_aux, gen_labels)

        enc_log_pw, enc_log_qw = encoder.get_lpw_lqw()
        dec_log_pw, dec_log_qw = decoder.get_lpw_lqw()
        enc_log_likelihood = reconst_loss + kl_div
        dec_log_likelihood = reconst_loss + (g_loss) * 10

        # print("rec_loss",reconst_loss.item())
        # print("kl_div",kl_div.item())
        # print("g_loss",g_loss.item())
        enc_kl[i] = enc_log_qw - enc_log_pw
        dec_kl[i] = dec_log_qw - dec_log_pw
        enc_log_likelihoods[i] = enc_log_likelihood
        dec_log_likelihoods[i] = dec_log_likelihood
        # rec_scores[i] = pred_rec.mean()
        # sam_scores[i] = pred_sam.mean()

    return enc_kl.mean(), dec_kl.mean(), enc_log_likelihoods.mean(), dec_log_likelihoods.mean()  # , rec_scores.mean(), sam_scores.mean()


def reset_grad():
    dis_optimizer.zero_grad()
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def criterion(kl, log_likelihood):
    return (kl / n_batch + log_likelihood) /args.batch_size


def criterion_reW(kl, i, log_likelihood):
    M = n_batch
    weight = (2 ^ (M - i)) / (2 ^ M - 1)
    # print("kl", kl.item())
    # print("loglikelihood", log_likelihood.item())
    return (kl * weight) / M + log_likelihood


encoder = Encoder(args).to(device)
decoder = Decoder(args).to(device)
discriminator = Discriminator(args).to(device)
enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.5 * args.lr, betas=(0.5, 0.999))

bcewl = nn.BCEWithLogitsLoss(reduction='sum')
bce_sum = nn.BCELoss(reduction='sum')
bce = nn.BCELoss()
mse = nn.MSELoss(reduction='sum')
ce_sum = nn.CrossEntropyLoss(reduction='sum')
ce = nn.CrossEntropyLoss()

# Start training
encoder.train()
decoder.train()
discriminator.train()


for epoch in range(args.epochs):
    x, y = dt.next_batch(args.batch_size)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y)

    y = y.type(torch.LongTensor).to(device)
    #x = x.to(device).view(-1, args.n_input)
    z = torch.randn(args.batch_size, args.n_z).to(device)  # z~N(0,1)
    gen_labels = torch.LongTensor(np.random.randint(0, y_dim, args.batch_size)).to(device)

    real_labels = torch.Tensor(args.batch_size, 1).to(device).fill_(1.0)
    fake_labels = torch.Tensor(args.batch_size, 1).to(device).fill_(0.0)

    # ================================================================== #
    #                        Train the generator                         #
    # ================================================================== #
    free_params(decoder)
    free_params(encoder)
    frozen_params(discriminator)
    enc_kl, dec_kl, enc_log_likelihood, dec_log_likelihood = forward_pass_samples(x, z, gen_labels, real_labels)
    enc_loss = criterion(enc_kl, enc_log_likelihood)
    dec_loss = criterion(dec_kl, dec_log_likelihood)

    reset_grad()
    enc_loss.backward(retain_graph=True)
    enc_optimizer.step()

    reset_grad()
    dec_loss.backward(retain_graph=True)
    dec_optimizer.step()



    # ================================================================== #
    #                      Train the discriminator                       #
    # ================================================================== #
    frozen_params(decoder)
    frozen_params(encoder)
    free_params(discriminator)

    real_pred, real_aux = discriminator(x)

    d_loss_real = bce(real_pred, real_labels) + ce(real_aux, y)

    gen_imgs = decoder(z, gen_labels)

    fake_pred, fake_aux = discriminator(gen_imgs.detach())

    d_loss_fake = bce(fake_pred, fake_labels) + ce(fake_aux, gen_labels)

    d_loss = (d_loss_real + d_loss_fake)

    reset_grad()
    d_loss.backward()
    dis_optimizer.step()
    # print("fake",fake_score.mean().item())
    # print("real",real_score.mean().item())

    if (epoch + 1) % 1 == 0:
        print("Epoch[{}/{}], enc_Loss: {:.4f} ,dec_Loss: {:.4f}, d_Loss: {:.4f}"
              .format(epoch + 1, args.epochs, enc_loss.item(), dec_loss.item(), d_loss.item()))


# Save the model checkpoints
torch.save(encoder.state_dict(), check_point_dir + '/encdoer_%d.pkl' %(epoch + 1))
torch.save(decoder.state_dict(), check_point_dir + '/decoder_%d.pkl' %(epoch + 1))
torch.save(discriminator.state_dict(), check_point_dir + '/discriminator_%d.pkl' %(epoch + 1))
# torch.save(encoder.state_dict(), './' + check_point_dir + '/encoder.ckpt')
# torch.save(decoder.state_dict(), './' + check_point_dir + '/decoder.ckpt')
# torch.save(discriminator.state_dict(), './' + check_point_dir + '/discriminator.ckpt')

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
    encoder.eval()
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




