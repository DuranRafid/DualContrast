import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image


class Model(nn.Module):
    def __init__(self, z_dim, x_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim, x_dim)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        z_mu, z_log_var, c_mu, c_log_var = self.encoder(x)
        z = self.sampling(z_mu, z_log_var)
        c = self.sampling(c_mu, c_log_var)
        return self.decoder(z, c), z_mu, z_log_var, c_mu, c_log_var

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = self.my_conv_layer(1, 16)
        self.conv2 = self.my_conv_layer(16, 32)
        self.conv3 = self.my_conv_layer(32, 48)
        self.conv4 = self.my_conv_layer(48, 64)
        self.fc_1 = nn.Linear(576, z_dim)
        self.fc_2 = nn.Linear(576, z_dim)
        self.fc_3 = nn.Linear(576, z_dim)
        self.fc_4 = nn.Linear(576, z_dim)
        self.lt = z_dim

    def my_conv_layer(self, in_f, out_f):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=5, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(num_features=out_f),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
        return self.fc_1(x), self.fc_2(x), self.fc_3(x), self.fc_4(x)

class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim1=256, h_dim2=512):
        super(Decoder, self).__init__()
        self.fc1_1 = nn.Linear(z_dim, h_dim1)
        self.fc1_2 = nn.Linear(z_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1 * 2, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim2)
        self.fc4 = nn.Linear(h_dim2, h_dim2)
        self.fc5 = nn.Linear(h_dim2, x_dim * x_dim)
        self.x_dim = x_dim

    def forward(self, z, c):
        h_z = F.relu(self.fc1_1(z))
        h_c = F.relu(self.fc1_1(c))
        h = torch.cat([h_z, h_c], dim=-1)
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        x = self.fc5(h)
        x = x.view(x.size(0), 1, self.x_dim, self.x_dim)
        #x = self.activation(x)
        return x


def loss_vae(recon_x, x, z_mu, z_log_var, c_mu, c_log_var):
    n = x.size(0)
    C = x.size(1)
    H = x.size(2)
    W = x.size(3)
    Recon = F.mse_loss(recon_x, x, reduction='mean')  # reconstruction loss
    KLD = 1.00* (-0.5 - z_log_var + 0.5 * (z_mu ** 2 + z_log_var.exp() ** 2)).sum().div(n * C * H * W)
    KLD += 1.0* (-0.5 - c_log_var + 0.5 * (c_mu ** 2 + c_log_var.exp() ** 2)).sum().div(n * C * H * W)
    return Recon + KLD

def cos_sim_loss(f1, f2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    product = torch.abs(cos(f1, f2))
    return torch.sum(product) / f1.data.nelement()

def cos_dis_loss(f1, f2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    cos_val = torch.abs(cos(f1, f2))
    product = torch.sub(torch.ones_like(cos_val), cos_val)
    return torch.sum(product) / f1.data.nelement()


def train(model, num_epochs, loader, transformation, optimizer, encoder_optimizer):
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_vae_loss = 0
        epoch_con_c_loss = 0
        epoch_con_z_loss = 0
        count = 0
        model.train()
        
        for batch_idx, data in enumerate(loader):
            im = data.cuda()
            im_c_pos_z_neg = transformation(im)
            out, z_mu, z_log_var, c_mu, c_log_var = model(im)
            vae_loss1 = loss_vae(out, im, z_mu, z_log_var, c_mu, c_log_var)

            out_c_pos_z_neg, z_neg_mu, z_neg_log_var, c_pos_mu, c_pos_log_var = model(im_c_pos_z_neg)
            vae_loss2 = loss_vae(out_c_pos_z_neg, im_c_pos_z_neg, z_neg_mu, z_neg_log_var,c_pos_mu, c_pos_log_var)
            vae_loss = vae_loss1 + vae_loss2
            random_idx = torch.randperm(len(c_mu))
            im_c_neg = im[random_idx].cuda()
            _, _, c_neg_mu, c_neg_log_var = model.encoder(im_c_neg)

            con_c_loss = cos_sim_loss(c_mu, c_neg_mu) + cos_dis_loss(c_mu, c_pos_mu) 
            con_z_neg_loss = cos_sim_loss(z_mu, z_neg_mu)
            loss = vae_loss + con_c_loss + con_z_neg_loss
            epoch_loss += loss.item()
            epoch_vae_loss+= vae_loss.item()
            epoch_con_c_loss+= con_c_loss.item()
            epoch_con_z_loss+= con_z_neg_loss.item()    
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            random_idx1 = torch.randperm(len(z_mu))
            random_idx2 = torch.randperm(len(z_mu))
            style_latent_space = torch.FloatTensor(bs, z_dim).cuda()
            style_latent_space.normal_(0., 1.)
            _, _, c1_mu, c1_log_var = model.encoder(im[random_idx1].cuda())
            _, _, c2_mu, c2_log_var = model.encoder(im[random_idx2].cuda())
            
            
            im_z_pos_1 = model.decoder(style_latent_space, c1_mu.detach())
            im_z_pos_2 = model.decoder(style_latent_space, c2_mu.detach())

            z_pos_mu_1, _, _, _ = model.encoder(im_z_pos_1)
            z_pos_mu_2, _, _, _ = model.encoder(im_z_pos_2)

            con_z_pos_loss = cos_dis_loss(z_pos_mu_1, z_pos_mu_2)
            encoder_optimizer.zero_grad()
            con_z_pos_loss.backward()
            encoder_optimizer.step()
            epoch_loss += con_z_pos_loss.item()
            epoch_con_z_loss+= con_z_pos_loss.item()
            count += 1
        print('Epoch: ', epoch, 'Total Loss: ', epoch_loss / count,
              'VAE Loss: ', epoch_vae_loss / count,
              'Con(C) Loss: ', epoch_con_c_loss / count,
              'Con(Z) Loss: ', epoch_con_z_loss / count)
    return model


def get_square_interpolation_figures(model, dataset, x_dim, z_dim):
    torch.manual_seed(777)
    samples = []
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    index = 0
    while len(samples) < 5:
        dat = next(iter(loader))
        dat = dat.cpu().numpy().reshape(x_dim, x_dim, 1)
        samples.append(dat)

    # Take c from column_samples, Take z from row_samples
    # Make a grid of them then decode for every one
    model.eval()
    column_samples = torch.from_numpy(np.array(samples)).float().cuda().permute(0, 3, 1, 2)
    row_samples = torch.from_numpy(np.array(samples)).float().cuda().permute(0, 3, 1, 2)
    _, _, c, _ = model.encoder(column_samples)
    z, _, _, _ = model.encoder(row_samples)
    out_list = []
    for i in range(z.shape[0]):
        for j in range(c.shape[0]):
            out = model.decoder(z[i].view(-1, z_dim), c[j].view(-1, z_dim)).view(1, x_dim, x_dim)
            out_list.append(out.cpu().detach().numpy())
    out_arr = np.array(out_list)
    out = torch.from_numpy(out_arr)
    save_image(make_grid(out, nrow=5), 'duca_starmen_square_interpolation.png')
    save_image(make_grid(column_samples, nrow=5), 'duca_starmen_square_columns.png')
    save_image(make_grid(row_samples, nrow=1), 'duca_starmen_square_rows.png')

import pickle
def loadpickle(filename):
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
    return arr

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Train and Evaluate DualContrast on your dataset')
    parser.add_argument('-z', '--z-dim', type=int, default=16)
    parser.add_argument('-bs', '--batch-size', type=int, default=100)
    parser.add_argument('-ep', '--num-epochs', type=int, default=100)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('-train-only', '--only-train-model', action='store_true', default=False)
    parser.add_argument('-eval-only', '--only-evaluate-model', action='store_true', default=False)
    args = parser.parse_args()
    num_epochs = args.num_epochs
    bs = args.batch_size
    lr = args.learning_rate
    z_dim = args.z_dim
    train_only_flag = args.only_train_model  # Only Train the model?
    eval_only_flag = args.only_evaluate_model  # Only Evaluate a saved model?

    x_dim = 64 
    model = Model(z_dim=z_dim, x_dim=x_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr)
    
    train_data = loadpickle('data/starmen_train_dataset.pkl')
    train_dataset = []
    for i in range(len(train_data)):
        dat = torch.from_numpy(train_data[i].reshape(1,64,64)).float()
        train_dataset.append(dat)
    

    from torch.utils.data import random_split
    generator = torch.Generator().manual_seed(777)
    proportions = [.6, .4]
    lengths = [int(p * len(train_dataset)) for p in proportions]
    lengths[-1] = len(train_dataset) - sum(lengths[:-1])
    train_dataset, test_dataset = random_split(train_dataset, lengths, generator=generator)
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)

    # Train the model
    if not eval_only_flag:
        transformation = transforms.RandomAffine(90)
        
        model = train(model, num_epochs, train_loader, transformation, optimizer, encoder_optimizer)
        # Save the model
        torch.save(model.state_dict(), 'dualcontrast_starmen.pt')

    else:
        try:
            model.load_state_dict(torch.load('dualcontrast_starmen.pt'))
        except IOError:
            print("Error: No saved model found!")

    if not train_only_flag:
        # Get qualitative results of image generation with content-style transfer
        get_square_interpolation_figures(model, train_dataset, x_dim, z_dim)

