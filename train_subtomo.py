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
        self.encoder = ConvEncoder(z_dim, x_dim)
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

class ConvEncoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        self.pixel = pixel
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)  # 7 x 7
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)  # 4 x 4
        self.conv4 = nn.Conv3d(128, 512, kernel_size=4, stride=1, padding=0)  # 1 x 1
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=1, stride=1, padding=0)
        self.fc_1 = nn.Linear(1024, latent_dims)
        self.fc_2 = nn.Linear(1024, latent_dims)
        self.fc_3 = nn.Linear(1024, latent_dims)
        self.fc_4 = nn.Linear(1024, latent_dims)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        phi = x.view(x.size(0), -1)
        return self.fc_1(phi), self.fc_2(phi), self.fc_3(phi), self.fc_4(phi)


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim1=512, h_dim2=1024):
        super(Decoder, self).__init__()
        self.fc1_1 = nn.Linear(z_dim, h_dim1)
        self.fc1_2 = nn.Linear(z_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1 * 2, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim2)
        self.fc4 = nn.Linear(h_dim2, h_dim2)
        self.fc5 = nn.Linear(h_dim2, 5000)
        self.fc6 = nn.Linear(5000, 1*(x_dim)**3)
        self.x_dim = x_dim

    def forward(self, z, c):
        h_z = F.leaky_relu(self.fc1_1(z))
        h_c = F.leaky_relu(self.fc1_1(c))
        h = torch.cat([h_z, h_c], dim=-1)
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        h = F.leaky_relu(self.fc4(h))
        h = F.leaky_relu(self.fc5(h))
        x = self.fc6(h)
        x = x.view(x.size(0), 1, self.x_dim, self.x_dim, self.x_dim)
        return x


def loss_vae(recon_x, x, z_mu, z_log_var, c_mu, c_log_var):
    n = x.size(0)
    C = x.size(1)
    H = x.size(2)
    W = x.size(3)
    D = x.size(4)
    Recon = 3*F.mse_loss(recon_x, x, reduction='mean')  # reconstruction loss
    KLD = 0.01* (-0.5 - z_log_var + 0.5 * (z_mu ** 2 + z_log_var.exp() ** 2)).sum().div(n * C * H * W* D)
    KLD += 0.01* (-0.5 - c_log_var + 0.5 * (c_mu ** 2 + c_log_var.exp() ** 2)).sum().div(n * C * H * W *D)
    return Recon + KLD

def cos_sim_loss(f1, f2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    product = torch.abs(cos(f1, f2))
    return 3*torch.sum(product) / f1.data.nelement()

def cos_dis_loss(f1, f2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    cos_val = torch.abs(cos(f1, f2))
    product = torch.sub(torch.ones_like(cos_val), cos_val)
    return 3*torch.sum(product) / f1.data.nelement()

import kornia
class Transformer(object):
    def __call__(self, image, theta):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, C, D, H, W = image.size()
        center = torch.tensor([D / 2, H / 2, W / 2]).repeat(B, 1).to(device=device)
        scale = torch.ones(B, 1).to(device=device)
        angle = torch.rad2deg(theta)
        no_trans = torch.zeros(B, 3).to(device=device)
        M = kornia.get_affine_matrix3d(translations=no_trans, center=center, scale=scale, angles=angle)
        affine_matrix = M[:, :3, :]
        rotated_image = kornia.warp_affine3d(image, affine_matrix, dsize=(D, H, W), align_corners=False,
                                             padding_mode='zeros')
        return rotated_image


def train(model, num_epochs, loader, transformation, optimizer, encoder_optimizer, vae_scheduler, cycle_scheduler):
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_vae_loss = 0
        epoch_con_c_loss = 0
        epoch_con_z_loss = 0
        count = 0
        model.train()
        
        for batch_idx, data in enumerate(loader):
            im = data.cuda()
            with torch.no_grad():
                angles = torch.FloatTensor(im.size(0), 3).uniform_(-np.pi / 2, np.pi / 2).to(device=im.device)
                im_c_pos_z_neg = transformation(im, angles)
            out, z_mu, z_log_var, c_mu, c_log_var = model(im)
            vae_loss1 = loss_vae(out, im, z_mu, z_log_var, c_mu, c_log_var)

            _, z_neg_mu, z_neg_log_var, c_pos_mu, c_pos_log_var = model(im_c_pos_z_neg)
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
        vae_scheduler.step()
        cycle_scheduler.step()
        print('Epoch: ', epoch, 'Total Loss: ', epoch_loss / count,
              'VAE Loss: ', epoch_vae_loss / count,
              'Con(C) Loss: ', epoch_con_c_loss / count,
              'Con(Z) Loss: ', epoch_con_z_loss / count)
    return model


def save_embedding(model, loader, dataset_name):
    z_list = []
    c_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # print(batch_idx)
            im = data.cuda()
            z_mu, z_log_var, c_mu, c_log_var = model.encoder(im)
            # z_mu, z_log_var = model.z_encoder(im)
            # print(z_deformation.shape, z_deformation_independent.shape)
            z_list.append(z_mu.cpu().numpy().reshape(-1, z_dim))
            c_list.append(c_mu.cpu().numpy().reshape(-1, z_dim))

    z_arr = np.array(z_list).reshape(-1, z_dim)
    c_arr = np.array(c_list).reshape(-1, z_dim)
    np.save(dataset_name+'_style_embedding.npy',z_arr)
    np.save(dataset_name+'_content_embedding.npy',c_arr)

import pickle
def loadpickle(filename):
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
    return arr

def data_loader(dataset_name, pixel, batch_size=100,normalize=True):
    X_train = loadpickle('data/'+dataset_name + '.pkl') 
    X_test = X_train.copy()
    print(X_train.shape)

    if normalize:
        print('# normalizing particles')
        mu = X_train.reshape(-1, pixel * pixel * pixel).mean(1)
        std = X_train.reshape(-1, pixel * pixel * pixel).std(1)
        X_train = (X_train - mu[:, np.newaxis, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis, np.newaxis]
        mu = X_test.reshape(-1, pixel * pixel * pixel).mean(1)
        std = X_test.reshape(-1, pixel * pixel * pixel).std(1)
        X_test = (X_test - mu[:, np.newaxis, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis, np.newaxis]

    X_train = X_train.reshape(X_train.shape[0], 1, pixel, pixel, pixel)
    X_test = X_test.reshape(X_test.shape[0], 1, pixel, pixel, pixel)
    train_x = torch.from_numpy(X_train).float()
    test_x = torch.from_numpy(X_test).float()

    train_loader = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_x, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Train and Evaluate DualContrast on your dataset')
    parser.add_argument('-z', '--z-dim', type=int, default=50)
    parser.add_argument('-bs', '--batch-size', type=int, default=50)
    parser.add_argument('-ep', '--num-epochs', type=int, default=150)
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

    x_dim = 32  
    model = Model(z_dim=z_dim, x_dim=x_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr)
    vae_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    cycle_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=50, gamma=0.1)
    dataset_name = 'composition_conformation_subtomograms'

    train_loader, test_loader = data_loader(dataset_name=dataset_name, pixel = x_dim, batch_size=bs)

    if not eval_only_flag:
        transformation = Transformer()
        model = train(model, num_epochs, train_loader, transformation, optimizer, encoder_optimizer)
        torch.save(model.state_dict(), 'dualcontrast_composition_conformation.pt')

    else:
        try:
            model.load_state_dict(torch.load('dualcontrast_composition_conformation.pt'))
        except IOError:
            print("Error: No saved model found!")

    if not train_only_flag:

        #Save the embeddings for content and transformation factor 
        save_embedding(model, test_loader, dataset_name)


