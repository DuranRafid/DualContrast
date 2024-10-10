import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid, save_image
import pickle

def loadpickle(fname):
    with open(fname, 'rb') as f:
        array = pickle.load(f)
    f.close()
    return array
    
class Model(nn.Module):
    def __init__(self, z_dim, x_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim, x_dim)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 

    def forward(self, x):
        z_mu, z_log_var, c_mu, c_log_var = self.encoder(x)
        z = self.sampling(z_mu, z_log_var)
        c = self.sampling(c_mu, c_log_var)
        return self.decoder(z, c), z_mu, z_log_var, c_mu, c_log_var

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.conv1 = self.my_conv_layer(3, 16)
        self.conv2 = self.my_conv_layer(16, 32)
        self.conv3 = self.my_conv_layer(32, 64)
        self.conv4 = self.my_conv_layer(64, 128)
        self.fc1 = nn.Linear(1152, latent_dims, bias=True)
        self.fc2 = nn.Linear(1152, latent_dims, bias=True)
        self.fc3 = nn.Linear(1152, latent_dims, bias=True)
        self.fc4 = nn.Linear(1152, latent_dims, bias=True)
        self.lt = latent_dims
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
        return self.fc1(x), self.fc2(x), self.fc3(x), self.fc4(x) 

class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim1=512, h_dim2=1024):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim*2, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim1)
        self.fc3 = nn.Linear(h_dim1, h_dim2)
        self.fc4 = nn.Linear(h_dim2, h_dim2)
        self.fc5 = nn.Linear(h_dim2, x_dim*x_dim*3)
        self.activation = nn.Sigmoid()
        self.x_dim = x_dim

    def forward(self, z, c):
        h = torch.cat([z,c],dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        x = self.fc5(h)
        x = x.view(x.size(0),3, self.x_dim, self.x_dim)
        x = self.activation(x)
        return x


def loss_vae(recon_x, x, z_mu, z_log_var, c_mu, c_log_var):
    n = x.size(0)
    C = x.size(1)
    H = x.size(2)
    W = x.size(3)
    BCE = F.mse_loss(recon_x, x, reduction='mean')  # reconstruction loss
    KLD = 0.01*(-0.5 - z_log_var + 0.5 * (z_mu**2 + z_log_var.exp() ** 2)).sum().div(n*C*H*W)
    KLD += 0.01*(-0.5 - c_log_var + 0.5 * (c_mu ** 2 + c_log_var.exp() ** 2)).sum().div(n*C*H*W)
    return BCE + KLD    

def get_pred_score(model):
    train_z_list = []
    train_c_list = []
    train_class_labels = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data/255.0
            im = data.cuda()
            im = im.permute(0, 3, 1, 2)
            z_mu, z_log_var, c_mu, c_log_var = model.encoder(im)
            train_class_labels.append(labels.cpu().numpy())
            train_z_list.append(z_mu.cpu().numpy().reshape(-1, z_dim))
            train_c_list.append(c_mu.cpu().numpy().reshape(-1, z_dim))

    train_z_arr = np.array(train_z_list).reshape(-1, z_dim)
    train_c_arr = np.array(train_c_list).reshape(-1, z_dim)
    train_class_labels = np.array(train_class_labels).reshape(-1, 1)

    test_z_list = []
    test_c_list = []
    test_class_labels = []
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data/255.0
            im = data.cuda()
            im = im.permute(0, 3, 1, 2)
            z_mu, z_log_var, c_mu, c_log_var = model.encoder(im)
            test_class_labels.append(labels.cpu().numpy())
            test_z_list.append(z_mu.cpu().numpy().reshape(-1, z_dim))
            test_c_list.append(c_mu.cpu().numpy().reshape(-1, z_dim))

    test_z_arr = np.array(test_z_list).reshape(-1, z_dim)
    test_c_arr = np.array(test_c_list).reshape(-1, z_dim)
    test_class_labels = np.array(test_class_labels).reshape(-1, 1)

    from sklearn.linear_model import LogisticRegression
    z_model = LogisticRegression() 
    z_model.fit(train_z_arr, train_class_labels.ravel())
    z_pred_score = z_model.score(test_z_arr, test_class_labels.ravel())
    print('z Prediction Score', z_pred_score)
    c_model = LogisticRegression() 
    c_model.fit(train_c_arr, train_class_labels.ravel())
    c_pred_score = c_model.score(test_c_arr, test_class_labels.ravel())
    print('c Prediction Score', c_pred_score)
    return z_pred_score, c_pred_score

def cos_sim_loss(f1, f2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    product = torch.abs(cos(f1, f2))
    return torch.sum(product)/f1.data.nelement()

def cos_dis_loss(f1, f2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    cos_val = torch.abs(cos(f1, f2))
    product = torch.sub(torch.ones_like(cos_val), cos_val)
    return torch.sum(product)/f1.data.nelement()

def train(model, num_epochs, loader, transformation, optimizer, encoder_optimizer, vae_scheduler, cycle_scheduler):
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_vae_loss = 0
        epoch_con_c_loss = 0
        epoch_con_z_loss = 0
        count = 0
        model.train()
        
        for batch_idx, (data, _) in enumerate(loader):
            data = data / 255.0
            im = data.cuda()
            im = im.permute(0, 3, 1, 2)
            im_c_pos_z_neg = transformation(im)
            out, z_mu, z_log_var, c_mu, c_log_var = model(im)
            vae_loss1 = loss_vae(out, im, z_mu, z_log_var, c_mu, c_log_var)

            out_c_pos_z_neg, z_neg_mu, z_neg_log_var, c_pos_mu, c_pos_log_var = model(im_c_pos_z_neg)
            vae_loss2 = loss_vae(out_c_pos_z_neg, im_c_pos_z_neg, z_neg_mu, z_neg_log_var,c_pos_mu, c_pos_log_var)
            vae_loss = vae_loss1 + vae_loss2
            random_idx = torch.randperm(len(c_mu))
            im_c_neg = im[random_idx].cuda()
            _, _, _, c_neg_mu, c_neg_log_var = model(im_c_neg)
            con_c_loss = cos_sim_loss(c_mu, c_neg_mu) + cos_dis_loss(c_mu, c_pos_mu)
            con_z_neg_loss = cos_sim_loss(z_mu, z_neg_mu)
            loss = vae_loss + con_z_neg_loss +con_c_loss
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
        vae_scheduler.step()
        cycle_scheduler.step()
    return model

def save_embedding(model, train_loader, test_loader):
    z_list = []
    c_list = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(train_loader):
            im = data.cuda()
            im = im/255.0
            im = im.permute(0, 3, 1, 2)
            z_mu, z_log_var, c_mu, c_log_var = model.encoder(im)
            z_list.append(z_mu.cpu().numpy().reshape(-1, z_dim))
            c_list.append(c_mu.cpu().numpy().reshape(-1, z_dim))
            labels.append(label.cpu().numpy())

    z_arr = np.array(z_list).reshape(-1, z_dim)
    c_arr = np.array(c_list).reshape(-1, z_dim)
    labels = np.array(labels).reshape(-1,1)
    np.save('duca_linemod_train_transformation_embedding.npy',z_arr)
    np.save('duca_linemod_train_content_embedding.npy',c_arr)
    np.save('linemod_train_labels.npy',labels)

    z_list = []
    c_list = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,label) in enumerate(test_loader):
            im = data.cuda()
            im = im/255.0
            im = im.permute(0, 3, 1, 2)
            z_mu, z_log_var, c_mu, c_log_var = model.encoder(im)
            z_list.append(z_mu.cpu().numpy().reshape(-1, z_dim))
            c_list.append(c_mu.cpu().numpy().reshape(-1, z_dim))
            labels.append(label.cpu().numpy())

    z_arr = np.array(z_list).reshape(-1, z_dim)
    c_arr = np.array(c_list).reshape(-1, z_dim)
    labels = np.array(labels).reshape(-1,1)
    np.save('duca_linemod_test_transformation_embedding.npy',z_arr)
    np.save('duca_linemod_test_content_embedding.npy',c_arr)
    np.save('linemod_test_labels.npy',labels)

def get_square_interpolation_figures(model):
    torch.manual_seed(777)
    samples = []
    loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    label_ls = []
    index = 0
    while len(label_ls)<15:
        dat, label = next(iter(loader))
        #print(index, len(label_ls))
        if label.item() not in label_ls:
            label_ls.append(label.item())
            dat = dat/255.0
            dat = dat.cpu().numpy().reshape(x_dim, x_dim, 3)
            samples.append(dat)
        index = index+1

    #Take c from column_samples, Take z from row_samples
    #Make a grid of them then decode for every one

    column_samples = torch.from_numpy(np.array(samples)).float().cuda().permute(0,3,1,2)
    row_samples = torch.from_numpy(np.array(samples)).float().cuda().permute(0,3,1,2)
    _, _, c, _ = model.encoder(column_samples)
    z, _, _, _ = model.encoder(row_samples)
    out_list = []
    for i in range(z.shape[0]):
        for j in range(c.shape[0]):
            out = model.decoder(z[i].view(-1,z_dim),c[j].view(-1,z_dim)).view(3, x_dim, x_dim)
            out_list.append(out.cpu().detach().numpy())
    out_arr = np.array(out_list)
    out = torch.from_numpy(out_arr)
    print(out.shape)
    save_image(make_grid(out, nrow=15), 'dualcontrast_linemod_square_interpolation.png')
    save_image(make_grid(column_samples, nrow=15), 'dualcontrast_linemod_square_columns.png')
    save_image(make_grid(row_samples, nrow=1), 'dualcontrast_linemod_square_rows.png')
    print(out_arr.shape)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Train and Evaluate DualContrast on your dataset')
    parser.add_argument('-z', '--z-dim', type=int, default=16)
    parser.add_argument('-bs', '--batch-size', type=int, default=100)
    parser.add_argument('-ep', '--num-epochs', type=int, default=200)
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
    vae_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    cycle_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=50, gamma=0.1)
    
    train_dataset = loadpickle('data/linemod_masked_train_dataset.pkl')
    test_dataset = loadpickle('data/linemod_masked_test_dataset.pkl')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)

    # Train the model
    if not eval_only_flag:
        transformation = transforms.RandomAffine(90)
        
        model = train(model, num_epochs, train_loader, transformation, optimizer, encoder_optimizer, vae_scheduler, cycle_scheduler)
        # Save the model
        torch.save(model.state_dict(), 'dualcontrast_linemod.pt')

    else:
        try:
            model.load_state_dict(torch.load('models/dualcontrast_linemod.pt'))
        except IOError:
            print("Error: No saved model found!")

    if not train_only_flag:
        # Get D_score(c|c) and D_score(c|z)
        get_pred_score(model, train_loader, test_loader)
        
        #Save the embeddings for content and transformation factor 
        save_embedding(model, train_loader, test_loader)
        
        # Get qualitative results of image generation with content-style transfer
        get_square_interpolation_figures(model, train_dataset, x_dim, z_dim)


