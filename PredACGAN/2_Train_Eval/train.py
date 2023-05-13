import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from torch import nn
import torch.utils.data
import torchvision.datasets as data_set
import torchvision.transforms as transforms
from torch.nn.utils import spectral_norm
from torch import optim
from torch.autograd import Variable
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU

Lambda = 32
epochs = 1048576 * 128
class generator(nn.Module):

    # generator model
    def __init__(self, in_channels):  # in_channels = 350
        super(generator, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1024, bias=True)

        self.t1 = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU()
        )
        self.t2 = nn.Sequential(
            nn.Linear(1024, 4, bias=True),
            nn.Identity()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.t1(x)
        x = self.t2(x)
        xx = x[:,0].clone().unsqueeze(1)
        y = torch.cat((xx, nn.functional.softmax(x[:,1:4],dim=1).clone()),1)
        return y  # output of generator hat of y


class discriminator(nn.Module):

    def __init__(self, classes=250):
        super(discriminator, self).__init__()
        self.c1 = nn.Sequential(
            spectral_norm(nn.Linear(4, 2048, bias=True)),
            nn.LeakyReLU(0.2)
        )
        self.c2 = nn.Sequential(
            spectral_norm(nn.Linear(2048, 2048, bias=True)),
            nn.LeakyReLU(0.2)
        )
        self.c3 = nn.Sequential(
            spectral_norm(nn.Linear(2048, 2048, bias=True)),
            nn.LeakyReLU(0.2)
        )

        self.fc_source = spectral_norm(nn.Linear(2048, 1))  # 0,1
        self.fc_class = spectral_norm(nn.Linear(2048, classes))  # output hat of x (batch_size, 250)
        self.sig = nn.Sigmoid()
        self.iden = nn.Identity()

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        rf = self.fc_source(x)
        c = self.iden(self.fc_class(
            x))  # checks class(label) of data--i.e. to which label the data belongs in the CIFAR10 dataset

        return rf, c


scaler = StandardScaler()
#scaler = joblib.load('scaler.pkl')
lr = 0.00001#0.00001
batch_size = 2048

real_label = torch.FloatTensor(batch_size).cuda()  # (100,1)
real_label.fill_(1)
real_label = real_label.unsqueeze(1)

fake_label = torch.FloatTensor(batch_size).cuda()  # (100,1)
fake_label.fill_(0)
fake_label = fake_label.unsqueeze(1)

# MSE loss pred - label
def compute_acc(preds, labels):
    loss = nn.MSELoss()
    output = loss(preds, labels)
    return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class HingeLoss(torch.nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = 1. - torch.mul(output, target)
        return torch.mean(F.relu(hinge_loss))


class Dataset_(Dataset):
    def __init__(self):
        dataset = pd.read_csv('train_one_hot_2%.csv')
        dataset = dataset.to_numpy()
        np.random.shuffle(dataset)
        self.y_data = torch.from_numpy(dataset[:, 250:])
        dataset = np.log(dataset[:,:250])
        dataset = scaler.fit_transform(dataset)
        self.len = dataset.shape[0]
        self.x_data = torch.from_numpy(dataset)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

#dataset = joblib.load('train_one_hot_2%.pkl')
dataset = Dataset_()#
#joblib.dump(scaler, 'scaler.pkl')
#joblib.dump(dataset, 'train_one_hot_2%.pkl', protocol=4)
train = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gen = generator(350).cuda()
disc = discriminator().cuda()

gen.apply(weights_init)
torch.autograd.set_detect_anomaly(True)
optimD = optim.Adam(disc.parameters(), lr*2, betas=(0., 0.9))
optimG = optim.Adam(gen.parameters(), lr, betas=(0., 0.9))

source_obj = nn.BCELoss()
class_obj = nn.MSELoss()
CE_obj = nn.BCEWithLogitsLoss()


for epoch in range(epochs):
    for i, data in enumerate(train):

        x_data, y_data = data
        #y_data = y_data.view(batch_size,4)
        data_row = x_data.shape[0]
        x_data, y_data = x_data.cuda(), y_data.cuda()
        x_data, y_data = x_data.float(), y_data.float()

        source_, dis_x = disc(y_data)  # feed real x to discriminator
        source_, dis_x = source_.float(), dis_x.float()
        d_loss_real = HingeLoss()(source_, torch.ones(data_row,1).cuda())
        class_error_d = class_obj(dis_x, x_data)# Lc

        accuracy = compute_acc(dis_x, x_data)

        # training with fake data

        optimD.zero_grad()     
        noise_ = np.random.normal(0, 1, (data_row, 100))  # (100,100)

        noise = ((torch.from_numpy(noise_)).float())
        noise = noise.cuda()  # converting to tensors in order to work with pytorch
        noise = torch.cat([noise, x_data], dim=1)  # (100, 100 + 250)


        noise_y = gen(noise)
        source_, dis_x = disc(noise_y)
        source_, dis_x = source_.float(), dis_x.float()
        d_loss_fake = HingeLoss()(source_, -torch.ones(data_row,1).cuda())
        d_loss_GAN = d_loss_real + d_loss_fake #Ls
        d_loss = d_loss_GAN + Lambda * class_error_d
        d_loss.backward()
        optimD.step()
        

        label = np.random.normal(0, 1, (data_row, 250))  # (100, 250)
        label = (torch.from_numpy(label))  # (100, 250)
        label = label.cuda()  # converting to tensors in order to work with pytorch
        label = label.float()

        
        
        
        if i % 16 ==0:#16 ==0:
            gen.zero_grad()
            
            noise = ((torch.from_numpy(noise_)).float())
            noise = noise.cuda()  # converting to tensors in order to work with pytorch
            noise = torch.cat([noise, x_data], dim=1)  # (100, 100 + 250)
            noise_y = gen(noise)
            source_, dis_x = disc(noise_y)
            source_, dis_x = source_.float(), dis_x.float()
    
            error_gen_GAN = -torch.mean(source_)#- torch.mean(source_) # original
            class_error_g = class_obj(dis_x, x_data)
            error_gen = error_gen_GAN + Lambda * class_error_g
            error_gen = error_gen.float()
            error_gen.backward()
            optimG.step()
            
            if i%2000 == 0:
                print(noise_y)

        iteration_now = epoch * len(train) + i
        print("Epoch--[{} / {}], Loss_Discriminator--[{}], Loss_Generator--[{}], Loss_Class_D--[{}], Loss_Class_G--[{}], LOSS--[{}]".format(epoch, epochs,
                                                                                                    d_loss_GAN.item(),
                                                                                                    error_gen_GAN.item(),
                                                                                                    class_error_d.item(),
                                                                                                    class_error_g.item(),
                                                                                                    accuracy.item()))
        if (epoch & (epoch-1)):
            pass
        else:
            if i == 0:
                torch.save(gen, './model/wgan_generator_epoch'+ str(epoch) +'.pt') #d_class_loss rate
                torch.save(gen.state_dict(), './model/wgan_state_dict_epoch'+ str(epoch) +'.pt')



PATH = './'
