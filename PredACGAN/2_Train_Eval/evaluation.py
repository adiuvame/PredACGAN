import torch
import pandas as pd
import numpy as np
import scipy.stats as stats
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import matplotlib.pyplot as plt
import math
import sys

epoch_num = 128

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

PATH = './model/'
scaler = StandardScaler()

GPU_NUM = 1
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU


model = torch.load(PATH +'wgan_generator_epoch'+ str(epoch_num) +'.pt')
model.load_state_dict(torch.load(PATH + 'wgan_state_dict_epoch'+ str(epoch_num)+'.pt'))
model.eval()
model.to(device)
test = pd.read_csv('test_one_hot_2%.csv')
test = test.to_numpy()
#test = joblib.load('test_one_hot_2%.pkl')

test_x = test[:,:250]
test_x = scaler.fit_transform(test_x)
test_x = np.log(test_x)
test_y = test[:,250:]
batch_size = 100
KL_ = np.array([])
y_ = np.array([])


print('Start')
for i in range(len(test)):
    if not np.isnan(test[i,0]):
        x = np.array([test_x[i,:]])
        for j in range(batch_size-1):
            x = np.append(x, [test_x[i,:]], axis = 0)
        z = np.random.normal(0, 1, (100,100))
        concat_x = np.concatenate((z,x), axis= 1)
        concat_x = ((torch.from_numpy(concat_x)).float())
        concat_x = concat_x.cuda()
        y = model(concat_x)
        y = y.cpu().detach().numpy()
        KL1 = 0
        KL2 = 0
        KL = 0
        y2 = np.nanmean(y, axis=0)
        y_ = np.append(y_,y2)
        KL1 = 0
        KL2 = 0
        m = np.argmax(np.nanmean(y[:, 1:], axis=0))
        m = m + 1
        for j in range(len(y)):
            others_sum = 0
            for k in range(1,4):
                if k != m:
                    others_sum = others_sum + y[j,k]
            KL1 = y[j,m] * (np.log(y[j,m] / others_sum)) + KL1
            KL2 = others_sum * (np.log(others_sum / y[j,m])) + KL2



        KL = KL1 + KL2
        KL = KL/100
        KL_ = np.append(KL_, -KL)
    else:
        KL_ = np.append(KL_,np.NaN)
        y_ = np.append(y_,np.NaN)
        y_ = np.append(y_,np.NaN)
        y_ = np.append(y_,np.NaN)
        y_ = np.append(y_,np.NaN)

    if i % 1000 == 0:
        print('%d / %d'%(i, len(test)))




KL_ = KL_.reshape(-1,1)
print("KL")
print(KL_)
#joblib.dump(KL_,'./'+str(epoch_num)+ '/result_KL'+ str(epoch_num) +'.pkl')
joblib.dump(KL_,'./model/result_KL.pkl')
KL_ = pd.DataFrame(KL_)
#KL_.to_csv('./'+str(epoch_num)+ '/result_KL'+ str(epoch_num) +'.csv')
KL_.to_csv('./model/result_KL.csv')

print("y")
y_ = y_.reshape(-1,4)
print(y_)
y_ = pd.DataFrame(y_)
y_.to_csv('./model/result_y.csv', index = False)
joblib.dump(y_, './model/result_y.pkl')