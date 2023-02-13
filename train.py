import torch
from torch.utils.data import Dataset
from model import GainTune
import numpy as np
from torch.utils.data import DataLoader


import os
import pandas as pd
from torchvision.io import read_image

def rolling_window(a, window):
    shape = a.shape[:-2] + (a.shape[0] - window + 1,a.shape[1], window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class data(Dataset):
    def __init__(self, files, batch_size = 10):
        self.files = files

        thrust_cmds  =np.empty((0,1))
        ang_vel_cmds =np.empty((0,3))
        ppo_thrust   =np.empty((0,1))
        ppo_ang      =np.empty((0,3))
        
        for i in range(len(files)):
            saved_data = np.load(files[i])

            ts = saved_data['ts']

            l = len(ts)
            rl = len(ts[ts>10.0])

            p = l - rl
            ts = ts[p:]
            thrust_cmds =  np.concatenate((thrust_cmds, np.expand_dims(saved_data['thrust_cmds'][p:],1)),axis=0)
            ang_vel_cmds = np.concatenate((ang_vel_cmds,saved_data['ang_vel_cmds'][p:]),axis=0)
            ppo_thrust =   np.concatenate((ppo_thrust,np.expand_dims(saved_data['ppo_acc'][p:],1)),axis=0)
            ppo_ang =      np.concatenate((ppo_ang, saved_data['ppo_ang'][p:]),axis=0)
        
        # self.thrust_cmds = rolling_window(thrust_cmds,5)
        self.thrust_cmds = thrust_cmds[5:]
        # self.thrust_cmds = np.expand_dims(self.thrust_cmds,1)
        self.ang_vel_cmds = ang_vel_cmds[5:]
        # self.ang_vel_cmds = np.expand_dims(self.ang_vel_cmds,1)
        self.ppo_thrust = rolling_window(ppo_thrust,5)
        # self.ppo_thrust = np.expand_dims(self.ppo_thrust,1)
        self.ppo_ang = rolling_window(ppo_ang,5)
        # self.ppo_ang = np.expand_dims(self.ppo_ang,1)

        self.pid = np.concatenate((self.thrust_cmds,self.ang_vel_cmds),axis=1)
        self.ppo = np.concatenate((self.ppo_thrust,self.ppo_ang),axis=1)
        self.pid = torch.tensor(self.pid).unsqueeze(-1).float()
        self.ppo = torch.tensor(self.ppo).float()
    def __len__(self,):
        return len(self.thrust_cmds)

    def __getitem__(self, idx):
        
        pid  = self.pid[idx]
        ppo = self.ppo[idx]
        return dict(pid=pid,ppo=ppo)

files = ["Data/log_12_02_20.npz",
         "Data/log_12_02_21.npz",
         "Data/log_12_02_22.npz",
         "Data/log_12_02_23.npz",
         "Data/log_12_02_24.npz",
         "Data/log_13_02_02.npz",
         "Data/log_13_02_03.npz" ]

train_data = data(files)
batch_size=10
train_loader = DataLoader(train_data, batch_size=batch_size)

gain_model = GainTune(batch_size)

opt = torch.optim.Adam(gain_model.parameters(),lr = 0.001)
criterion = torch.nn.MSELoss()

for ep in range(50):
    run_loss = 0.0
    print("epoch : ", ep)
    for i, batch_data in enumerate(train_loader,0):
        inp = batch_data["ppo"]
        target = batch_data["pid"]
        if inp.size()[0]==batch_size:
            opt.zero_grad()

            output, h = gain_model(inp)


            loss = criterion(output, target)
            loss.backward()
            opt.step()

            run_loss+=loss.item()

            if i%100==99:
                print("\t step : ",i, "\t Loss : ", run_loss/100)
                run_loss = 0.0


        # print(output.size())
        # exit()

# a = data1.__getitem__(idx=4)
# import pdb;pdb.set_trace()
# inp = torch.tensor(np.random.rand(20,4,5))
# model = GainTune(10)
# op = model.forward(a["ppo"])
# print(op.size())
