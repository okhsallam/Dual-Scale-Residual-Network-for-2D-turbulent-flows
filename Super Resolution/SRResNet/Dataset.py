# from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader

import configparser
config = configparser.ConfigParser()

Config_file='Config.ini'
config.read(Config_file)

scale_factor = eval(config['Model']['scale_factor'])
Dataset_path = eval(config['Train_Dataset']['path'])

x_start = int(eval(config['Train_Dataset']['x_start']))
y_start = int(eval(config['Train_Dataset']['y_start']))

x_end = int(eval(config['Train_Dataset']['x_Elements'])) + x_start
y_end = int(eval(config['Train_Dataset']['y_Elements'])) + y_start


class OF_Dataset(Dataset):
    


    def __init__(self):
       # Here I load all data set to the CPU memory
       
       self.U_lrs = torch.from_numpy(np.load(Dataset_path+"/"+"U_lr_" + str(scale_factor)+".npy")[:,:,int(y_start/scale_factor):int(y_end/scale_factor),int(x_start/scale_factor):int(x_end/scale_factor)])
       self.U_hrs = torch.from_numpy(np.load(Dataset_path+'/'+'U_hr.npy')[:,:,y_start:y_end,x_start:x_end])
       
       self.n_samples =self.U_lrs.shape[0] 
       
       self.U_lrs = self.U_lrs.to(torch.float32)
       self.U_hrs = self.U_hrs.to(torch.float32)

# y = y.to(torch.long)

       # print(self.U_hrs[200])
       
  

    def __getitem__(self, idx):
       
        

        # return (self.U_hrs[idx], self.U_lrs[idx])
        return self.U_hrs[idx] ,self.U_lrs[idx]


    def __len__(self):
          return  self.n_samples
  
    
    
# dataset=OF_Dataset()


    
    





