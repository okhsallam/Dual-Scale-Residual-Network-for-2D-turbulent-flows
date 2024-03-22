# from PIL import Image
import numpy as np
# import torchvision
# import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import configparser
#%%
config = configparser.ConfigParser()

Config_file='Config.ini'
config.read(Config_file)

Dataset_path = eval(config['Train_Dataset']['path'])

x_start = int(eval(config['Train_Dataset']['x_start']))
y_start = int(eval(config['Train_Dataset']['y_start']))

x_end = int(eval(config['Train_Dataset']['x_Elements'])) + x_start
y_end = int(eval(config['Train_Dataset']['y_Elements'])) + y_start

#%%
# Output_Numpy  = np.load(Dataset_path+"/"+"Output_Train.npy")


class OF_Dataset(Dataset):
    


    def __init__(self):
       std_output_dataset = np.zeros(3)
       # Here I load all data set to the CPU memory
       
       Input_Numpy  = np.load(Dataset_path+"/"+"Input_Train.npy")
       Output_Numpy  = np.load(Dataset_path+"/"+"Output_Train.npy")
       
        ###########################################3
       # Normalize the Output
       for i in range(0,3):
           std_output_dataset[i] = np.std(Output_Numpy[:,i,:,:])   
           Output_Numpy[:,i,:,:] = Output_Numpy[:,i,:,:]/std_output_dataset[i]
       
       # Save the std of the all outputs 
      
       np.save('Results/std_output_dataset.npy',std_output_dataset)

        ###########################################3
        # Normaliza the Input 
       std_input_velocity_dataset = np.std(Input_Numpy[:,0:2,:,:])
       std_input_vorticity_dataset = np.std(Input_Numpy[:,2,:,:])
       Input_Numpy[:,0:2,:,:] = Input_Numpy[:,0:2,:,:]/std_input_velocity_dataset
       Input_Numpy[:,2,:,:] = Input_Numpy[:,2,:,:]/std_input_vorticity_dataset
       np.save('Results/std_input_velocity_dataset.npy',std_input_velocity_dataset)
       np.save('Results/std_input_vorticity_dataset.npy',std_input_vorticity_dataset)

       ###########################################
       
       self.Input= torch.from_numpy(Input_Numpy)
       self.Output= torch.from_numpy(Output_Numpy)
       
       
       
       self.n_samples =self.Input.shape[0] 
       
       self.Input = self.Input.to(torch.float32)
       self.Output = self.Output.to(torch.float32)
       
    



    def __getitem__(self, idx):
        output_sample = self.Output[idx]
        input_sample = self.Input[idx]

        return output_sample, input_sample


    def __len__(self):
          return  self.n_samples
  
    
    
# dataset=OF_Dataset()


    
    





