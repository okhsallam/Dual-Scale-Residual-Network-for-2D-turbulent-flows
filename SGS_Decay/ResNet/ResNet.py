from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
from Loss import*
from Generator import*
import os 
import time
from matplotlib.colors import LogNorm
# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

#%% Get values form the Config file  
import configparser
config = configparser.ConfigParser()
Config_file='Config.ini'
config.read(Config_file)
###
Dataset_Train_path = eval(config['Train_Dataset']['path'])
Dataset_Test_path = eval(config['Test_Dataset']['path'])


Dimension = eval(config['Model']['Dimension'])
train_iter = eval(config['Model']['iteration'])
display_step = eval(config['Model']['display_step'])
loss_write_step =eval(config['Model']['loss_write_step'])

#%%
# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################
#%%  Create a results file 
isExist = os.path.exists("Results")
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs("Results")
   
# Create a loss file to write 
file1 = open('Results/Loss_resnet.txt', 'w')
file1.close()
#%%
#%% Train the generator only 
def train_resnet(resnet, dataloader, device, lr=1e-4, total_steps=1e6, display_step=500):
    resnet = resnet.to(device).train()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

    cur_step = 0
    mean_loss = 0.0
    while cur_step < total_steps:
        for Output_real, Input_real in tqdm(dataloader, position=0):
            Output_real = Output_real.to(device)
            Input_real = Input_real.to(device)

    
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                Output_fake = resnet(Input_real)
                
                loss_tau  = Loss.img_loss(Output_real, Output_fake)
                
                g_loss  = loss_tau 
                
            if cur_step % loss_write_step == 0:
              file1 = open('Results/Loss_resnet.txt', 'a')
              file1.write(str(cur_step) + " ")
              file1.write(str(g_loss.cpu().detach().numpy()) + " \n")
           
              file1.close()

            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()

            mean_loss += g_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                # print(loss_spectrum)
                print('Step {}: resnet loss: {:.5f}'.format(cur_step, mean_loss))
                # print(Input_real.size())
                
                # # Plot tau real
                plt.figure(figsize=(15,4))
                plt.subplot(1,3,1)
                plt.imshow(Output_real[0,0,:,:].cpu())
                plt.colorbar()
                plt.subplot(1,3,2)
                plt.imshow(Output_real[0,1,:,:].cpu())
                plt.colorbar()
                plt.subplot(1,3,3)
                plt.imshow(Output_real[0,2,:,:].cpu())
                plt.colorbar()
                plt.tight_layout()
                plt.show()
                # Plot tau fake
                plt.figure(figsize=(15,4))
                plt.subplot(1,3,1)
                plt.imshow(Output_fake[0,0,:,:].cpu().detach().numpy())
                plt.colorbar()
                plt.subplot(1,3,2)
                plt.imshow(Output_fake[0,1,:,:].cpu().detach().numpy())
                plt.colorbar()
                plt.subplot(1,3,3)
                plt.imshow(Output_fake[0,2,:,:].cpu().detach().numpy())
                plt.colorbar()
                plt.tight_layout()
                plt.show()
                
  
                
                mean_loss = 0.0

            cur_step += 1
            if cur_step == total_steps:
                break

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_res_blocks = eval(config['Model']['n_res_blocks']) 
generator = Generator(base_channels=64,n_res_blocks=n_res_blocks)
#%% Get the dataset 
Batch_Size = eval(config['Model']['Batch_Size'])
from Dataset import*
dataset=OF_Dataset()
dataloader = torch.utils.data.DataLoader(dataset,
    batch_size=Batch_Size,  shuffle=True,pin_memory=True)
    
#%% Train the Generator first 
tic  = time.time()
train_resnet(generator, dataloader, device, lr=1e-4, total_steps=train_iter, display_step=display_step)
toc  = time.time()
torch.save(generator, 'resnet.pt')
np.save('Results/Train_Time', tic-toc)



#%%    
def predictor(Input_Dataset_path,Output_Dataset_path):    
    std_output = np.load('Results/std_output_dataset.npy')
    Inputs =np.load(Input_Dataset_path)
    # Lets denormalize the inputs 
    Inputs[:,0:2,:,:] = Inputs[:,0:2,:,:]/np.load('Results/std_input_velocity_dataset.npy')
    Inputs[:,2,:,:] = Inputs[:,2,:,:]/np.load('Results/std_input_vorticity_dataset.npy')
    
    Inputs= np.array(Inputs, dtype='float32')
    Outputs_pred = np.zeros((Inputs.shape[0],Inputs.shape[1],Inputs.shape[2],Inputs.shape[3]))
    
 
    for t  in range(0,Inputs.shape[0]):
   
        Input = torch.from_numpy(Inputs[t,:,:,:])
        Input=Input.to(device)
        Output_pred =  generator(Input.reshape(1,Inputs.shape[1],Inputs.shape[2],Inputs.shape[3]))
        for i in range(0,3):
            Outputs_pred[t,i,:,:] = Output_pred.cpu().detach().numpy()[:,i,:,:]*std_output[i]
        
    np.save(Output_Dataset_path,Outputs_pred)




#%% Save the Training data predictions 
Train_Input_Dataset_path = Dataset_Train_path + '/Input_Train.npy'
Train_Output_Dataset_path = 'Results/Output_Train.npy'
predictor(Train_Input_Dataset_path,Train_Output_Dataset_path)



# Save the Testing data predictions 
Test_Input_Dataset_path = Dataset_Test_path[0] + '/Input_Test.npy'
Test_Output_Dataset_path = 'Results/Output_Test.npy'
predictor(Test_Input_Dataset_path,Test_Output_Dataset_path)














