from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
from Loss import*
from Generator import*
import os 
import time
#%% Get values form the Config file  
import configparser
config = configparser.ConfigParser()
Config_file='Config.ini'
config.read(Config_file)
###
Dataset_Train_path = eval(config['Train_Dataset']['path'])
Dataset_Test_path = eval(config['Test_Dataset']['path'])


Dimension = eval(config['Model']['Dimension'])
scale_factor = eval(config['Model']['scale_factor'])
train_iter = eval(config['Model']['iteration'])
display_step = eval(config['Model']['display_step'])
loss_write_step =eval(config['Model']['loss_write_step'])

#%%  Loss weights 
Beta_U = eval(config['Physics']['Beta_U'])
Beta_con = eval(config['Physics']['Beta_con'])
Beta_vor = eval(config['Physics']['Beta_vor'])
Beta_spectrum = eval(config['Physics']['Beta_spectrum'])

#%%
# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

def vorticity_single_frame(U):
    dv_dy, dv_dx = np.gradient(U[1,:,:])
    du_dy, du_dx = np.gradient(U[0,:,:])
    vorticity = du_dy- dv_dx
    return vorticity


#%%  Create a results file 
isExist = os.path.exists("Results")
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs("Results")
   
# Create a loss file to write 
file1 = open('Results/Loss_SRResNet.txt', 'w')
file1.close()
#%%
#%% Train the generator only 
def train_srresnet(srresnet, dataloader, device, lr=1e-4, total_steps=1e6, display_step=500):
    srresnet = srresnet.to(device).train()
    optimizer = torch.optim.Adam(srresnet.parameters(), lr=lr)

    cur_step = 0
    mean_loss = 0.0
    while cur_step < total_steps:
        for hr_real, lr_real in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)

    
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                hr_fake = srresnet(lr_real)
                
                loss_U  = Loss.img_loss(hr_real, hr_fake)
                
                loss_vorticity , loss_continuity = Loss.PI_loss(hr_real, hr_fake)
                # loss_spectrum  =  Loss.Spectrum_loss(hr_real, hr_fake)  
               # print(loss_spectrum)
                g_loss  = Beta_U*loss_U + Beta_vor*loss_vorticity + Beta_con*loss_continuity  #+ Beta_spectrum *loss_spectrum
                
            if cur_step % loss_write_step == 0:
              file1 = open('Results/Loss_SRResNet.txt', 'a')
              file1.write(str(cur_step) + " ")
              file1.write(str(loss_U.cpu().detach().numpy()) + " ")
              file1.write(str(loss_vorticity.cpu().detach().numpy()) + " ")
              file1.write(str(loss_continuity.cpu().detach().numpy()) + " ")
              file1.write(str(g_loss.cpu().detach().numpy()) + " \n")
           
              file1.close()

            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()

            mean_loss += g_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                # print(loss_spectrum)
                print('Step {}: SRResNet loss: {:.5f}'.format(cur_step, mean_loss))
                # print(lr_real.size())
                # Plot u
                plt.figure(figsize=(20,10))
                plt.subplot(1,3,1)
                plt.imshow(lr_real[0,0,:,:].cpu())
                plt.subplot(1,3,2)
                plt.imshow(hr_real[0,0,:,:].cpu())
                plt.subplot(1,3,3)
                plt.imshow(hr_fake[0,0,:,:].cpu().detach().numpy())
                plt.tight_layout()
                plt.show()
                # Plot v
                plt.figure(figsize=(20,10))
                plt.subplot(1,3,1)
                plt.imshow(lr_real[0,1,:,:].cpu())
                plt.subplot(1,3,2)
                plt.imshow(hr_real[0,1,:,:].cpu())
                plt.subplot(1,3,3)
                plt.imshow(hr_fake[0,1,:,:].cpu().detach().numpy())
                plt.tight_layout()
                plt.show()
                
                # Plot vorticity
                # vor_real = 
                # dv_dx, dv_dy = np.gradient(hr_fake[0,1,:,:].cpu().detach().numpy())
                # du_dx, du_dy = np.gradient(hr_fake[0,0,:,:].cpu().detach().numpy())
                # vorticity = du_dy- dv_dx
                
                
                vorticity_hr_real = vorticity_single_frame(hr_real[0,:,:,:].cpu())
                vorticity_lr_real = vorticity_single_frame(lr_real[0,:,:,:].cpu())
                vorticity_hr_fake = vorticity_single_frame(hr_fake[0,:,:,:].cpu().detach().numpy())

                
                
                plt.figure(figsize=(20,10))
                plt.subplot(1,3,1)
                plt.imshow(vorticity_lr_real)
                plt.subplot(1,3,2)
                plt.imshow(vorticity_hr_real)
                plt.subplot(1,3,3)
                plt.imshow(vorticity_hr_fake)
                plt.tight_layout()
                plt.show()
                
                
                
                
                # show_tensor_images(lr_real * 2 - 1)
                # show_tensor_images(hr_fake.to(hr_real.dtype))
                # show_tensor_images(hr_real)
                mean_loss = 0.0

            cur_step += 1
            if cur_step == total_steps:
                break

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_ps_blocks = int(np.log2(scale_factor))
n_res_blocks = eval(config['Model']['n_res_blocks']) 
generator = Generator(base_channels=64,n_res_blocks=n_res_blocks, n_ps_blocks=n_ps_blocks)

#%% Get the dataset 
Batch_Size = eval(config['Model']['Batch_Size'])
from Dataset import*
dataset=OF_Dataset()
dataloader = torch.utils.data.DataLoader(dataset,
    batch_size=Batch_Size,  shuffle=True,pin_memory=True)
    
#%% Train the Generator first 
tic  = time.time()
train_srresnet(generator, dataloader, device, lr=1e-4, total_steps=train_iter, display_step=display_step)
toc  = time.time()

torch.save(generator, 'srresnet.pt')

np.save('Results/Train_Time', tic-toc)
#%% Save the predictions 


    
        
def predictor(Dataset_path,i):    
    U_lrs =np.load(Dataset_path+"/"+"U_lr_" + str(scale_factor)+".npy")
 
    U_hrs_pred = np.zeros((U_lrs.shape[0],U_lrs.shape[1],int(U_lrs.shape[2]*scale_factor),int(U_lrs.shape[3]*scale_factor)))
    U_lrs= np.array(U_lrs, dtype='float32')
    
 
    for t  in range(0,U_lrs.shape[0]):
    
        U_lr = torch.from_numpy(U_lrs[t,:,:,:])
        U_lr=U_lr.to(device)
        U_hr =  generator(U_lr.reshape(1,U_lrs.shape[1],U_lrs.shape[2],U_lrs.shape[3]))
        U_hrs_pred[t,:,:,:] = U_hr.cpu().detach().numpy()



    np.save('Results/U_hrs_pred_'+str(i),U_hrs_pred)

Cases_paths = [Dataset_Train_path] + Dataset_Test_path

# Cases_paths = [Dataset_Train_path,Dataset_Test_path_1,Dataset_Test_path_2,Dataset_Test_path_3]

i=0
for  path in Cases_paths: 
    predictor(path,i)
    i = i+1
















