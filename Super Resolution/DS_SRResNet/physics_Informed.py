import numpy as np 
import torch 

# This function can be deleted 

def vorticity_calc(U):
    # print(U.shape[0])
    
    dv_dx = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    dv_dy = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    du_dx = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    du_dy = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    vorticity = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))


    for i in range(0,U.shape[0]):
    
        dv_dy[i,:,:], dv_dx[i,:,:] = torch.gradient(U[i,1,:,:])
        du_dy[i,:,:], du_dx[i,:,:] = torch.gradient(U[i,0,:,:])
        vorticity[i,:,:] = du_dy[i,:,:] - dv_dx[i,:,:]
        
        
        
    return vorticity 



# This function can be deleted 
def continuity_calc(U):
    # print(U.shape[0])
    
    dv_dx = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    dv_dy = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    du_dx = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    du_dy = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    continuity = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))


    for i in range(0,U.shape[0]):
    
        dv_dy[i,:,:], dv_dx[i,:,:] = torch.gradient(U[i,1,:,:])
        du_dy[i,:,:], du_dx[i,:,:] = torch.gradient(U[i,0,:,:])
        continuity[i,:,:] = du_dx[i,:,:] + dv_dy[i,:,:]

        
    return continuity


def PI_calc(U):
    # print(U.shape[0])
    
    dv_dx = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    dv_dy = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    du_dx = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    du_dy = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    vorticity = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    continuity = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))


    for i in range(0,U.shape[0]):
    
        dv_dy[i,:,:], dv_dx[i,:,:] = torch.gradient(U[i,1,:,:])
        du_dy[i,:,:], du_dx[i,:,:] = torch.gradient(U[i,0,:,:])
        vorticity[i,:,:] = du_dy[i,:,:] - dv_dx[i,:,:]
        continuity[i,:,:] = du_dx[i,:,:] + dv_dy[i,:,:]
        
        
        
        
    return vorticity , continuity





# This function is not yet complete 
def omni_spectrum_calc(U):
    # spectrum_2D = torch.zeros((U.shape[0],U.shape[2],U.shape[3]))
    spectrum_2D = torch.zeros((U.shape[0],64,256))
    for i in range(0,U.shape[0]):
        spectrum_2D[i,:,:] = torch.fft.fft(U[i,0,0:64,0:256])

    
    spectrum_2D_averaged = torch.sum(torch.mean(spectrum_2D,0),0)
    spectrum_2D_averaged=spectrum_2D_averaged/torch.max(spectrum_2D_averaged)
    return spectrum_2D_averaged










