import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16

#%%  Inputs 
u_rms_Target = 0.5
Re_0_Target = 25000

kp = 12
N= 1024
#%%  Computations
k = np.linspace(0.01,N,N)
s=4
E_Foam  = 0.5*(u_rms_Target**2)*((k/kp)**s)*np.exp(-s/2*(k/kp)**2)*(N)
E_Foam_Int = np.trapz(E_Foam,k)
nu  = (u_rms_Target * 2*np.pi/kp)/Re_0_Target
#%%  Outputs 
print('Ea = ',E_Foam_Int)
print('nu = ',nu)




