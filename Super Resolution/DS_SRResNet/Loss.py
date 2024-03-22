import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  physics_Informed import*
#%%

from torchvision.models import vgg19

class Loss(nn.Module):
    '''
    Loss Class
    Implements composite content+adversarial loss for SRGAN
    Values:
        device: 'cuda' or 'cpu' hardware to put VGG network on, a string
    '''

    def __init__(self, device='cuda'):
        super().__init__()

        vgg = vgg19(pretrained=True).to(device)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    @staticmethod
    def img_loss(x_real, x_fake):
        # print(x_real.shape)
        return F.mse_loss(x_real, x_fake)
    
    
   
    # This function should be appended by strain, enstrophy and other flow parameters
    def PI_loss(x_real, x_fake):

        vorticity_real,continuity_real =  PI_calc(x_real)
        vorticity_fake,continuity_fake =  PI_calc(x_fake)
        vorticity_error =  F.mse_loss(vorticity_real, vorticity_fake)
        
        
        continuity_error =  F.mse_loss(continuity_real, continuity_fake)
        return vorticity_error , continuity_error

    def Spectrum_loss(x_real, x_fake):
        x_real_spectrum = omni_spectrum_calc(x_real)
        x_fake_spectrum = omni_spectrum_calc(x_fake)
        spectrum_error =  F.mse_loss(x_real_spectrum, x_fake_spectrum)
        
        return spectrum_error
    

    def adv_loss_NEW(x, is_real):
        target = torch.zeros_like(x) if is_real else torch.ones_like(x)
        return F.binary_cross_entropy_with_logits(x, target)
    
    def adv_loss(self, x, is_real):
        target = torch.zeros_like(x) if is_real else torch.ones_like(x)
        return F.binary_cross_entropy_with_logits(x, target)



    def vgg_loss(self, x_real, x_fake):
        return F.mse_loss(self.vgg(x_real), self.vgg(x_fake))

    def forward(self, generator, discriminator, hr_real, lr_real):
        ''' Performs forward pass and returns total losses for G and D '''
        hr_fake = generator(lr_real)
        fake_preds_for_g = discriminator(hr_fake)
        fake_preds_for_d = discriminator(hr_fake.detach())
        real_preds_for_d = discriminator(hr_real.detach())

        g_loss = (
            0.001 * self.adv_loss(fake_preds_for_g, False) + \
            # 0.006 * self.vgg_loss(hr_real, hr_fake) + \
            self.img_loss(hr_real, hr_fake)
        )
        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) + \
            self.adv_loss(fake_preds_for_d, False)
        )

        return g_loss, d_loss, hr_fake
