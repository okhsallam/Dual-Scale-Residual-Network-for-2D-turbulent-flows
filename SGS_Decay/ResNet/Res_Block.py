import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)



class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class
    Values
        channels: the number of channels throughout the residual block, a scalar
    '''

    def __init__(self, n_feats=64):
        super(ResidualBlock, self).__init__()
        
       
        

        kernel_size_1 = 3


        self.conv = conv(n_feats, n_feats, kernel_size_1)

        self.relu = nn.PReLU()
        
        self.conv_1_1 =  conv(n_feats, n_feats, 1)
        self.BN_1  = nn.BatchNorm2d(n_feats)

    def forward(self, x):
    
        
        output = self.conv(x)
        # output = self.BN_1(output)
        output = self.relu(output)

        output = self.conv(output)
        # output = self.BN_1(output)
        output += x

        
        
        return output













