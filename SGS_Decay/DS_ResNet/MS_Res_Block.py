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
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.PReLU()
        
        self.BN_1  = nn.BatchNorm2d(n_feats)
        self.BN_2  = nn.BatchNorm2d(int(n_feats*2))
       

    def forward(self, x):
        input_1 = x
        
        output_3_1 = self.conv_3_1(input_1)
        # output_3_1 = self.BN_1(output_3_1)
        output_3_1 = self.relu(output_3_1)
        
        
        output_5_1 = self.conv_5_1(input_1)
        # output_5_1 = self.BN_1(output_5_1)
        output_5_1 = self.relu(output_5_1)
        
        
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        
        output_3_2 = self.conv_3_2(input_2)
        # output_3_2 = self.BN_2(output_3_2)
        output_3_2 = self.relu(output_3_2)
        
        
        output_5_2 = self.conv_5_2(input_2)
        # output_5_2 = self.BN_2(output_5_2)
        output_5_2 = self.relu(output_5_2)
        
        
        
        
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        
        # output = self.BN_1(output)
        
        
        output += x
        output = self.relu(output)
        return output













