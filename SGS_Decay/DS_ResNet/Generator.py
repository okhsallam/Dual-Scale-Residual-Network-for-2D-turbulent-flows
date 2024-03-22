import torch
import torch.nn as nn
import torch.nn.functional as F
# from Res_Block import ResidualBlock
from MS_Res_Block import ResidualBlock

import configparser
config = configparser.ConfigParser()
Config_file='Config.ini'
config.read(Config_file)
###

Dimension = eval(config['Model']['Dimension'])

#%%
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        base_channels: number of channels throughout the generator, a scalar
        n_ps_blocks: number of PixelShuffle blocks, a scalar
        n_res_blocks: number of residual blocks, a scalar
    '''

    def __init__(self, base_channels=64, n_res_blocks=16):
        super().__init__()
        # Input layer
        self.in_layer = nn.Sequential(
            # nn.Conv2d(3, base_channels, kernel_size=9, padding=4),  # Shoud be 3 for 3D problems 
            nn.Conv2d(int(Dimension), base_channels, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_res_blocks):
            res_blocks += [ResidualBlock(base_channels)]

        res_blocks += [
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1), # Padding  = kernel size // 2      
            nn.BatchNorm2d(base_channels),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)


        # Output layer
        self.out_layer = nn.Sequential(
            # nn.Conv2d(base_channels, 3, kernel_size=9, padding=4),  # Shoud be 3 for 3D problems 
            nn.Conv2d(base_channels, int(Dimension), kernel_size=9, padding=4),
            # nn.Tanh(),
          #  nn.LazyLinear(300)


        )

    def forward(self, x):
        x_res = self.in_layer(x)
        x = x_res + self.res_blocks(x_res)
        x = self.out_layer(x)
        # print(x.shape)
        return x
