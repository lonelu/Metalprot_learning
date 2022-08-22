"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains classes for Metalprot_learning models and datasets. 
"""

#imports
import torch.nn as nn
import torch.nn.functional as F
import torch

def compute_channel_dims(in_width: int, config: dict):
    """
    Utility function for computing dimensionality of features after convolution.
    """
    for layer in [x for x in config.keys() if x in ['block0', 'block1', 'block2']]:
        kernel_size = config[layer]['kernel_size'] 
        padding=config[layer]['padding']
        _out_width = ((in_width + (2 * padding) - kernel_size) // 1) + 1
        if 'kernel_size_pool' in config[layer].keys():
            kernel_size_pool = config[layer]['kernel_size_pool']
            out_width = ((_out_width - kernel_size_pool) // 2) + 1
        else:
            out_width = _out_width
        in_width = out_width
    return in_width

class ConvBn2d(nn.Module):
    """
    Utility class for 2d convolution with a batch norm.
    """
    def __init__(self, in_channel, out_channel, kernel_size, padding=0, dilation=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SwishFunction(torch.autograd.Function):
    """
    Utility class for swish activation function.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.sigmoid(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        return grad_output * (sigmoid * (1 + x * (1 - sigmoid)))

F_swish = SwishFunction.apply

class Swish(nn.Module):
    """
    Second utility class for swish activation function.
    """
    def forward(self, x):
        return F_swish(x)

class Residual(nn.Module):
    def __init__(self,in_channel, dilation):
        super(Residual, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=dilation, dilation=dilation, bias=False) #given that the height and width of the output must be the same as that of the input, the kernel size must always be 3
        self.bn   = nn.BatchNorm2d(in_channel)
        self.act  = Swish()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x+residual
        return x

class AlphafoldNet(nn.Module):
    """
    Class for Alphafold-like neural network.
    """
    def __init__(self, config: dict):
        super(AlphafoldNet, self).__init__()
        self.encodings = config['encodings']
        self.block_n1 = nn.Sequential(
            ConvBn2d(config['block_n1']['in'], config['block_n1']['out'], kernel_size=(2 * config['block_n1']['padding']) + 1, padding=config['block_n1']['padding']), 
            Swish(), 
            nn.Dropout(config['block_n1']['dropout_n1']),
        ) #based on the constraint that the width and height of the output must be 12, the kernel size and padding are coupled
        self.block0 = nn.Sequential(
            ConvBn2d(4 + config['block_n1']['out'], config['block0']['out'], kernel_size=config['block0']['kernel_size'], padding=config['block0']['padding']),
            Swish(), 
            nn.Dropout(config['block0']['dropout_0']),
        )
        self.block1 = nn.Sequential(
            Residual(config['block0']['out'], dilation=config['block1']['dilation_residual']),
            Residual(config['block0']['out'], dilation=config['block1']['dilation_residual']),
            ConvBn2d(config['block0']['out'], config['block1']['out'], kernel_size=config['block1']['kernel_size'], padding=config['block1']['padding']),
            Swish(), 
            nn.MaxPool2d(kernel_size=config['block1']['kernel_size_pool']),
            nn.Dropout(config['block1']['dropout_1']),
        )
        self.block2 = nn.Sequential(
            Residual(config['block1']['out'], dilation=config['block2']['dilation_residual']),
            Residual(config['block1']['out'], dilation=config['block2']['dilation_residual']),
            ConvBn2d(config['block1']['out'], config['block2']['out'], kernel_size=config['block2']['kernel_size'], padding=config['block2']['padding']),
            Swish(), 
            nn.MaxPool2d(kernel_size=config['block2']['kernel_size_pool']),
            nn.Dropout(config['block2']['dropout_2']),
        )
        self.block3 = nn.Sequential(
            Residual(config['block2']['out'], dilation=config['block3']['dilation_residual']),
            Residual(config['block2']['out'], dilation=config['block3']['dilation_residual']),
            nn.Dropout(config['block3']['dropout_3']),
        )
        self.linear1 = nn.Sequential(nn.Linear((compute_channel_dims(12, config) ** 2) * config['block2']['out'],config['linear1']['out']),
            nn.Dropout(config['linear1']['dropout_l1']))
        self.linear2 = nn.Sequential(nn.Linear(config['linear1']['out'],config['linear2']['out']), 
            nn.Dropout(config['linear2']['dropout_l2']))
        self.linear3 = nn.Linear(config['linear2']['out'],config['linear3']['out'])

    def forward(self, x):
        x = x.float()
        x1 = x[:, :4, :, :] #the backbone atom distance matrix channels for a batch
        x2 = x[:, 4:, :, :] #the sequence encoding channels for a batch
        x2 = self.block_n1(x2)              
        x = torch.cat([x1,x2],1)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.dropout(x,0.5, training=self.training)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
