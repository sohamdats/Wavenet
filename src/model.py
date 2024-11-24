import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math

import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            device = m.weight.device  # Get the current device of the module
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class GatedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):      
  
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class CausalConv(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.padding = (kernel_size - 1)
        kernel_size = (1, kernel_size)
        self.conv = nn.Conv2d(
            dim, dim, kernel_size, 1, padding=0
        )
 
    def forward(self, x):

        x = F.pad(x, (self.padding, 0))
        h = self.conv(x)     
        output = h[:, :, :, :x.size(-1)]         
        return output


class DilatedConvBlock(nn.Module):
    
    def __init__(self, dim, kernel_size, dilation, n_classes):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.dilat_conv = nn.Conv2d(dim, 
                                    dim * 2, 
                                    (1, kernel_size), 
                                    1, 
                                    padding=0, 
                                    dilation=dilation)

        self.gate = GatedNetwork()
        # self.expand_conv = nn.Conv2d(dim, dim*2, 1)
        
        self.class_cond_embedding = nn.Embedding(n_classes, dim*2)

    def forward(self, x, label):
        
        input_x = x
        
        h_label = self.class_cond_embedding(label)
        h_temp = F.pad(x, (self.padding, 0))  
        h = self.dilat_conv(h_temp)

        h = h[:, :, :, :input_x.size(-1)]
       
        output = self.gate(h + h_label[:, :, None, None])
        return output + input_x

        
class Wavenet(nn.Module):
    
    def __init__(self, input_dim=256, dim = 64, num_layers=15, n_classes=5):
        super().__init__()
        
        
        self.audio_sample_embeddings = nn.Embedding(input_dim, dim)
        self.causal_conv = CausalConv(dim=dim, kernel_size=7)
        
        self.blocks = nn.ModuleList()
        
        self.final_layer = nn.Sequential(
            nn.ReLU(True), 
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True), 
            nn.Conv2d(512, input_dim, 1)
        )
        
        dilation = 1
        for i in range(num_layers):
            
            dilation *= 2
            if dilation > 512: 
                dilation = 1
                
            self.blocks.append(
                DilatedConvBlock(dim, kernel_size=7, dilation=dilation, n_classes=n_classes)
                )
            
        self.final_gate = GatedNetwork()
        
        # self.apply(weights_init)

    def forward(self, x, label):
        
        shp = x.size() + (-1, )
        x = self.audio_sample_embeddings(x.view(-1)).view(shp)
        x = x.permute(0, 3, 1, 2).contiguous()
        
          
        h = self.causal_conv(x)
  
        for conv_idx, conv in enumerate(self.blocks):
            h = conv(h, label)
        
        h = self.final_layer(h)
    
        return h
    
    def generate(self, shape=(1, 190000), batch_size=32):
        
        param = next(self.parameters())
        label = torch.zeros(batch_size, dtype=torch.int64, device=param.device)

        x = torch.zeros(
            batch_size, *shape, 
            dtype=torch.int64, device=param.device
        )
        
        for i in range(shape[0]):
            for  j in range(shape[1]):
                
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], dim=-1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
                
        return x

