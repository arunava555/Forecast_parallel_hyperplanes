#!/usr/bin/env python


import torch.nn as nn
import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights 
from einops import rearrange#, reduce, repeat
import numpy as np





##################################################################################################
class Encoder_Network2D(nn.Module):
    def __init__(self):
        super(Encoder_Network2D, self).__init__()
        model=convnext_tiny(weights='IMAGENET1K_V1')
        
        ################ Remove the classification Layer #######################################
        self.model=nn.Sequential(*(list(model.children())[:-1])) 
        
            
    def forward(self, img):
        ftr=torch.squeeze(self.model(img), dim=[2,3]) # 768 dimensional feature
        return ftr
        
#################################################################################################



##################################################################################################
class Classification_Network(nn.Module):
    def __init__(self):
        super(Classification_Network, self).__init__()
        ##### New Classification Layer
        self.fc1=nn.Linear(in_features=768, out_features=1, bias=True)
        self.fc2=nn.Linear(in_features=1, out_features=1, bias=False)
        self.fc3=nn.Linear(in_features=1, out_features=1, bias=False)
        
        self.gamma=nn.Parameter(data=torch.from_numpy(np.array([1.0])).to(dtype=torch.float32), requires_grad=True)
        self.fc_gamma=nn.Linear(in_features=1, out_features=1, bias=True)
        
    def forward(self, ftr, t):
        # t: B,N
        r=self.fc1(ftr)
        risk=self.fc2(r)
        pred=self.fc3(r)
        
        ###############################################
        B, T=t.shape
        t=rearrange(t, 'b t -> (b t) 1')
        bias=self.fc_gamma(t)
        
        bias=rearrange(bias, '(b t) 1 -> b t', b=B, t=T) # B,t
        ################################################
        
        pred=pred+bias
        return risk, pred
#################################################################################################

