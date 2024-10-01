#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[11]:


def my_risk_concordance_loss_paired(risk1, risk2):
    # Which pairs can be compared
    risk_mtrx=risk2-risk1 # risk1<risk2 . 
    ####################################################
    gt=torch.ones_like(risk_mtrx).to(device)
    ####################################################
    # Now randomly reverse the risk matrix and the gt for half of the pairs
    r=torch.randint_like(risk_mtrx, low=0, high=2) # 0 or 1
    idx=(r==1)
    ###################################################
    gt[idx]=0
    ###################################################
    risk_mtrx[idx]=risk_mtrx[idx]*-1
    ######
    # tcnv_row<tcnv_col and tcnv_row is not censored, ie. it is conversion date. 
    # So, row converts before col => risk_row should be >risk_col. Then
    # (risk_row-risk_col) should be 
        #>0 for correct ordering(GT=1) & 
        #<0 for incorrect ordering(GT=0, if the risk order is reversed).
        # Sigmoid makes <0 in range [0 to 0.5]    >0 in range [0.5 to 1]
    risk_mtrx=F.sigmoid(risk_mtrx)
    loss=F.binary_cross_entropy_with_logits(risk_mtrx, gt)
    return loss
    

