#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Risk score ranking loss where (Im,In) image pairs are formed across all images in the training 
batch across patients. This is used only when the conversion time labels tcnv are available
'''

def my_risk_concordance_loss(risk, indctr, tcnv):
    # Which pairs can be compared
    # 1. tcnv_row< tcnv_col
    # 2. (i)indctr_row should not be censored    or   (ii) row & col come from same eye
    # censored = 1 => conversion event happened, 0=> censored
    
    ###### eye_flag is a unique index id s.t. all rsks of scans of the same patient have the same id ###
    B=indctr.shape[0]//3 # rsk1 from img1 of pair, rsk2 from img2 of pair, rsk3 from ODE 
    eye_flag=torch.arange(start=0, end=B, step=1).to(device) # 0 to B-1
    eye_flag=torch.cat((eye_flag, eye_flag, eye_flag), dim=0) # 3B   
    eye_flag=torch.unsqueeze(eye_flag, dim=1) # 3B,1
    ######################################################
    eye_flag_row=eye_flag.repeat(1,(3*B)) # 3B, 3B  repeat a column 3B times
    eye_flag_row=eye_flag_row.view(-1) # 9B^2
    
    eye_flag_col=torch.transpose(eye_flag, 0,1) # 1,3B
    eye_flag_col=eye_flag_col.repeat((3*B),1) # 3B, 3B repeat a row 3B times
    eye_flag_col=eye_flag_col.view(-1) # 9B^2
    
    ### repeat the process for tcnv
    tcnv_row=tcnv.repeat(1, (3*B)) # 3B,3B
    tcnv_row=tcnv_row.view(-1) # 9B^2
    
    tcnv_col=torch.transpose(tcnv, 0,1) # 1,3B
    tcnv_col=tcnv_col.repeat((3*B), 1) # 3B, 3B
    tcnv_col=tcnv_col.view(-1) # 9B^2
        
    ### repeat the process for indictr
    indctr_row=indctr.repeat(1, (3*B)) # 3B,3B
    indctr_row=indctr_row.view(-1)  # 9B^2
    
    indctr_col=torch.transpose(indctr, 0,1) # 1,3B
    indctr_col=indctr_col.repeat((3*B),1)   # 3B,3B
    indctr_col=indctr_col.view(-1)  # 9B^2
        
    ### repeat the process for risk scores 
    risk_row=risk.repeat(1, (3*B)) # 3B,3B
    risk_row=risk_row.view(-1)  # 9B^2
    
    risk_col=torch.transpose(risk, 0, 1) # 1,3B
    risk_col=risk_col.repeat((3*B), 1) # 3B,3B
    risk_col=risk_col.view(-1)   # 9B^2
    
    ######################################################
    # compute the difference in risk
    risk_mtrx=(risk_row-risk_col) # 9B^2    
    #############################################
    
    # 1. Choose pairs where tcnv_row< tcnv_col, and 
    # 2. Either (i)indctr_row should not be censored    or   (ii) row & col come from same eye
    # Event indicator 1=>event occured; 0=>censored
    
    idx=(tcnv_row<tcnv_col) & ( (indctr_row==1) | (eye_flag_row==eye_flag_col) )
    risk_mtrx=risk_mtrx[idx] # P
    del idx
    ##############################################################################################
    risk_mtrx=torch.unsqueeze(risk_mtrx,dim=1) # P,1
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
    risk_mtrx=F.sigmoid(risk_mtrx)
    loss=F.binary_cross_entropy_with_logits(risk_mtrx, gt)
    return loss











'''
Risk score ranking loss to be used for the unsupervised Domain Adaptation.
We only consider Ij, Ik pairs used during training which are two visits of the same eye (each sample in a batch)
'''
def my_risk_concordance_loss_unsupervised(risk1, risk2):    
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
    risk_mtrx=F.sigmoid(risk_mtrx)
    loss=F.binary_cross_entropy_with_logits(risk_mtrx, gt)
    return loss
    

