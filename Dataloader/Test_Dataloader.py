#!/usr/bin/env python
# coding: utf-8

from scipy.special import comb
from torch.utils.data import Dataset
import os
import numpy as np
import torch


class test_dataset(Dataset):
    def __init__(self):
        pwd=os.getcwd()
        #print('pwd from inside the test dataloader: '+str(pwd))
        
        # load appropriate set of images for training
        a=np.load(pwd+'/Dataloader/data_splits/tst.npz') # contains details of split of the test set
        self.sampling_index=a['sampling_index'] # Used for eye-level bootstrapping.
        self.eye_lst=a['eye_lst'] # Eye_ID
        self.vis_lst=a['vis_lst'] # which visit
        self.tcnv_lst=a['tcnv_lst'] # the time to the first conversion visit
        self.indctr_lst=a['indctr_lst'] # whether conversion occurs (1) or censored (0)
        self.gt_lst=a['gt_lst'] # GT labels (0/1) indicating conversion within fixed time-points (6/12/18/24/... months)
        self.nm_lst=a['nm_lst'] # name of the npz file containing the preprocessed image (central 5-Bscans)
        del a
        ###########################################################
        # 
        self.pth='/msc/home/achakr83/MICCAI24/Mar1/preprocess_PINNACLE_Data/preprocessed_five_Bscans/'
        self.dct_img, self.dct_ll, self.dct_ul= self.pre_load_data()
        print(self.nm_lst.shape[0])
    
    
    ###########################################################################################################
    
    def pre_load_data(self):
        # A list of all eyes "eye__vis"
        p=1.0 # What fraction of images to pre-load
        # A fraction of images are loaded in the RAM for faster computation.Alternatively,make p=0
        
        # Now load top p*eye_vis.shape[0]
        dct_img={}
        dct_ll={}
        dct_ul={}
        sz=int(p*self.nm_lst.shape[0])
        
        for k in range(0, sz):
            #print(k)
            a=np.load(self.pth+self.nm_lst[k]+'.npz')
            img=a['img']
            ll=a['ll']
            ul=a['ul']
            del a
            
            
            dct_img[self.nm_lst[k]]=img
            dct_ll[self.nm_lst[k]]=ll
            dct_ul[self.nm_lst[k]]=ul
            del img, ll, ul
        
        return dct_img, dct_ll, dct_ul
        
    ##########################################################################################################
    
    def data_augmentation(self, img, nm, ll, ul):
        
        ################# Random Crop top-bottom,  left-right  ###################
        mn_ht=12#random.randint(0,23)
        mn_col=12#random.randint(0,23) # 247 to be cropped to 224: 23
        
        if ((mn_ht>ll) and (ll>0)):
            mn_ht=ll
        
        img=img[mn_ht:mn_ht+224, mn_col:mn_col+224,:]  # Nb: These scans have central 5 B-scans
        
        ############ Normalize intensity in range [0,1]  #########################
        img=img.astype(np.float32)
        if np.max(img)>300:
            img=img/65535  # then most probably it is uint16
        else:
            img=img/255 # uint8
        #######
        img=(img*2)-1   # range -1,1
        ##########################################################################
        # Add channel dimension
        img=np.transpose(img, axes=(2, 0, 1)) # 5,H,W
        img=np.expand_dims(img, axis=0) # 1,5,H,W        
        return img
    
    ##########################################################################################################
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""     
        
        ##### Load image ##############
        eye=self.eye_lst[index]
        #vis=self.vis_lst[index]
        nm=self.nm_lst[index]
        tcnv=self.tcnv_lst[index]       # only used for metrics as numpy 
        indctr=self.indctr_lst[index]   # only numpy
        gt=self.gt_lst[index]  # for validation, it will only be used as numpy not pytorch tensor
        
        
        if self.dct_img[nm] is None:
            a=np.load(self.pth+nm+'.npz')
            img=a['img']
            ll=a['ll']
            ul=a['ul']
            del a
        else:
            # img is pre-loaded in RAM
            img=self.dct_img[nm]
            ll=self.dct_ll[nm]
            ul=self.dct_ul[nm]
        
        
        img=self.data_augmentation(img, eye[-2:], ll, ul)
        img=torch.FloatTensor(img)
        
        ########## tcnv is in days : unnormalized ########
        # The central 5 B-scans are divided into three 3 channel images with consecutive slices
        # Average of their predictions is used as the final prediction.
        sample={'img1': img[0:1, 0:3,:,:], 'img2':img[0:1,1:4,:,:], 'img3': img[0:1, 2:5,:,:], 
                'gt':gt, 'tcnv': tcnv, 'indctr': indctr, 'nm': nm}
        return sample 

    def __len__(self):
        
        return self.nm_lst.shape[0]

