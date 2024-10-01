#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.special import comb
from torch.utils.data import Dataset
import os
import pickle
import random 
from scipy.ndimage import zoom
from skimage import filters
import numpy as np
import torch


class validation_dataset(Dataset):
    def __init__(self):
        pwd=os.getcwd()        
        # load appropriate set of images for training
        a=np.load(pwd+'/Dataloader/data_splits/val.npz') # contains details of split of the validation set
        self.sampling_index=a['sampling_index'] #  Used for eye-level bootstrapping.
        self.eye_lst=a['eye_lst']  #  Eye_ID
        self.vis_lst=a['vis_lst']  # visit date
        self.tcnv_lst=a['tcnv_lst'] # the time to the first conversion visit
        self.indctr_lst=a['indctr_lst'] # whether conversion occurs (1) or censored (0)
        self.gt_lst=a['gt_lst']  # GT labels (0/1) indicating conversion within fixed time-points (6/12/18/24/... months)
        self.nm_lst=a['nm_lst']  # filename containing preprocessed scan eyeid__visitdate.npz
        del a
        ###########################################################
        self.pth='/msc/home/achakr83/MICCAI24/Mar1/preprocess_PINNACLE_Data/preprocessed_five_Bscans/' # path containing preprocessed scans
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
        
        img=img[mn_ht:mn_ht+224, mn_col:mn_col+224,1:4] 
        
        ############ Normalize intensity in range [0,1]  #########################
        img=img.astype(np.float32)
        if np.max(img)>300:
            img=img/65535  # then most probably it is uint16
        else:
            img=img/255 # uint8
        #######
        img=(img*2)-1
        
        # Add channel dimension
        img=np.transpose(img, axes=(2, 0, 1)) # 3,H,W
        img=np.expand_dims(img, axis=0) # 1,3,H,W        
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
        sample={'img': img, 'gt':gt, 'tcnv': tcnv, 'indctr': indctr, 'nm': nm}
        return sample 

    def __len__(self):
        
        return self.nm_lst.shape[0]


# In[4]:


################ These are transformations used for Data-Augmentation #####################

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


######################################################################################################

def nonlinear_transformation(x):
    
    points = [[0, 0], [random.random()*0.3, random.random()*0.3], [random.random()*0.3, random.random()*0.3], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    
    xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

########################################################################################################




def image_in_painting(x):
    img_rows, img_cols, img_deps=x.shape
    # number of patches to inpaint: random between 0 to mx_cnt
    mx_cnt=5
    cnt=random.randint(0, mx_cnt)
    
    for k in range(0, cnt):
        # ht of block to be inpainted randomly varies between img_rows/xx and img_rows/yy
        block_noise_size_x = random.randint(img_rows//32, img_rows//8) # row size of the block
        block_noise_size_y = random.randint(img_cols//32, img_cols//8) # col size of the block
        block_noise_size_z = random.randint(1,2)

        # the starting point (top-left corner) of the block to be inpainted
        noise_x = random.randint(3, img_rows-block_noise_size_x-3) # 3 px margin on all sides
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        
        x[noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y,
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, block_noise_size_y, block_noise_size_z)
    return x

#######################################################################################################


def randomCrop(img):
    
    r1=float(random.randint(8, 10))/10.0  # 80-100% of original eye
    height=int(r1*img.shape[0])
    # random aspect ratio:
    r2=float(random.randint(8, 12))/10.0  # 80-120% of height (as long as it is within limit)
    width=int(np.minimum(img.shape[1], r2*height))
    
    #height=# x2 % of img.shape[0]
    #width=# x1*x2 % of img.shape[1]
    
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width,:]
    return img


# In[5]:


class train_dataset(Dataset):
    def __init__(self, fold, prcnt, discard_converted=False):
        # preprocessed image paths
        self.pth='/msc/home/achakr83/MICCAI24/Mar1/preprocess_PINNACLE_Data/preprocessed_five_Bscans/'
        pkl_pth=os.getcwd()+'/Dataloader/data_splits/'
        
        # stratified_trn_prcnt25_fold2.pickle
        with open(pkl_pth+'stratified_trn_prcnt'+str(prcnt)+'_fold'+str(fold)+'.pickle', 'rb') as handle:
            a = pickle.load(handle)
        
        self.cnvrt0_6=a[0]
        self.cnvrt6_12=a[1] 
        self.cnvrt12_18=a[2]
        self.cnvrt18_24=a[3]
        self.cnvrt24_30=a[4]
        self.cnvrt30_36=a[5]
        
        self.nocnv0_6=a[6]
        self.nocnv6_12=a[7]
        self.nocnv12_18=a[8]
        self.nocnv18_24=a[9]
        self.nocnv24_30=a[10]
        self.nocnv30_36=a[11]
        del a
        
        ##################################
        self.epoch_size=250
        ### Full nm_lst ###
        nm_lst=np.concatenate((self.cnvrt0_6['nm_lst'], self.cnvrt6_12['nm_lst'], self.cnvrt12_18['nm_lst'],
                               self.cnvrt18_24['nm_lst'], self.cnvrt24_30['nm_lst'], self.cnvrt30_36['nm_lst'],
                               self.nocnv0_6['nm_lst'], self.nocnv6_12['nm_lst'], self.nocnv12_18['nm_lst'],
                               self.nocnv18_24['nm_lst'], self.nocnv24_30['nm_lst'], self.nocnv30_36['nm_lst']),
                              axis=0)
        count=np.concatenate((self.cnvrt0_6['nm_freq'], self.cnvrt6_12['nm_freq'], self.cnvrt12_18['nm_freq'],
                               self.cnvrt18_24['nm_freq'], self.cnvrt24_30['nm_freq'], self.cnvrt30_36['nm_freq'],
                               self.nocnv0_6['nm_freq'], self.nocnv6_12['nm_freq'], self.nocnv12_18['nm_freq'],
                               self.nocnv18_24['nm_freq'], self.nocnv24_30['nm_freq'], self.nocnv30_36['nm_freq']),
                              axis=0)
        
        idx=np.argsort(-count) # ascending -count= descending count
        self.full_nm_lst=nm_lst[idx]
        del nm_lst, idx, count
        
        self.dct_img, self.dct_ll, self.dct_ul = self.pre_load_data() 
        self.std=0.001
        
    ################################################################################################################
    
    def pre_load_data(self):
        # A list of all eyes "eye__vis"
        p=1.0 # What fraction of images to pre-load
        
        # Now load top p*eye_vis.shape[0]
        dct_img={}
        dct_ll={}
        dct_ul={}
        sz=int(p*self.full_nm_lst.shape[0])
        #img_pth='/'
        for k in range(0, sz):
            a=np.load(self.pth+self.full_nm_lst[k]+'.npz')
            img=a['img']
            ll=a['ll']
            ul=a['ul']
            del a
            
            dct_img[self.full_nm_lst[k]]=img
            dct_ll[self.full_nm_lst[k]]=ll
            dct_ul[self.full_nm_lst[k]]=ul
            del img, ll, ul
        
        return dct_img, dct_ll, dct_ul
            
    
    ################################################################################################################
    
    def random_sampling(self, dct):
        eye_lst=dct['eyes']
        r=np.random.randint(0, eye_lst.shape[0])
        eye=eye_lst[r]
        
        vis1_lst=dct['vis1_lst'][r]
        vis2_lst=dct['vis2_lst'][r]
        gt1_lst=dct['gt1_lst'][r]
        gt2_lst=dct['gt2_lst'][r]
        tcnv1_lst=dct['tcnv1_lst'][r]
        tcnv2_lst=dct['tcnv2_lst'][r]
        indctr1_lst=dct['indctr1_lst'][r]
        indctr2_lst=dct['indctr2_lst'][r]
        tintrvl_lst=dct['tintrvl_lst'][r]
        del dct, r
        
        
        r=np.random.randint(0, vis1_lst.shape[0])
        vis1=vis1_lst[r]
        vis2=vis2_lst[r]
        gt1=gt1_lst[r]
        gt2=gt2_lst[r]
        tcnv1=tcnv1_lst[r]
        tcnv2=tcnv2_lst[r]
        indctr1=indctr1_lst[r]
        indctr2=indctr2_lst[r]
        tintrvl=tintrvl_lst[r]
        del r
        del eye_lst, vis1_lst, vis2_lst, gt1_lst, gt2_lst, tcnv1_lst, tcnv2_lst, 
        del indctr1_lst, indctr2_lst, tintrvl_lst
        
        ######### Now read the image from the npz file using eye #########
        if self.dct_img[eye+'__'+vis1] is None:
            print('\n img not pre-loaded')
            a=np.load(self.pth+eye+'__'+vis1+'.npz')
            img1=a['img']
            ll1=a['ll']
            ul1=a['ul']
            del a
        else:
            # img is pre-loaded in RAM
            img1=self.dct_img[eye+'__'+vis1]
            ll1=self.dct_ll[eye+'__'+vis1]
            ul1=self.dct_ul[eye+'__'+vis1]
            
        
        if self.dct_img[eye+'__'+vis2] is None:
            print('\n img not pre-loaded')
            a=np.load(self.pth+eye+'__'+vis2+'.npz')
            img2=a['img']
            ll2=a['ll']
            ul2=a['ul']
            del a
        else:
            # img is pre-loaded in RAM
            img2=self.dct_img[eye+'__'+vis2]
            ll2=self.dct_ll[eye+'__'+vis2]
            ul2=self.dct_ul[eye+'__'+vis2]
        
        ###################################################################
        img1=self.data_augmentation(img1, eye[-2:], ll1, ul1) # last 2 alphabets of name : either OD or OS
        img2=self.data_augmentation(img2, eye[-2:], ll2, ul2)
        
        #print('img1 shp: '+str(img1.shape)+'  img2 shp: '+str(img2.shape))
        
        return img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        
    #################################################################################################################
    
    def data_augmentation(self, img, nm, ll, ul):
        
        ################## Align all images to OD through flipping  ###############
        #if nm=='OS':
            # flip to OD
        #    img=img[:,::-1]
        
        # random flip
        if random.random()>0.5:
            img=img[:,::-1] # random flip to OS
        ################# Random Crop top-bottom,  left-right  ###################
        mn_ht=random.randint(0,23)
        mn_col=random.randint(0,23) # 247 to be cropped to 224: 23
        mn_slc=random.randint(0,2)
        
        if ((mn_ht>ll) and (ll>0)):
            mn_ht=ll
        
        img=img[mn_ht:mn_ht+224, mn_col:mn_col+224, mn_slc:mn_slc+3] 
        
        ############ Normalize intensity in range [0,1]  #########################
        img=img.astype(np.float32)
        if np.max(img)>300:
            img=img/65535  # then most probably it is uint16
        else:
            img=img/255 # uint8
        ##########################################################################
        # random blurring of image for data aug
        # random noise for data augmentation
        # contrast change
        if (random.random()>0.5):
            img=nonlinear_transformation(img)
            
        img=(img*2)-1
                
        if (random.random()>0.5):
            img=img + (np.random.randn(img.shape[0], img.shape[1], img.shape[2])*self.std)
        
        
        if (random.random()>0.5):
            r=random.random()*1.0 # between [0,1]
            #print(img.shape)
            img = filters.gaussian(img, sigma=(r, r, r), truncate=3, channel_axis=2) # r,r,r for 3D
        
        
        # Random crop resize ?
        # Randomly crop between 80-100% of the image size and then resize to 224 X224
        # random masking from model genesis
        if (random.random()>0.5):
            img=randomCrop(img)
        
        # Random masking regions in the image
        if (random.random()>0.1):
            img=image_in_painting(img)
            
        # resize to 224,224
        scl_ht=224.0/img.shape[0]
        scl_wd=224.0/img.shape[1]
        img=zoom(img, (scl_ht,scl_wd, 1))
        
        
        # Add channel dimension
        img=np.transpose(img, axes=(2, 0, 1)) # 3,H,W
        img=np.expand_dims(img, axis=0) # 1,3,H,W   
                
        return img
    
    
    ###########################################################################################################
    
    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""     
        
        img1_lst=[]
        img2_lst=[]
        gt1_lst=[]
        gt2_lst=[]
        tcnv1_lst=[]
        tcnv2_lst=[]
        indctr1_lst=[]
        indctr2_lst=[]
        tintrvl_lst=[]
        
        
        #### The  
        #### Select an eye that converts within 6 months ###
        #print('within 6 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.cnvrt0_6)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        #### Select an eye that converts within 6 to 12 months ###
        #print('between 6-12 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.cnvrt6_12)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        ########### Select an eye which converts between 12-18 months #########
        #print('between 12-18 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.cnvrt12_18)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        ########### Select an eye which converts between 18-24 months #########
        #print('between 18-24 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.cnvrt18_24)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        ########### Select an eye which converts between 24-30 months #########
        #print('between 24-30 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.cnvrt24_30)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        ########### Select an eye which converts between 30-36 months #########
        #print('between 30-36 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.cnvrt30_36)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        #### Select an eye that is censored within 6 months ###
        #print('within months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.nocnv0_6)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        #### Select an eye that is censored between 6 to 12 months ###
        #print('between 6-12 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.nocnv6_12)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        #### Select an eye that is censored between 12 to 18 months ###
        #print('between 12-18 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.nocnv12_18)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        #### Select an eye that is censored between 18 to 24 months ###
        #print('between 18-24 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.nocnv18_24)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        #### Select an eye that is censored between 24 to 30 months ###
        #print('between 24-30 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.nocnv24_30)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        #### Select an eye that is censored between 30 to 36 months ###
        #print('between 30-36 months')
        img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl=self.random_sampling(self.nocnv30_36)
        
        img1_lst.append(img1)
        img2_lst.append(img2)
        gt1_lst.append(gt1)
        gt2_lst.append(gt2)
        tcnv1_lst.append(tcnv1)
        tcnv2_lst.append(tcnv2)
        indctr1_lst.append(indctr1)
        indctr2_lst.append(indctr2)
        tintrvl_lst.append(tintrvl)
        del img1,img2,gt1,gt2,tcnv1,tcnv2,indctr1,indctr2,tintrvl
        
        ############################################################################
        
        img1_lst=np.stack(img1_lst, axis=0) # B,1,H,W
        img2_lst=np.stack(img2_lst, axis=0)
        gt1_lst=np.stack(gt1_lst, axis=0)
        gt2_lst=np.stack(gt2_lst, axis=0)
        tcnv1_lst=np.stack(tcnv1_lst, axis=0)
        tcnv2_lst=np.stack(tcnv2_lst, axis=0)
        indctr1_lst=np.stack(indctr1_lst, axis=0)
        indctr2_lst=np.stack(indctr2_lst, axis=0)
        tintrvl_lst=np.stack(tintrvl_lst, axis=0)
        
        r=np.random.permutation(img1_lst.shape[0])
        img1_lst=img1_lst[r,0:1,:,:]
        img2_lst=img2_lst[r,0:1,:,:]
        gt1_lst=gt1_lst[r]
        gt2_lst=gt2_lst[r]
        tcnv1_lst=tcnv1_lst[r]
        tcnv2_lst=tcnv2_lst[r]
        indctr1_lst=indctr1_lst[r]
        indctr2_lst=indctr2_lst[r]
        tintrvl_lst=tintrvl_lst[r]
        del r
        
        img1_lst=torch.FloatTensor(img1_lst)
        img2_lst=torch.FloatTensor(img2_lst)
        gt1_lst=torch.FloatTensor(gt1_lst)
        gt2_lst=torch.FloatTensor(gt2_lst)
        tcnv1_lst=torch.FloatTensor(tcnv1_lst)
        tcnv2_lst=torch.FloatTensor(tcnv2_lst)
        indctr1_lst=torch.FloatTensor(indctr1_lst)
        indctr2_lst=torch.FloatTensor(indctr2_lst)
        tintrvl_lst=torch.FloatTensor(tintrvl_lst)
        
        
        tcnv1_lst=tcnv1_lst/(36*30)  # Normalize 0-3 years to the range of 0-1
        tcnv2_lst=tcnv2_lst/(36*30)
        tintrvl_lst=tintrvl_lst/(36*30)
        
        sample={'img1':img1_lst, 'img2':img2_lst, 'gt1':gt1_lst, 'gt2': gt2_lst,
                'tcnv1':tcnv1_lst, 'tcnv2':tcnv2_lst, 'indctr1':indctr1_lst, 'indctr2': indctr2_lst,
                'tintrvl': tintrvl_lst}
        return sample 

    def __len__(self):
        return self.epoch_size # smallest of all time-bin lists

