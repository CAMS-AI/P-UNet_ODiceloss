#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2021-12-25 23:00
@Author  : Xiaoxiong
@File    : utils.py
"""
import numpy as np
import sys
sys.path.append("..") 
import pandas as pd
import glob,os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import lr_scheduler
from model.loss import WeightedCrossEntropyLoss,DiceLoss
import pickle
from config import cfg
from model.UnetModel import T_net,Grid_Unet

def custom_crop(data,point,crop_range):
    '''
        根据点位剪裁网格数据
        data:输入数据...,H,W或B,S,C,H,W,或C,H,W,ndarray
        point:剪裁参照点,(y,x),为平面坐标位置，左下角原点
        crop_range:剪裁范围，X方向1/2，Y方向1/2长度，(y_range,x_range)
        剪裁将按照2*x_range+1,2*y_range+1范围剪裁
    '''
    sh=data.shape
    H,W=sh[-2:]
    y,x=point
    y_r,x_r=crop_range
    assert (x-x_r>=0) and (x+x_r<W)
    assert (y-y_r>=0) and (y+y_r<H)
    croped=data.copy()[...,x-x_r:x+x_r+1,y-y_r:y+y_r+1]
    return croped

#读取对应点数据
def ReadPointsData(data,points):
    return data[...,points[:,0],points[:,1]]

#读取站点坐标，站点坐标左下角为原点(0,0)
def ReadStaPoints(fname='../../datas/test/example00001/ji_loc_inputs_01.txt'):
    df=pd.read_csv(fname,names=['Y','X'],index_col=None)
    points=np.hstack((df.Y.values.reshape(-1,1),df.X.values.reshape(-1,1)))
    return points

def ReadStaPoints_test(fname=cfg.STATION_LIST):
    df=pd.read_csv(fname,names=['Y','X'],index_col=None,sep="\s+")
    points=np.hstack((df.Y.values.reshape(-1,1),df.X.values.reshape(-1,1)))
    return points

def getnamelist(fpath='../../datas/train/example00001/',str='obs_grid_temp*'):
    id_file=fpath+str
    l=len(glob.glob(id_file))
    print(l)
    filelist=glob.glob(id_file)
    return np.array(filelist)

def get_x_mean_std(vals_idx=None):
    '''
        返回mean_std和transform
    '''
    with open(os.path.join(cfg.MEAN_STD_DIR,'x.pkl'),'rb') as fx:
        xdicts=pickle.load(fx)     
    x_mean_std=np.hstack((xdicts['mean'].reshape(-1,1),xdicts['std'].reshape(-1,1)))
    if vals_idx is not None:
        x_mean_std=x_mean_std[vals_idx,:]        
    return x_mean_std

def get_mean_std(y_name='temp',vals_idx=None):
    '''
        返回mean_std和transform
    '''
    with open(os.path.join(cfg.MEAN_STD_DIR,y_name+'.pkl'),'rb') as fy:
        ydicts=pickle.load(fy)
    y_mean_std=np.array([ydicts['mean'],ydicts['std']])
    y_mean_std=torch.from_numpy(y_mean_std).to(cfg.DEVICE)
    with open(os.path.join(cfg.MEAN_STD_DIR,'x.pkl'),'rb') as fx:
        xdicts=pickle.load(fx)     
    x_mean_std=np.hstack((xdicts['mean'].reshape(-1,1),xdicts['std'].reshape(-1,1)))
    if vals_idx is not None:
        x_mean_std=x_mean_std[vals_idx,:]        
    transform =transforms.Normalize(
                mean=x_mean_std[:,0].tolist(),
                std=x_mean_std[:,1].tolist())
    return y_mean_std,transform

def create_net(train=True,model_path=None,device=cfg.DEVICE):
    '''
        初始化网络
    '''  
    model = Grid_Unet(in_channels=cfg.MODEL.IN_CHANNEL,out_channels=cfg.MODEL.OUT_CHANNEL)
    if model_path is not None:model.load_state_dict(torch.load(model_path))
    if train:model=model.apply(weights_init)
    # if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)
    model=model.to(device) 
    if train:
        LR_step_size = 10
        LR = 1e-3
        thresholds=np.array(cfg.RAIN.THRESHOLDS)
        weights = np.ones_like(thresholds)
        balancing_weights = cfg.BALANCING_WEIGHTS
        # for i, threshold in enumerate(cfg.RAIN.THRESHOLDS):
        #     weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (thresholds >= threshold)
        # weights = weights + 1
        # weights = np.array([1] + weights.tolist())
        # weights = torch.from_numpy(weights).to(device).float()
        criterion=DiceLoss(cfg.MODEL.OUT_CHANNEL,weight=balancing_weights)
        # criterion = WeightedCrossEntropyLoss(thresholds, weights).to(device)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=LR,weight_decay=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=0.1)
        #mult_step_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30000, 60000], gamma=0.1)
        return model, optimizer, criterion, exp_lr_scheduler
    return model

def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            #nn.init.orthogonal_(m.weight)#正交初始化，适用于RNN
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.kaiming_noem_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        # if isinstance(m, nn.BatchNorm2d):
        #     nn.init.constant(m.weight, 1)
        #     nn.init.constant(m.bias, 0)
    #return model

#if __name__=="__main__":
    # data=np.random.randn(20,12,4,45,45)
    # d1=custom_crop(data,(10,15),(5,5))
    # print(d1.shape)
    # flist=getnamelist()
    # print(flist)
    #predict_K_fold()

    


