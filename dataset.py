#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2021-12-27 01:00
@Author  : Xiaoxiong
@File    : dataset.py
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import xarray as xr
import os,glob
import time
from tqdm import tqdm

class GribDataset(Dataset):
    def __init__(self,fdir,f_list,y_name=None,vals_idx=None,transform=None,forecast=False,validate=False):
        '''
            返回Dataset
            fdir:包含x和y的根目录
            f_list:x中的文件名列表
            y_name:y的前缀，'rain'或'temp'
            vals_idx:选取的变量所对应的下标,默认None为全选
            forecast:是否为预报模式，用于输出预报格点文件
        '''
        self.y_name = y_name
        self.transform = transform
        self.fdir=fdir
        self.f_list = f_list
        self.f_list.sort()
        self.vals_idx=vals_idx
        self.forecast=forecast
        self.validate=validate
        if self.validate:self.forecast=False

    def __len__(self):
        return len(self.f_list)

    def __getitem__(self, index):
        fp=self.f_list[index]
        x=np.load(os.path.join(self.fdir,'x',fp))
        if self.vals_idx is not None:x=x[self.vals_idx,:,:]
        x=torch.from_numpy(x).float()
        if self.transform is not None:
            x = self.transform(x)
        if self.y_name is None: return x
        if self.forecast:    
            return x,fp 
        elif self.validate:
            y=np.load(os.path.join(self.fdir,'y',self.y_name,fp))
            y=torch.from_numpy(y).float()
            return x,y,str.split(fp,'.')[0]
        else:
            y=np.load(os.path.join(self.fdir,'y',self.y_name,fp))
            y=torch.from_numpy(y).float()
            return x,y



def get_loaders(
    fdir,
    f_list,
    y_name=None,
    vals_idx=None,
    batch_size=32,
    transform=None,
    forecast=False,
    validate=False,
    num_works=4,
    pin_memory=True,
    shuffle=True,
):
    '''
        返回loader
        fdir:包含x和y的根目录
        f_list:x中的文件名列表
        y_name:y的前缀，'rain'或'temp'
        vals_idx:选取的变量所对应的下标,默认None为全选
    '''
    ds = GribDataset(
        fdir,
        f_list,
        y_name=y_name,
        vals_idx=vals_idx,
        transform=transform,
        forecast=forecast,
        validate=validate
    )

    loader =DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_works,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    return loader

#获取 selected 在 val_names (list类型) 中对应的下标
def get_idx(selected,val_names):
    idx=[]
    for v in selected:
        idx.append(val_names.index(v))
    return idx

def test(y_name):    
    start = time.time()
    DIR="../datas/processed/train/"
    flist=np.array(os.listdir(os.path.join(DIR,'x')))
    train_loader = get_loaders(
        DIR,flist,y_name=y_name,batch_size=32,transform=None,
        )
    xls=[]
    yls=[]
    for x, y in tqdm(train_loader):
        #xls.append(x.numpy())
        yls.append(y.numpy())
    #xls=np.vstack(xls)
    yls=np.vstack(yls)
    yls=yls.mean(axis=(0,1,2))
    yls[yls>-999]=1
    yls[yls<-999]=0
    np.save(os.path.join(DIR,f'{y_name}mask.npy'))
    #print(xls.shape,yls.shape)
    end = time.time()
    print ('read cost time:',str(end-start))

if __name__=="__main__":
    test()