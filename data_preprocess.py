#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2021-12-27 01:00
@Author  : Xiaoxiong
@File    : data_preprocess.py
"""
import sys 
sys.path.append("..") 
import xarray as xr
import pandas as pd
import numpy as np
import os,glob,pickle
import time
from tqdm import tqdm
from dataset import get_loaders
from config import cfg

def read_nc(x_name=None,y_name=None):
    if x_name is not None:
        xds=xr.open_dataset(x_name)
        vars=[xds[name].values for name in xds]
        ls1=np.array(vars[0:9]).transpose(0,3,1,2) #前9个变量有5个层次高度，后面都是单层
        ls1[0,:,:,:]=np.clip(ls1[0,:,:,:],0,100)
        ls1=ls1.reshape(-1,ls1.shape[-2],ls1.shape[-1])
        ls2=np.array(vars[9:])
        x=np.vstack((ls1,ls2))
        xds=xds.close()
    if y_name is None: return x
    yds=xr.open_dataset(y_name) 
    y=np.array([yds[name].values for name in yds]).squeeze() #仅有一个变量
    yds=yds.close()
    if x_name is None: return y      
    return x,y

def read_sample(fp,y_name=None,x_name=None):
    '''
        读取单个个例，及文件夹下对应文件
    '''
    xls=[] 
    yls=[]
    if x_name is not None:
        x_flist=np.array(glob.glob(os.path.join(fp,x_name)))
        x_flist.sort()
        flist=x_flist  
    if y_name is not None:
        y_flist=np.array(glob.glob(os.path.join(fp,f'*obs_grid_{y_name}*')))
        y_flist.sort()
        flist=y_flist      
    for i in range(flist.shape[0]):
        if y_name is not None and x_name is not None:
            x,y=read_nc(x_flist[i],y_flist[i])
            yls.append(y)
            xls.append(x)
        elif y_name is None: 
            x=read_nc(x_flist[i],None)
            xls.append(x) 
        elif x_name is None:
            y=read_nc(None,y_flist[i])
            yls.append(y)   
    xls=np.array(xls)
    yls=np.array(yls)
    if y_name is None: 
        return xls
    elif x_name is None:
        yls=yls[:,np.newaxis,:,:]
        return yls
    else:
        yls=yls[:,np.newaxis,:,:]
        return xls,yls


def NC_to_NPY_train(dir=os.path.join(cfg.ROOT_DATA,'train'),s_dir=cfg.DATA_DIR,x_name='*grid_inputs*'):
    '''
        将NC数据处理成npy格式数据
        dir:NC数据存储根目录，所有个例均以example*文件名存储
        s_dir:数据输出根目录，x
        x_name:模式数据名检索字段，默认设置'*grid_inputs*'
    '''
    print('Converting Datas...')
    flist=glob.glob(os.path.join(dir,'*example*'))   
    flist=np.array(flist)
    flist.sort()
    if not os.path.exists(s_dir):os.mkdir(s_dir)
    if not os.path.exists(os.path.join(s_dir,'x')):os.mkdir(os.path.join(s_dir,'x'))
    if not os.path.exists(os.path.join(s_dir,'y')):os.mkdir(os.path.join(s_dir,'y'))
    if not os.path.exists(os.path.join(s_dir,'y','rain')):os.mkdir(os.path.join(s_dir,'y','rain'))
    if not os.path.exists(os.path.join(s_dir,'y','temp')):os.mkdir(os.path.join(s_dir,'y','temp'))
    for fp in tqdm(flist): 
        fname=fp[-12:]
        x_flist=np.array(glob.glob(os.path.join(fp,x_name)))
        x_flist.sort()
        flist=x_flist     
        for fn in flist:
            tidx=fn[-5:-3]
            if not os.path.exists(os.path.join(fp,f'obs_grid_rain{tidx}.nc')) or not os.path.exists(os.path.join(fp,f'obs_grid_temp{tidx}.nc')): continue
            if os.path.exists(os.path.join(s_dir,'x',fname+'_'+tidx+'.npy')) and os.path.exists(os.path.join(s_dir,'y','rain',fname+'_'+tidx+'.npy')) \
            and os.path.exists(os.path.join(s_dir,'y','temp',fname+'_'+tidx+'.npy')) :continue
            x=read_nc(fn)
            y_rain=read_nc(None,os.path.join(fp,f'obs_grid_rain{tidx}.nc'))
            y_temp=read_nc(None,os.path.join(fp,f'obs_grid_temp{tidx}.nc'))
            x[0:5,:,:]=np.clip(x[0:5,:,:],0,100) #将相对湿度的值限制在0-100
            x=x[...,2:69-(69-64-2),(73-64)//2:73-(73-64-(73-64)//2)]
            y_temp=y_temp[...,2:69-(69-64-2),(73-64)//2:73-(73-64-(73-64)//2)]
            y_rain=y_rain[...,2:69-(69-64-2),(73-64)//2:73-(73-64-(73-64)//2)]
            np.save(os.path.join(s_dir,'x',fname+'_'+tidx+'.npy'),x)  
            np.save(os.path.join(s_dir,'y','rain',fname+'_'+tidx+'.npy'),y_rain)       
            np.save(os.path.join(s_dir,'y','temp',fname+'_'+tidx+'.npy'),y_temp)    
    print('Convert data finished!')


def NC_to_NPY_test(dir=cfg.TEST_STATION_DIR,s_dir=cfg.DATA_DIR,x_name='*grid_inputs*'):
    '''
        将NC数据处理成npy格式数据，此处仅处理test数据中预报场
        dir:NC数据存储根目录，所有个例均以example*文件名存储
        s_dir:数据输出根目录，x
        x_name:模式数据名检索字段，默认设置'*grid_inputs*'
    '''
    print('Converting Datas...')
    flist=glob.glob(os.path.join(dir,'*example*'))   
    flist=np.array(flist)
    flist.sort()
    if not os.path.exists(s_dir):os.mkdir(s_dir)
    if not os.path.exists(os.path.join(s_dir,'x')):os.mkdir(os.path.join(s_dir,'x'))
    for fp in tqdm(flist):  
        #assert x_flist.shape[0]==9
        fname=fp[-12:]
        x_flist=np.array(glob.glob(os.path.join(fp,x_name)))
        x_flist.sort()  
        for fn in x_flist:
            tidx=fn[-5:-3]
            if os.path.exists(os.path.join(s_dir,'x',fname+'_'+tidx+'.npy')):continue
            x=read_nc(fn)
            x[0:5,:,:]=np.clip(x[0:5,:,:],0,100) #将相对湿度的值限制在0-100
            x=x[...,2:69-(69-64-2),(73-64)//2:73-(73-64-(73-64)//2)]#对数据进行64*64剪裁
            np.save(os.path.join(s_dir,'x',fname+'_'+tidx+'.npy'),x)            
    print('Convert data finished!')

def modify_abnormal_temp(y):
    y1=y.copy()
    y1[y1<-9999]=np.nan
    idxs=np.where(y1<-16)
    idxs=np.array(idxs)
    if idxs.size>0 : 
        idxs_t=np.unique(idxs[0])
        for i in idxs_t:
            p_idx=np.where(y1[i]<-16)
            y1[i][y1[i]<-16]=np.nan
            m=np.nanmean(y1[i])
            y[i][p_idx]=m
    return y

def modify_abnormal_rain(y):
    y1=y.copy()
    y1[y1<-9999]=np.nan
    idxs=np.where(y1>300)
    if np.array(idxs).size>0:
        y[idxs]=0
        print('ab checked')
    return y



def get_norm_param(fdir,y_name):
    '''
        用于获取数据中各变量标准化参数，最大，最小，平均值，标准差
        以dictionary输出
    '''
    flist=np.array(os.listdir(os.path.join(fdir,'x')))
    train_loader = get_loaders(
        fdir,flist,y_name=y_name,batch_size=32,transform=None,
        )
    xls=[]
    yls=[]
    for x, y in tqdm(train_loader):
        xls.append(x.numpy())
        if y_name is not None:
            yls.append(y.numpy())
    xls=np.vstack(xls)
    print(xls.shape)
    dicx={}  
    dicx['max']=xls.max(axis=(0,2,3))
    dicx['min']=xls.min(axis=(0,2,3))
    dicx['mean']=xls.mean(axis=(0,2,3))
    dicx['std']=xls.std(axis=(0,2,3))
    pkl_x=os.path.join(fdir,"m","x.pkl")
    with open(pkl_x, 'wb') as f:
        pickle.dump(dicx, f, pickle.HIGHEST_PROTOCOL) #保存文件
        print("保存：",pkl_x)
    if y_name is not None:
        yls=np.vstack(yls)
        print(yls.shape)
        yls[yls<-9999]=np.nan
        dicy={}
        dicy['max']=np.nanmax(yls)
        dicy['min']=np.nanmin(yls)
        dicy['mean']=np.nanmean(yls)
        dicy['std']=np.nanstd(yls)  
        pkl_y=os.path.join(fdir,"m",f"{y_name}.pkl")
        with open(pkl_y, 'wb') as f:
            pickle.dump(dicy, f, pickle.HIGHEST_PROTOCOL) #保存文件
            print("保存：",pkl_y)   
    print('get norm parameters finished!')



def get_all_station(fdir,fname='*ji_loc_inputs*'): #获取所有站点，并保存
    fdlist=glob.glob(os.path.join(fdir,'example*'))
    plist=[]
    for fd in fdlist:
        flist=glob.glob(os.path.join(fd,fname))
        for fp in flist:
            df=pd.read_csv(fp,names=['Y','X'],index_col=None,sep="\s+")
            points=list(np.hstack((df.Y.values.reshape(-1,1),df.X.values.reshape(-1,1))))
            plist=list((set(tuple(i) for i in plist)).union(set((tuple(j) for j in points))))   
    filename = open(os.path.join(cfg.MEAN_STD_DIR,'stationlist.txt'),'w')
    for p in plist:
        filename.write(str(p[0])+','+str(p[1]))
        filename.write('\n')
    filename.close()
    print('站点列表保存完毕!')

def station_offset(fdir,offset_x=4,offset_y=2,fname='*ji_loc_inputs*'): #获取所有站点，并保存
    df=pd.read_csv(os.path.join(cfg.MEAN_STD_DIR,'stationlist_64.txt'))
    fdlist=glob.glob(os.path.join(fdir,'example*'))
    plist=[]
    for fd in fdlist:
        flist=glob.glob(os.path.join(fd,fname))
        for fp in flist:
            df=pd.read_csv(fp,names=['Y','X'],index_col=None,sep="\s+")
            points=list(np.hstack((df.Y.values.reshape(-1,1),df.X.values.reshape(-1,1))))
            plist=list((set(tuple(i) for i in plist)).union(set((tuple(j) for j in points))))   
    filename = open(os.path.join(cfg.MEAN_STD_DIR,'stationlist.txt'),'w')
    for p in plist:
        filename.write(str(p[0]-offset_y)+','+str(p[1]-offset_x))
        filename.write('\n')
    filename.close()
    print('站点列表保存完毕!')


def get_total_mask(fdir,y_name):   #获取总的mask
    flist=np.array(os.listdir(os.path.join(fdir,'x')))
    train_loader = get_loaders(
        fdir,flist,y_name=y_name,batch_size=32,transform=None,
        )
    yls=[]
    for _, y in tqdm(train_loader):
        #xls.append(x.numpy())
        yls.append(y.numpy())
    #xls=np.vstack(xls)
    yls=np.vstack(yls)
    yls[yls>-9999]=1
    yls[yls<=-9999]=0
    yls=yls.sum(axis=(0,1,2))
    np.save(os.path.join(fdir,f'{y_name}_mask.npy'),yls)
    print('mask created!')

def clip_data(fdir): #用于将相对湿度的值限制在0-100
    flist=os.listdir(os.path.join(fdir,'x'))
    for fp in flist:
        x=np.load(os.path.join(fdir,'x',fp))
        x[:,0:5,:,:]=np.clip(x[:,0:5,:,:],0,100)
        np.save(os.path.join(fdir,'x',fp),x)
    print('clip finished!')

#裁剪成64*64,注意对应的站点坐标也要调整！！
def crop_data(fdir,y=True): 
    flist=os.listdir(os.path.join(fdir,'x'))
    for fp in flist:
        x=np.load(os.path.join(fdir,'x',fp))
        x=x[...,2:69-(69-64-2),(73-64)//2:73-(73-64-(73-64)//2)]
        np.save(os.path.join(fdir,'x',fp),x)
        if y:
            rain=np.load(os.path.join(fdir,'y','rain_'+fp))
            rain=rain[...,2:69-(69-64-2),(73-64)//2:73-(73-64-(73-64)//2)]
            np.save(os.path.join(fdir,'y','rain_'+fp),rain)

            temp=np.load(os.path.join(fdir,'y','temp_'+fp))
            temp=temp[...,2:69-(69-64-2),(73-64)//2:73-(73-64-(73-64)//2)]
            np.save(os.path.join(fdir,'y','temp_'+fp),temp)
        print(fp)
    print('crop finished!')

def split_00_12(fdir,sdir,y_name=cfg.Y_NAME,x_name='*grid_inputs*'):
    flist=os.listdir(fdir)
    l_00=[]
    l_12=[]
    for fp in flist:
        try:
            if x_name is not None:
                x_flist=np.array(glob.glob(os.path.join(fdir,fp,x_name)))
                assert x_flist.shape[0]==9
            if y_name is not None:
                y_flist=np.array(glob.glob(os.path.join(fdir,fp,f'*obs_grid_{y_name}*')))
                assert y_flist.shape[0]==9   
        except:
            print(fp+'个例文件数目异常！')
            continue 
        fname=glob.glob(os.path.join(fdir,fp,'*_12-36h'))[0]
        tidx=fname[-9:][0:2]
        if tidx=='00':l_00.append(fp)
        if tidx=='12':l_12.append(fp)
    str = '\n'
    f=open(os.path.join(sdir,"00.txt"),"w")
    f.write(str.join(l_00))
    f.close()
    str = '\n'
    f=open(os.path.join(sdir,"12.txt"),"w")
    f.write(str.join(l_12))
    f.close()
    f=open(os.path.join(sdir,"tidx.pkl"),"wb")
    pickle.dump({'00':np.array(l_00),'12':np.array(l_12)},f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print('finished!')

def split_to_times(fdir=cfg.ROOT_DATA,sdir=cfg.TEST_DIR,y_name=cfg.Y_NAME):
    if not os.path.exists(os.path.join(sdir,'x')):os.mkdir(os.path.join(sdir,'x'))
    # if not os.path.exists(os.path.join(sdir,'y')):os.mkdir(os.path.join(sdir,'y'))
    # if not os.path.exists(os.path.join(sdir,'y','rain')):os.mkdir(os.path.join(sdir,'y','rain'))
    # if not os.path.exists(os.path.join(sdir,'y','temp')):os.mkdir(os.path.join(sdir,'y','temp'))
    xlist=os.listdir(os.path.join(fdir,'processed','test','x'))
    for f in xlist:
        sname=str.split(f,'.')[0]
        x=np.load(os.path.join(fdir,'processed','test','x',f))
        # y_rain=np.load(os.path.join(fdir,'processed','train','y','rain_'+f))
        # y_temp=np.load(os.path.join(fdir,'processed','train','y','temp_'+f))
        for i in range(x.shape[0]):
            np.save(os.path.join(sdir,'x',sname+'_'+str(i+1).zfill(2)),x[i])
            # np.save(os.path.join(sdir,'y','rain',sname+'_'+str(i+1).zfill(2)),y_rain[i])
            # np.save(os.path.join(sdir,'y','temp',sname+'_'+str(i+1).zfill(2)),y_temp[i])
        print(f)
    print('finished!')



def replace_nan(fname):
    x=np.load(fname)
    x[np.isnan(x)]=-99999.
    np.save(fname,x)





        
if __name__=='__main__':
    fdir='/root/YXX/Racing/datas/processed/train/y'
    
    
    # NC_to_NPY(dir=os.path.join(cfg.ROOT_DATA,'train'),s_dir=cfg.DATA_DIR,y_name='temp')
    # get_norm_param(fdir,y_name='temp')
    NC_to_NPY_train(dir=os.path.join(cfg.ROOT_DATA,'train'),s_dir=cfg.DATA_DIR)
