import sys
from pandas.core import api
sys.path.insert(0, '../')
import numpy as np
from tqdm import tqdm
import pandas as pd
import os,pickle
from dataset import get_loaders,get_idx
from config import cfg
import xarray as xr
from analyze import draw_data,draw_RMS_MAE,draw_scatter,draw_max_min
from utils import ReadPointsData,ReadStaPoints_test

T_IDX=np.array(['01','02','03','04','05','06','07','08','09'])


def find_wrong_t(y_name='temp'):
    vals_idx=get_idx(['2 metre temperature'],cfg.VAL_NAMES)
    data_list=os.listdir(os.path.join(cfg.DATA_DIR,"x"))
    data_loader = get_loaders(
        cfg.DATA_DIR,data_list,y_name=y_name,vals_idx=vals_idx,batch_size=3*cfg.BATCH_SIZE,
        transform=None,num_works=cfg.NUM_WORKERS,pin_memory=cfg.PIN_MEMORY,shuffle=False,validate=True)
    RMSE=[]
    MAE=[]
    ls_data=[]
    ls_label=[]
    ls_fname=[]
    for data,label,fname in tqdm(data_loader): 
        data=data.numpy().squeeze()-273.15
        label=label.numpy().squeeze()
        label[label<=-9999]=np.nan
        mse = np.sqrt(np.nanmean(((data-label)**2), axis=(0,2, 3)))
        mae = np.nanmean((np.abs((data-label))), axis=(0,2, 3))
        RMSE.append(mse)
        MAE.append(mae)
        ls_data.append(data)
        ls_label.append(label)
        ls_fname+=fname
    ls_data=np.vstack(ls_data)
    ls_label=np.vstack(ls_label)
    idx_wrong=np.array(np.where(ls_label<-16))
    #print(idx_wrong)
    f_idx=list(set(idx_wrong[0]))
    f_idx.sort(key=list(idx_wrong[0]).index)
    t_idx=list(set(idx_wrong[1]))
    t_idx.sort(key=list(idx_wrong[1]).index)
    y_idx=list(set(idx_wrong[2]))
    y_idx.sort(key=list(idx_wrong[2]).index)
    x_idx=list(set(idx_wrong[3]))
    x_idx.sort(key=list(idx_wrong[3]).index)

    print('f_idx:',np.array(ls_fname)[f_idx])
    print('t_idx:',T_IDX[t_idx])
    print('y_idx:',y_idx)
    print('x_idx:',x_idx)


# list1 = [0, 3, 2, 3, 1, 0, 9, 8, 9, 7]
# list2 = list(set(list1))
# print(list2)        # [0, 1, 2, 3, 7, 8, 9]
# list2.sort(key = list1.index)
# print(list2) 


def find_wrong_r(y_name='rain'):
    vals_idx=get_idx(['Total precipitation'],cfg.VAL_NAMES)
    data_list=os.listdir(os.path.join(cfg.DATA_DIR,"x"))
    data_loader = get_loaders(
        cfg.DATA_DIR,data_list,y_name=y_name,vals_idx=vals_idx,batch_size=3*cfg.BATCH_SIZE,
        transform=None,num_works=cfg.NUM_WORKERS,pin_memory=cfg.PIN_MEMORY,shuffle=False,validate=True)
    RMSE=[]
    MAE=[]
    ls_data=[]
    ls_label=[]
    ls_fname=[]
    for data,label,fname in tqdm(data_loader): 
        data=data.numpy().squeeze()*1000
        label=label.numpy().squeeze()
        label[label<=-9999]=np.nan
        ls_data.append(data)
        ls_label.append(label)
        ls_fname+=fname
    ls_data=np.vstack(ls_data)
    ls_label=np.vstack(ls_label)
    # l_unique=ls_label.copy().reshape(-1,ls_label.shape[-2],ls_label.shape[-1])
    # l_unique=l_unique.reshape(-1,ls_label.shape[-2],ls_label.shape[-1])
    # l_unique=np.unique(l_unique,axis=0)
    idx_wrong=np.array(np.where(ls_label>150))
    print(idx_wrong)
    pp=[]
    for i in range(idx_wrong.shape[1]):
        pp.append([np.array(ls_fname)[idx_wrong[0,i]],T_IDX[idx_wrong[1,i]]])
    pp=np.unique(np.array(pp),axis=0)
    print(pp)
    for f,t in pp:
        ds=xr.open_dataset(f'/root/YXX/Racing/datas/train/{f}/obs_grid_rain{t}.nc')
        datas=ds.obs_rain.values
        datas[datas<0]=np.nan
        sname=f+'_'+t
        draw_data(datas,sname)



def evaluat_NWP(y_name='temp',
                data_list=os.listdir(os.path.join(cfg.DATA_DIR,"x")),
                sdir=os.path.join(cfg.SAVE_ROOT,'NWP_EVA'),
                vals_idx=get_idx(['2 metre temperature'],cfg.VAL_NAMES)
                ):
    if not os.path.exists(sdir):os.mkdir(sdir)
    points=ReadStaPoints_test(cfg.STATION_LIST)
    data_loader = get_loaders(
        cfg.DATA_DIR,data_list,y_name=y_name,vals_idx=vals_idx,batch_size=3*cfg.BATCH_SIZE,
        transform=None,num_works=cfg.NUM_WORKERS,pin_memory=cfg.PIN_MEMORY,shuffle=False,forecast=True,validate=True)
    ls_data=[]
    ls_label=[]
    ls_mse=[]
    ls_mae=[]
    ls_fname=[]
    for data,label,fnames in tqdm(data_loader): 
        if y_name=='temp':
            data=data.numpy().squeeze()-273.15
        label=label.numpy().squeeze()
        label[label<=-9999]=np.nan
        mse =  (data-label)**2
        mae =  np.abs((data-label))
        ls_mse.append(mse.squeeze())
        ls_mae.append(mae.squeeze())
        ls_data.append(data)
        ls_label.append(label)
        ls_fname+=fnames
    
    rmse=np.vstack(ls_mse)
    rmse=ReadPointsData(rmse,points)
    mae=np.vstack(ls_mae)
    mae=ReadPointsData(mae,points)
    preds=np.vstack(ls_data)
    preds=ReadPointsData(preds,points)
    obs=np.vstack(ls_label)
    obs=ReadPointsData(obs,points)
    f=open(os.path.join(cfg.MEAN_STD_DIR,"tidx.pkl"),"rb")
    tidx=pickle.load(f)
    f.close()
    f_00=list(tidx['00'])
    f_12=list(tidx['12'])
    name_00=list(set(ls_fname)&set(f_00))
    name_12=list(set(ls_fname)&set(f_12))
    idx_00=get_idx(name_00,ls_fname)
    idx_12=get_idx(name_12,ls_fname)
    obs_00=obs[idx_00]
    obs_12=obs[idx_12]
    preds_00=preds[idx_00]
    preds_12=preds[idx_12]      
    mae_00=mae[idx_00]
    mae_12=mae[idx_12]
    rmse_00=rmse[idx_00]
    rmse_12=rmse[idx_12]


    sta_mae=np.nanmean(mae,axis=(0,2)).reshape(-1,1)
    sta_rmse=np.sqrt(np.nanmean(rmse,axis=(0,2))).reshape(-1,1)
    df = pd.DataFrame(np.hstack((sta_mae,sta_rmse)),columns=['mae','rmse'])
    df.to_csv(os.path.join(cfg.MEAN_STD_DIR,f'sta_eva.csv'),index=None)

    sta_mae_00=np.nanmean(mae_00,axis=(0,2)).reshape(-1,1)
    sta_rmse_00=np.sqrt(np.nanmean(rmse_00,axis=(0,2))).reshape(-1,1)
    df = pd.DataFrame(np.hstack((sta_mae_00,sta_rmse_00)),columns=['mae','rmse'])
    df.to_csv(os.path.join(cfg.MEAN_STD_DIR,f'sta_eva_00.csv'),index=None)

    sta_mae_12=np.nanmean(mae_12,axis=(0,2)).reshape(-1,1)
    sta_rmse_12=np.sqrt(np.nanmean(rmse_12,axis=(0,2))).reshape(-1,1)
    df = pd.DataFrame(np.hstack((sta_mae_12,sta_rmse_12)),columns=['mae','rmse'])
    df.to_csv(os.path.join(cfg.MEAN_STD_DIR,f'sta_eva_12.csv'),index=None)
    
    draw_max_min(mae,points,'mae',sdir)
    draw_max_min(mae_00,points,'mae_00',sdir)
    draw_max_min(mae_12,points,'mae_12',sdir)
    draw_max_min(preds,points,'preds',sdir)
    draw_max_min(preds_00,points,'preds_00',sdir)
    draw_max_min(preds_12,points,'preds_12',sdir)
    draw_max_min(obs,points,'obs',sdir)
    draw_max_min(obs_00,points,'obs_00',sdir)
    draw_max_min(obs_12,points,'obs_12',sdir)

def evaluate_values(ls_data,ls_label,sdir,modelname='NWP'):
    points=ReadStaPoints_test(cfg.STATION_LIST)
    obs_min_grid=np.nanmin(ls_label,axis=(0,1))
    draw_data(obs_min_grid,'OBS_min',sdir)
    obs_max_grid=np.nanmax(ls_label,axis=(0,1))
    draw_data(obs_max_grid,'OBS_max',sdir)
    obs_mean_grid=np.nanmean(ls_label,axis=(0,1))
    draw_data(obs_mean_grid,'OBS_mean',sdir)
    nwp_max_grid=np.nanmax(ls_data,axis=(0,1))
    draw_data(nwp_max_grid,f'{modelname}_max',sdir)
    nwp_min_grid=np.nanmin(ls_data,axis=(0,1))
    draw_data(nwp_min_grid,f'{modelname}_min',sdir)
    nwp_mean_grid=np.nanmean(ls_data,axis=(0,1))
    draw_data(nwp_mean_grid,f'{modelname}_mean',sdir)
    print(f'{modelname}_max:',ls_data.max(),f'{modelname}_min:',ls_data.min(),f'{modelname}_mean:',ls_data.mean())
    print('OBS_max:',np.nanmax(ls_label),'OBS_min:',np.nanmin(ls_label),'OBS_mean:',np.nanmean(ls_label))
    nwp_max=np.nanmax(ls_data,axis=(0,2,3)).reshape(-1,1)
    nwp_min=np.nanmin(ls_data,axis=(0,2,3)).reshape(-1,1)
    nwp_mean=np.nanmean(ls_data,axis=(0,2,3)).reshape(-1,1)
    obs_max=np.nanmax(ls_label,axis=(0,2,3)).reshape(-1,1)
    obs_min=np.nanmin(ls_label,axis=(0,2,3)).reshape(-1,1)
    obs_mean=np.nanmean(ls_label,axis=(0,2,3)).reshape(-1,1)
    df_data=np.hstack((nwp_max,nwp_min,nwp_mean,obs_max,obs_min,obs_mean))
    #results={'rmse':np.array(RMSE),'mae':np.array(MAE)}
    df = pd.DataFrame(df_data,
                    columns=[f'{modelname}_max',f'{modelname}_min',f'{modelname}_mean','OBS_max','OBS_min','OBS_mean'])
    df.to_csv(os.path.join(sdir,f'{modelname}_eva.csv'),index=None)

    nwp_sta=ReadPointsData(ls_data,points)
    obs_sta=ReadPointsData(ls_label,points)
    #站点数据评估
    obs_min_sta=np.nanmin(obs_sta,axis=(0,1))
    draw_scatter(obs_min_sta,points,'sta_OBS_min',sdir)
    obs_max_sta=np.nanmax(obs_sta,axis=(0,1))
    draw_scatter(obs_max_sta,points,'sta_OBS_max',sdir)
    obs_mean_sta=np.nanmean(obs_sta,axis=(0,1))
    draw_scatter(obs_mean_sta,points,'sta_OBS_mean',sdir)
    nwp_max_sta=np.nanmax(nwp_sta,axis=(0,1))
    draw_scatter(nwp_max_sta,points,f'sta_{modelname}_max',sdir)
    nwp_min_sta=np.nanmin(nwp_sta,axis=(0,1))
    draw_scatter(nwp_min_sta,points,f'sta_{modelname}_min',sdir)
    nwp_mean_sta=np.nanmean(nwp_sta,axis=(0,1))
    draw_scatter(nwp_mean_sta,points,f'sta_{modelname}_mean',sdir)
    print(f'sta_{modelname}_max:',nwp_sta.max(),f'sta_{modelname}_min:',nwp_sta.min(),f'sta_{modelname}_mean:',nwp_sta.mean())
    print('sta_OBS_max:',np.nanmax(obs_sta),'sta_OBS_min:',np.nanmin(obs_sta),'sta_OBS_mean:',np.nanmean(obs_sta))
    sta_nwp_max=np.nanmax(nwp_sta,axis=(0,2)).reshape(-1,1)
    sta_nwp_min=np.nanmin(nwp_sta,axis=(0,2)).reshape(-1,1)
    sta_nwp_mean=np.nanmean(nwp_sta,axis=(0,2)).reshape(-1,1)
    sta_obs_max=np.nanmax(nwp_sta,axis=(0,2)).reshape(-1,1)
    sta_obs_min=np.nanmin(nwp_sta,axis=(0,2)).reshape(-1,1)
    sta_obs_mean=np.nanmean(nwp_sta,axis=(0,2)).reshape(-1,1)
    sta_df_data=np.hstack((sta_nwp_max,sta_nwp_min,sta_nwp_mean,sta_obs_max,sta_obs_min,sta_obs_mean))
    #results={'rmse':np.array(RMSE),'mae':np.array(MAE)}
    df = pd.DataFrame(sta_df_data,
                    columns=[f'{modelname}_max',f'{modelname}_min',f'{modelname}_mean','OBS_max','OBS_min','OBS_mean'])
    df.to_csv(os.path.join(sdir,f'{modelname}_sta_eva.csv'),index=None)


def evaluat_rmse_mae(ls_mse,ls_mae,sdir,modelname='NWP',d_rmse=False):
    '''
        ls_mse,ls_mae:B,S,L
        ls_data,ls_label:B,S,L
    '''
    max_mae=np.nanmax(ls_mae,axis=0)
    rmse_sta=np.nanmean(ls_mse,axis=0)
    mae_sta=np.nanmean(ls_mae,axis=0)
    #格点rmse，mae评估

    #站点rmse，mae评估
    points=ReadStaPoints_test(cfg.STATION_LIST)
    draw_RMS_MAE(np.sqrt(np.nanmean(rmse_sta,axis=0)),np.nanmean(mae_sta,axis=0),f'{modelname}_eva_sta',sdir)
    # for i in range(rmse_sta.shape[0]):
    #     if d_rmse:
    #         draw_scatter(np.sqrt(rmse_sta[i]),points,f'{modelname}_sta_rmse_{i+1}',sdir)
    #     draw_scatter(mae_sta[i],points,f'{modelname}_sta_mae_{i+1}',sdir)
    #     draw_scatter(max_mae[i],points,f'{modelname}_max_mae_{i+1}',sdir)
    draw_scatter(mae_sta,points,f'{modelname}_sta_all_mae',sdir)  
    draw_scatter(max_mae,points,f'{modelname}_max_mae_all',sdir)     
    df_data=np.hstack((np.sqrt(rmse_sta).reshape(-1,1),mae_sta.reshape(-1,1)))
    df = pd.DataFrame(df_data,columns=['sta_rmse','sta_OBS_mean'])
    df.to_csv(os.path.join(sdir,f'{modelname}_rmse_mae_eva.csv'),index=None)
    if d_rmse:
        draw_scatter(np.sqrt(np.nanmean(rmse_sta,axis=0)),points,f'{modelname}_sta_all_rmse',sdir)
        


if __name__=='__main__':
    evaluat_NWP()
    #find_wrong_r()