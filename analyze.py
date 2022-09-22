import sys
sys.path.append("..") 
import pickle
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os
from matplotlib import cm
import matplotlib.colors as mcolors
from config import cfg
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#from predict import save_predictions

def draw_ts(data,sname,sdir):    
    def autolable(rects):
        for rect in rects:
            height = rect.get_height()
            if height>=0:
                plt.text(rect.get_x()+rect.get_width()/2.0 - 0.3,height+0.02,'%.3f'%height)
            else:
                plt.text(rect.get_x()+rect.get_width()/2.0 - 0.3,height-0.06,'%.3f'%height)
                # 如果存在小于0的数值，则画0刻度横向直线
                plt.axhline(y=0,color='black')

    norm = plt.Normalize(0,1)
    norm_values = norm(data)
    map_vir = cm.get_cmap(name='inferno')
    colors = map_vir(norm_values)
    key_name=['ts_0.1','ts_3','ts_10','ts_20']
    plt.figure(figsize=(6,5)) #调用figure创建一个绘图对象
    plt.subplot(111)
    ax = plt.bar(key_name,data,width=0.5,color=colors,edgecolor='black') # edgecolor边框颜色
    sm = cm.ScalarMappable(cmap=map_vir,norm=norm)  # norm设置最大最小值
    sm.set_array([])
    plt.title(sname,size=16) 
    plt.colorbar(sm)
    autolable(ax)
    plt.savefig(os.path.join(sdir,'ts_'+f'{sname}+.png'), dpi=300,bbox_inches = 'tight')  
    plt.close()

def draw_max_min(data,points,sname,sdir):
    draw_scatter(np.nanmean(data,axis=(0,1)),points,f'mean_all_{sname}',sdir)
    draw_scatter(np.nanmax(data,axis=(0,1)),points,f'max_all_{sname}',sdir)
    draw_scatter(np.nanmin(data,axis=(0,1)),points,f'min_all_{sname}',sdir)
    data_mean=np.nanmean(data,axis=0)
    data_max=np.nanmax(data,axis=0)
    data_min=np.nanmin(data,axis=0)
    for i in range(data.shape[1]):
        draw_scatter(data_mean[i],points,f'mean_{sname}_{i+1}',sdir)
        draw_scatter(data_max[i],points,f'max_{sname}_{i+1}',sdir)
        draw_scatter(data_min[i],points,f'min_{sname}_{i+1}',sdir)

def draw_scatter(data,points,sname,sdir):
    plt.figure(figsize=(12,10))
    plt.scatter(points[:,1],points[:,0],marker='o',c=data,cmap='rainbow')
    plt.colorbar()
    plt.title(f'max:{np.nanmax(data):.3f},min:{np.nanmin(data):.3f},mean:{np.nanmean(data):.3f}',size=12,loc='left')
    plt.title(sname,size=14,loc='right')
    plt.savefig(os.path.join(sdir,f'{sname}.png'), dpi=300,bbox_inches = 'tight')
    plt.close()
    print('save:',os.path.join(sdir,f'{sname}.png'))

def draw_RMS_MAE(RMSE,MAE,sname,sdir):
    plt.figure(figsize=(12,10))
    plt.bar( np.arange(0,RMSE.size)-0.2,RMSE,color="g", width=0.4,label="RMSE") 
    plt.bar( np.arange(0,MAE.size)+0.2,MAE,color="b", width=0.4,label="MAE") 
    plt.xticks(np.arange(RMSE.size),((np.arange(RMSE.size))*3+12))
    plt.xlabel("Times")
    plt.ylabel("MAE & RMSE") 
    #plt.title(sname,size=16)
    plt.title(f'max_rmse:{np.nanmax(RMSE):.3f}, min_rmse:{np.nanmin(RMSE):.3f}, rmse_mean:{np.nanmean(RMSE):.3f}',size=12,loc='left')
    plt.title(f'max_mae:{np.nanmax(MAE):.3f}, min_mae:{np.nanmin(MAE):.3f}, mae_mean:{np.nanmean(MAE):.3f}',size=12,loc='right')
    plt.legend() 
    plt.savefig(os.path.join(sdir,sname+'.png'), dpi=300,bbox_inches = 'tight')
    plt.close()
    print('save:',os.path.join(sdir,f'{sname}.png'))

def draw_data(datas,sname,s_dir=cfg.SAVE_ROOT):
    lon=np.arange(107.375, 115.25+0.125, 0.125)
    lat=np.arange(31.5, 23.625-0.125, -0.125)
    proj = ccrs.PlateCarree() #创建投影

    fig=plt.figure(figsize=(12,10))
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
    reader = shpreader.Reader('/root/YXX/NP_UNET/datas/m/gadm36_CHN_1.shp')
    ax.add_geometries(reader.geometries(),ccrs.PlateCarree(),edgecolor = 'k',facecolor = 'none')
    extent = [lon.min(),lon.max(),lat.min(),lat.max()]
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    # linewidth=1.2, color='k', alpha=0.5, linestyle='--')
    # gl.xlabels_top = False #关闭顶端标签
    # gl.ylabels_right = False #关闭右侧标签
    # gl.xformatter = LONGITUDE_FORMATTER #x轴设为经度格式
    # gl.yformatter = LATITUDE_FORMATTER #y轴设为纬度格式
    ax.set_xticks(np.arange(107.375, 115.25+0.125, 0.75), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(31.5, 23.625-0.125, -0.75), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    # cbar_kwargs = {
    #         'orientation': 'horizontal',
    #         'label': 'Potential',
    #         'shrink': 0.8,
    #             }
    cf=ax.contourf(lon,lat,datas,levels=20,cmap='jet', transform=ccrs.PlateCarree())
    cbar=plt.colorbar(cf)
    ax.set_title(sname,size=16,loc='right')
    plt.title(f'max:{np.nanmax(datas):.3f},min:{np.nanmin(datas):.3f},mean:{np.nanmean(datas):.3f}',size=12,loc='left')
    plt.title(sname,size=16,loc='right')
    plt.savefig(os.path.join(s_dir,f'{sname}.png'), dpi=300,bbox_inches = 'tight')
    plt.close()
    print('save:',os.path.join(s_dir,f'{sname}.png'))

def data_analysis():
    r_mask=np.load('../../datas/processed/train/rain_mask.npy')
    lx=np.arange(0,73)
    ly=np.arange(0,69)
    gx,gy=np.meshgrid(lx,ly)

    res_freq = stats.relfreq(r_mask[r_mask>0].flatten(), numbins=20) #获取频率
    pdf_value = res_freq.frequency
    cdf_value = np.cumsum(res_freq.frequency)
    x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
    plt.bar(x, pdf_value, width=res_freq.binsize)
    plt.show()

    staf='../../datas/processed/train/m/stationlist.txt'
    df=pd.read_csv(staf,names=['Y','X'],index_col=None)
    points=np.hstack((df.Y.values.reshape(-1,1),df.X.values.reshape(-1,1)))

    plt.contourf(gx,gy,r_mask)
    plt.scatter(points[:,1],points[:,0])
    plt.colorbar()
    plt.show()

    pc=r_mask[points[:,0],points[:,1]].reshape(-1,1)
    pdata=np.hstack((points,pc))
    plt.scatter(pdata[:,1],pdata[:,0],c=pdata[:,2],norm=colors.LogNorm(vmin=pdata[:,2].min(), vmax=pdata[:,2].max()))
    plt.colorbar()
    plt.show()

def scalar_analysis(fname,sdir,sname,NWP_path):
    with open(fname,'rb') as f:
        Dic=pickle.load(f)
    train_loss=np.array(Dic['train_loss'])
    val_loss=np.array(Dic['valid_loss'])
    rmse=np.array(Dic['rmse']) #100,9
    mae=np.array(Dic['mae'])
    draw_loss_line(rmse,mae,train_loss,val_loss,sdir,sname)
    bestepoch=Dic['best_epoch']
    brmse=rmse[bestepoch]
    bmae=mae[bestepoch]
    draw_rmse_bar(brmse,bmae,sdir,sname,NWP_path)
    #return brmse,bmae,rmse,mae,Dic

def draw_rmse_bar(brmse,bmae,sdir,sname,NWP_path):
    df=pd.read_csv(NWP_path)
    N_rmse=df['rmse'].values
    N_mae=df['mae'].values
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(121)   #nrows,ncolumns,index
    ax1.bar( np.arange(0,bmae.size)-0.2,bmae,color="g", width=0.4,label="TrajGRU") 
    ax1.bar( np.arange(0,N_mae.size)+0.2,N_mae,color="b", width=0.4,label="NWP") 
    plt.xticks(np.arange(bmae.size),((np.arange(bmae.size))*3+12))
    ax1.set_xlabel("Times",size=14)
    ax1.set_ylabel("MAE",size=14) 
    plt.legend(loc='lower right',fontsize='x-large')
    plt.title('Traj_mae:{:.3f}'.format(bmae.mean()),loc='left',size=14)
    plt.title('NWP_mae:{:.3f}'.format(N_mae.mean()),loc='right',size=12)
    #plt.title('NWP and TrajGRU MAE',size=16)

    ax2 = fig.add_subplot(122)
    ax2.bar( np.arange(0,brmse.size)-0.2,brmse,color="g", width=0.4,label="TrajGRU") 
    ax2.bar( np.arange(0,N_rmse.size)+0.2,N_rmse,color="b", width=0.4,label="NWP") 
    plt.xticks(np.arange(brmse.size),((np.arange(brmse.size))*3+12))
    ax2.set_xlabel("Times",size=14)
    ax2.set_ylabel("RMSE",size=14) 
    plt.legend(loc='lower right',fontsize='x-large',facecolor='white')
    plt.title('Traj_rmse:{:.3f}'.format(brmse.mean()),loc='left',size=14)
    plt.title('NWP_rmse:{:.3f}'.format(N_rmse.mean()),loc='right',size=14)
    #plt.title('NWP and TrajGRU RMSE',size=16)
    plt.savefig(os.path.join(sdir,sname+'_bar.png'), dpi=300,bbox_inches = 'tight')
    plt.close()
    #plt.show()

def draw_loss_line(ts,train_loss,val_loss,sdir,sname):
    #colors=list(mcolors.TABLEAU_COLORS.keys()) #color=mcolors.TABLEAU_COLORS[colors[i]],label=str(step*(i+1)*tau)[:3]
    fig = plt.figure(figsize=(15,6))

    ax1 = fig.add_subplot(121)
    ax1.plot(np.arange(1,ts.shape[0]+1),ts[:,0],  "r-",label="ts_0.1")
    ax1.plot(np.arange(1,ts.shape[0]+1),ts[:,1],  "b-",label="ts_3")
    ax1.plot(np.arange(1,ts.shape[0]+1),ts[:,2],  "g-",label="ts_10")
    ax1.plot(np.arange(1,ts.shape[0]+1),ts[:,3],  "y-",label="ts_20")
    plt.legend(loc='center right',fontsize='large')
    ax1.set_xlim(1, train_loss.size)
    ax1.set_ylabel("TS",size=14)
    ax1.set_xlabel("epoch_number",size=14)
    plt.title("TS",size=16) 


    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(1,train_loss.size+1),train_loss,  "r-",label="train_loss")
    ax2.plot(np.arange(1,val_loss.size+1),val_loss,  "b-",label="valid_loss")
    plt.legend(loc='best',fontsize='large')
    ax2.set_xlim(1, train_loss.size)
    ax2.set_ylabel("Loss",size=14)
    ax2.set_xlabel("epoch_number",size=14)
    plt.title("LOSS",size=16) 
    plt.savefig(os.path.join(sdir,sname+'_Loss.png'), dpi=300,bbox_inches = 'tight')
    plt.close()
    #plt.show()

def draw_pod_far(TP,FP,TS,sdir,sname):
    #colors=list(mcolors.TABLEAU_COLORS.keys()) #color=mcolors.TABLEAU_COLORS[colors[i]],label=str(step*(i+1)*tau)[:3]
    fig = plt.figure(figsize=(18,12))
    TP=TP[:20]
    FP=FP[:20]
    TS=TS[:20]
    ax1 = fig.add_subplot(221)
    cf1=ax1.plot(np.arange(1,TP.shape[0]+1),TP[:,0],  "r-",label="TP")
    cf11=ax1.plot(np.arange(1,FP.shape[0]+1),FP[:,0],  "b-",label="FP")
    # ax1.plot(np.arange(1,pod.shape[0]+1),pod[:,2],  "g-",label="ts_10")
    # ax1.plot(np.arange(1,pod.shape[0]+1),pod[:,3],  "y-",label="ts_20")
    
    ax1.set_xlim(1, TP.shape[0])
    ax1.set_ylabel("TP",size=14)
    ax1.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    ax1.set_xticks(np.arange(1, TP.shape[0]+1, 2))
    ax1.set_xlabel("epoch_number",size=14)
    # plt.legend(loc='best',fontsize='large')
    ax12 = ax1.twinx()
    cf12=ax12.plot(np.arange(1,FP.shape[0]+1),TS[:,0],  "k-.",label="TS")
    # ax12.set_xticks(np.arange(0, TP.shape[1]+1, 2))
    ax12.set_ylabel("TS",size=14)
    # ax12.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    plt.legend(cf1+cf11+cf12,['TP','FP','TS'],loc='best',fontsize='large')
    
    # plt.tick_params(axis='both',which='major',labelsize=14)
    # fig.legend(loc="upper right",axis=ax1)
    # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title("Rainfall>=0.1mm",size=16)

    ax2 = fig.add_subplot(222)
    cf2=ax2.plot(np.arange(1,TP.shape[0]+1),TP[:,1],  "r-",label="TP")
    cf21=ax2.plot(np.arange(1,FP.shape[0]+1),FP[:,1],  "b-",label="FP")
    # ax1.plot(np.arange(1,pod.shape[0]+1),pod[:,2],  "g-",label="ts_10")
    # ax1.plot(np.arange(1,pod.shape[0]+1),pod[:,3],  "y-",label="ts_20")
    # plt.legend(loc='best',fontsize='large')
    ax2.set_xlim(1, TP.shape[0])
    ax2.set_ylabel("TP",size=14)
    ax2.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    ax2.set_xticks(np.arange(1, TP.shape[0]+1, 2))
    ax2.set_xlabel("epoch_number",size=14)
    # plt.legend(loc='best',fontsize='large')
    ax22 = ax2.twinx()
    cf22=ax22.plot(np.arange(1,FP.shape[0]+1),TS[:,1],  "k-.",label="TS")
    # ax12.set_xticks(np.arange(0, TP.shape[1]+1, 2))
    ax22.set_ylabel("TS",size=14)
    ax22.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    plt.legend(cf2+cf21+cf22,['TP','FP','TS'],loc='best',fontsize='large')
    # ax22 = ax2.twinx()
    # cf22=ax22.plot(np.arange(1,FP.shape[0]+1),FP[:,1],  "b-",label="FAR")
    # # ax22.set_xticks(np.arange(0, TP.shape[1]+1, 2))
    # ax22.set_ylabel("FP",size=14)
    # ax22.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    # plt.legend(cf2+cf22,['TP','FP'],loc='best',fontsize='large')
    # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax2.transAxes)
    plt.title("Rainfall>=3mm",size=16) 

    ax3 = fig.add_subplot(223)
    cf3=ax3.plot(np.arange(1,TP.shape[0]+1),TP[:,2],  "r-",label="TP")
    cf31=ax3.plot(np.arange(1,FP.shape[0]+1),FP[:,2],  "b-",label="FP")
    # ax1.plot(np.arange(1,pod.shape[0]+1),pod[:,2],  "g-",label="ts_10")
    # ax1.plot(np.arange(1,pod.shape[0]+1),pod[:,3],  "y-",label="ts_20")
    # plt.legend(loc='best',fontsize='large')
    ax3.set_xlim(1, TP.shape[0])
    ax3.set_ylabel("TP",size=14)
    ax3.set_xticks(np.arange(1, TP.shape[0]+1, 2))
    ax3.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    ax3.set_xlabel("epoch_number",size=14)
    # plt.legend(loc='best',fontsize='large')
    ax32 = ax3.twinx()
    cf32=ax32.plot(np.arange(1,FP.shape[0]+1),TS[:,2],  "k-.",label="TS")
    # ax12.set_xticks(np.arange(0, TP.shape[1]+1, 2))
    ax32.set_ylabel("TS",size=14)
    # ax32.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    plt.legend(cf3+cf31+cf32,['TP','FP','TS'],loc='best',fontsize='large')
    # ax32 = ax3.twinx()
    # cf32=ax32.plot(np.arange(1,FP.shape[0]+1),FP[:,2],  "b-",label="FAR")
    # # ax32.set_xticks(np.arange(0, TP.shape[1]+1, 2))
    # ax32.set_ylabel("FP",size=14)
    # ax32.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    # plt.legend(cf3+cf32,['TP','FP'],loc='best',fontsize='large')
    # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax3.transAxes)
    plt.title("Rainfall>=10mm",size=16) 

    ax4 = fig.add_subplot(224)
    cf4=ax4.plot(np.arange(1,TP.shape[0]+1),TP[:,3],  "r-",label="Tp")
    cf41=ax4.plot(np.arange(1,FP.shape[0]+1),FP[:,3],  "b-",label="FP")
    # ax1.plot(np.arange(1,pod.shape[0]+1),pod[:,2],  "g-",label="ts_10")
    # ax1.plot(np.arange(1,pod.shape[0]+1),pod[:,3],  "y-",label="ts_20")
    # plt.legend(loc='best',fontsize='large')
    ax4.set_xlim(1, TP.shape[0])
    ax4.set_ylabel("TP",size=14)
    ax4.set_xticks(np.arange(1, TP.shape[0]+1, 2))
    ax4.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    ax4.set_xlabel("epoch_number",size=14)
    # plt.legend(loc='best',fontsize='large')
    ax42 = ax4.twinx()
    cf42=ax42.plot(np.arange(1,FP.shape[0]+1),TS[:,3],  "k-.",label="TS")
    # ax12.set_xticks(np.arange(0, TP.shape[1]+1, 2))
    # ax42.set_ylabel("TS",size=14)
    ax42.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    plt.legend(cf4+cf41+cf42,['TP','FP','TS'],loc='best',fontsize='large')
    # ax42 = ax4.twinx()
    # cf42=ax42.plot(np.arange(1,FP.shape[0]+1),FP[:,3],  "b-",label="FAR")
    # # ax42.set_xticks(np.arange(0, TP.shape[1]+1, 2))
    # ax42.set_ylabel("FP",size=14)
    # ax42.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    # plt.legend(cf4+cf42,['TP','FP'],loc='best',fontsize='large')
    # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax4.transAxes)
    plt.title("Rainfall>=20mm",size=16) 
    # plt.tight_layout()
    plt.savefig(os.path.join(sdir,sname+'.png'), dpi=300,bbox_inches = 'tight')
    plt.close()
    #plt.show()

    # ax2 = fig.add_subplot(122)
    # ax2.plot(np.arange(1,train_loss.size+1),train_loss,  "r-",label="train_loss")
    # ax2.plot(np.arange(1,val_loss.size+1),val_loss,  "b-",label="valid_loss")
    # plt.legend(loc='best',fontsize='large')
    # ax2.set_xlim(1, train_loss.size)
    # ax2.set_ylabel("Loss",size=14)
    # ax2.set_xlabel("epoch_number",size=14)
    # plt.title("LOSS",size=16) 
    
def draw_tp_fp(TP,FP,TS,sdir,sname,labels=['WCE','OWBCE','Dice','OBDice','DTS']):
    #colors=list(mcolors.TABLEAU_COLORS.keys()) #color=mcolors.TABLEAU_COLORS[colors[i]],label=str(step*(i+1)*tau)[:3]
    fig = plt.figure(figsize=(10,4))
    st=['r-','b-','g-','y-','k-']
    TP=TP[:,:20]
    FP=FP[:,:20]
    tf=FP/TP
    TS=TS[:20]
    ax1 = fig.add_subplot(121)
    for i in range(5):
        ax1.plot(np.arange(1,TP.shape[1]+1),TP[i,:],st[i],label=labels[i])  
    ax1.set_xlim(1, TP.shape[1])
    ax1.set_ylabel("TP",size=14)
    ax1.ticklabel_format(style='sci',scilimits=(0,0), axis='y')
    ax1.set_xticks(np.arange(1, TP.shape[1]+1, 2))
    ax1.set_xlabel("epoch_number",size=14)
    
    plt.tick_params(labelsize=13)
    plt.legend(loc='upper right')
    plt.title(f"TP_Rainfall>={sname}mm",size=16)

    ax2 = fig.add_subplot(122)
    for i in range(5):
        ax2.plot(np.arange(1,TP.shape[1]+1),tf[i,:],  st[i],label=labels[i])  
    ax2.set_xlim(1, TP.shape[1])
    ax2.set_ylabel("FP/TP",size=14)
    ax2.set_xticks(np.arange(1, TP.shape[1]+1, 2))
    ax2.set_xlabel("epoch_number",size=14)
    plt.tick_params(labelsize=13)
    plt.legend(loc='upper right')
    plt.title(f"FP/TP_Rainfall>={sname}mm",size=16) 
    plt.tight_layout()
    plt.savefig(os.path.join(sdir,'TF1_'+sname+'.eps'),dpi=300)
    plt.savefig(os.path.join(sdir,'TF1_'+sname+'.png'),dpi=300)
    plt.close()

def draw_K_fold():
    save_dir=os.path.join(cfg.SAVE_ROOT,'K-Fold')
    sdir=os.path.join(cfg.SAVE_ROOT,'figures')
    if not os.path.exists(sdir):os.mkdir(sdir)
    NWP_path=os.path.join(cfg.MEAN_STD_DIR,'sta_eva.csv')
    sls_train=[]
    sls_val=[]
    ls_result=[]
    ls_bts=[]
    LNS=[]
    for i in range(0,5):
        scalar_path=os.path.join(save_dir,f'fold_{i+1}','scalars',"scalars.pkl")
        if not os.path.exists(scalar_path): continue
        sname=f'fold_{i+1}_scalars'
        with open(scalar_path,'rb') as f:
            Dic=pickle.load(f)
        train_loss=np.array(Dic['train_loss'])
        val_loss=np.array(Dic['valid_loss'])
        result=Dic['ts_result']
        pod=np.array(Dic['pod'])
        far=np.array(Dic['far'])
        lns=np.array([Dic['lns']])
        draw_tp_fp(lns.squeeze()[:,:,0],lns.squeeze()[:,:,1],result,sdir,f'fold{i+1}_TP_FP')
        LNS.append(lns)
        
        # draw_loss_line(result,train_loss,val_loss,sdir,sname)
        bestepoch=Dic['best_epoch']
        bts=result[bestepoch]
        ls_result.append(result)
        ls_bts.append(bts)
        # draw_ts(bts,sname,sdir)
        sls_train.append(train_loss)
        sls_val.append(val_loss)
    ls_bts=np.vstack(ls_bts).mean(axis=0)
    sls_train=np.array(sls_train).mean(axis=0)
    sls_val=np.array(sls_val).mean(axis=0)
    ls_result=np.array(ls_result).mean(axis=0)
    LNS=np.vstack(LNS).mean(axis=0)
    draw_pod_far(LNS[:,:,0],LNS[:,:,1],ls_result,sdir,'TP_FP')
    draw_loss_line(result,sls_train,sls_val,sdir,'kfold_mean')
    # draw_ts(ls_bts,'kfold_mean',sdir)
    print('finished!')

def draw_4methods():
    rootpath=[]
    rootpath.append('/root/YXX/NP_UNET/datas/output/unet_grid/rain')
    rootpath.append('/root/YXX/NP_UNET/datas/output/Unet_grid_PR_ORBCE/rain')
    rootpath.append('/root/YXX/NP_UNET/datas/output/Unet_grid_dice_pixleshuffle/rain')
    rootpath.append('/root/YXX/NP_UNET/datas/output/Unet_grid_dice_OR_pixle/rain')
    rootpath.append('/root/YXX/NP_UNET/datas/output/Tnet_grid_WMS_TS/rain')
    TP,FP,TS=[],[],[]
    for rdir in rootpath: 
        save_dir=os.path.join(rdir,'K-Fold')
        sdir=os.path.join(cfg.SAVE_ROOT,'figures')
        if not os.path.exists(sdir):os.mkdir(sdir)
        sls_train=[]
        sls_val=[]
        ls_result=[]
        ls_bts=[]
        LNS=[]
        for i in range(0,5):
            scalar_path=os.path.join(save_dir,f'fold_{i+1}','scalars',"scalars.pkl")
            if not os.path.exists(scalar_path): continue
            sname=f'fold_{i+1}_scalars'
            with open(scalar_path,'rb') as f:
                Dic=pickle.load(f)
            result=Dic['ts_result']
            lns=np.array([Dic['lns']])
            LNS.append(lns[:,:20,...])
            ls_result.append(result[:20])
        ls_result=np.array(ls_result).mean(axis=0)
        LNS=np.vstack(LNS).mean(axis=0)
        TP.append(LNS[:,:,0])
        FP.append(LNS[:,:,1])
        TS.append(ls_result)
    TP=np.array(TP)
    FP=np.array(FP)
    TS=np.array(TS)
    tsname=['0.1','3','10','20']
    for i in range(4):
        draw_tp_fp(TP[:,:,i],FP[:,:,i],TS[:,:,i],sdir,tsname[i])
    print('finished!')

def draw_single():
    save_dir=os.path.join(cfg.SAVE_ROOT)
    sdir=os.path.join(cfg.SAVE_ROOT,'figures')
    if not os.path.exists(sdir):os.mkdir(sdir)
    scalar_path=os.path.join(save_dir,'scalars',"scalars.pkl")
    if not os.path.exists(scalar_path): return
    sname='pod_far'
    with open(scalar_path,'rb') as f:
        Dic=pickle.load(f)
    train_loss=np.array(Dic['train_loss'])
    val_loss=np.array(Dic['valid_loss'])
    result=Dic['ts_result']   
    pod=np.array(Dic['pod'])
    far=np.array(Dic['far'])
    lns=np.array([Dic['lns']])
    # draw_pod_far(pod,far,sdir,sname)
    draw_loss_line(result,train_loss,val_loss,sdir,sname)
    # bestepoch=Dic['best_epoch']
    # bts=result[bestepoch]
    # draw_ts(bts,sname,sdir)
    print('finished!')

def draw_pred_analysis(dicts,sdir):
    rmse=np.nanmean(dicts['rmse'],axis=1)
    rmse_00=np.nanmean(dicts['rmse_00'],axis=1)
    rmse_12=np.nanmean(dicts['rmse_12'],axis=1)

    mae=np.nanmean(dicts['mae'],axis=(0,2))
    mae_00=np.nanmean(dicts['mae_00'],axis=(0,2))
    mae_12=np.nanmean(dicts['mae_12'],axis=(0,2))
    fig = plt.figure(figsize=(16,14))
    ax1 = fig.add_subplot(221)   #nrows,ncolumns,index
    ax1.bar( np.arange(0,mae.size)-0.2,mae,color="g", width=0.2,label="mae") 
    ax1.bar( np.arange(0,mae_00.size),mae_00,color="b", width=0.2,label="mae_00") 
    ax1.bar( np.arange(0,mae_12.size)+0.2,mae_12,color="r", width=0.2,label="mae_12") 
    plt.xticks(np.arange(mae.size),((np.arange(mae.size))*3+12))
    ax1.set_xlabel("Times",size=14)
    ax1.set_ylabel("MAE",size=14) 
    plt.legend(loc='lower right',fontsize='x-large')
    plt.title('mae:{:.3f},mae_00:{:.3f},mae_12:{:.3f}'.format(mae.mean(),mae_00.min(),mae_12.min()),loc='left',size=14)
    plt.title('MAE',loc='right',size=16)
    #plt.title('NWP and TrajGRU MAE',size=16)

    ax2 = fig.add_subplot(222)
    ax2.bar( np.arange(0,rmse.size)-0.2,rmse,color="g", width=0.2,label="TrajGRU") 
    ax2.bar( np.arange(0,rmse_00.size),rmse_00,color="b", width=0.2,label="rmse_00") 
    ax2.bar( np.arange(0,rmse_12.size)+0.2,rmse_12,color="r", width=0.2,label="rmse_12") 
    plt.xticks(np.arange(rmse.size),((np.arange(rmse.size))*3+12))
    ax2.set_xlabel("Times",size=14)
    ax2.set_ylabel("RMSE",size=14) 
    plt.legend(loc='lower right',fontsize='x-large',facecolor='white')
    plt.title('rmse:{:.3f},rmse_00:{:.3f},rmse_12:{:.3f}'.format(rmse.mean(),rmse_00.min(),rmse_12.min()),loc='left',size=14)
    plt.title('RMSE',loc='right',size=16)
    #plt.title('NWP and TrajGRU RMSE',size=16)
    

    mae_max=np.nanmax(dicts['mae'],axis=(0,2))
    mae_00_max=np.nanmax(dicts['mae_00'],axis=(0,2))
    mae_12_max=np.nanmax(dicts['mae_12'],axis=(0,2))

    mae_min=np.nanmin(dicts['mae'],axis=(0,2))
    mae_00_min=np.nanmin(dicts['mae_00'],axis=(0,2))
    mae_12_min=np.nanmin(dicts['mae_12'],axis=(0,2))

    ax3 = fig.add_subplot(223)   #nrows,ncolumns,index
    ax3.plot(np.arange(1,mae_max.shape[0]+1),mae_max,color='r',linestyle='-',label='mae_max')
    ax3.plot(np.arange(1,mae_min.shape[0]+1),mae_min,color='b',linestyle='-',label='mae_min')
    ax3.plot(np.arange(1,mae.shape[0]+1),mae,color='g',linestyle='-',label='mae')
    ax3.set_xlim(1, rmse.shape[0])
    #ax1.set_ylim(0, 0.01)
    ax3.set_xlabel("epoch_number",size=14)
    ax3.set_ylabel("MAE&RMSE",size=14)    
    # ax12.set_ylabel("Acc",color='b')
    plt.legend(loc='center right',fontsize='x-large')
    plt.title("MAE",size=16)
    ax4 = fig.add_subplot(224)
    ax4.plot(np.arange(1,mae_00_max.size+1),mae_00_max,  "r-",label="mae_00_max")
    ax4.plot(np.arange(1,mae_00_min.size+1),mae_00_min,  "r-.",label="mae_00_min")
    ax4.plot(np.arange(1,mae_12_max.size+1),mae_12_max,  "b-",label="mae_12_max")
    ax4.plot(np.arange(1,mae_12_min.size+1),mae_12_min,  "b-.",label="mae_12_min")
    ax4.plot(np.arange(1,mae_00.size+1),mae_00,  "g-",label="mae_00")
    ax4.plot(np.arange(1,mae_12.size+1),mae_12,  "g-.",label="mae_12")
    plt.legend(loc='center right',fontsize='x-large')
    ax4.set_xlim(1, mae_00.size)
    ax4.set_ylabel("MAE",size=14)
    plt.xticks(np.arange(rmse.size),((np.arange(rmse.size))*3+12))
    ax2.set_xlabel("Times",size=14)
    plt.title("MAE ANALYSIS",size=16) 
    plt.savefig(os.path.join(sdir,'preds_analysis.png'), dpi=300,bbox_inches = 'tight')


if __name__=="__main__":
    # fname='/root/YXX/Racing/datas/ouput/traj/scalars/scalars.pkl'
    # sdir='/root/YXX/Racing/datas/ouput/traj/scalars'
    # sname='train_temp'
    # scalar_analysis(fname,sdir,sname)
    # draw_K_fold()
    draw_4methods()
    # draw_single()
    
    