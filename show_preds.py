import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
#import maskout

def data_to_dat(dicts,save_dir):
    fname=dicts['fname']
    ls_preds=dicts['preds']
    ls_obs=dicts['obs']
    ls_NWP=dicts['NWP']
    # H,W=ls_preds[0].shape
    lon=np.arange(107.375, 115.25+0.125, 0.125)
    lat=np.arange(31.5, 23.625-0.125, -0.125)
    for i,pf in enumerate(fname):
        if pf not in ['2021052112_03','2021052800_12','2021060900_05','2021062700_04']: continue
        preds=ls_preds[i][::-1,:].flatten().astype('float32')
        obs=ls_obs[i][::-1,:].flatten().astype('float32')
        NWP=ls_NWP[i][::-1,:].flatten().astype('float32')
        np.save(os.path.join(save_dir,'preds_'+pf.split('.')[0]+'.npy'),preds)


def show_tp(dicts,save_dir):
    fname=dicts['fname']
    ls_preds=dicts['preds']
    ls_obs=dicts['obs']
    ls_NWP=dicts['NWP']
    # H,W=ls_preds[0].shape
    lon=np.arange(107.375, 115.25+0.125, 0.125)
    lat=np.arange(31.5, 23.625-0.125, -0.125)
    # lon,lat=np.meshgrid(np.arange(W),np.arange(H))
    proj = ccrs.PlateCarree() 
    extent = [lon.min(),lon.max(),lat.min(),lat.max()]
    reader = shpreader.Reader('/root/YXX/NP_UNET/datas/m/gadm36_CHN_1.shp')
    colorlevel=[0.1,3.0,10.0,20.0,50.0]#雨量等级
    colordict=['#A6F28F','#3DBA3D','#61BBFF','#0000FF']#颜色列表
    rain_map=mcolors.ListedColormap(colordict)#产生颜色映射
    rain_map.set_under('black')
    rain_map.set_over('#0000FF')
    norm=mcolors.BoundaryNorm(colorlevel,rain_map.N)#生成索引
    for i,pf in enumerate(fname):
        if pf not in ['2021052112_03','2021052800_12','2021060900_05','2021062700_04']: continue
        preds=ls_preds[i]
        obs=ls_obs[i]
        NWP=ls_NWP[i]
        
        # 创建画图空间
        # ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
        
        fig = plt.figure(figsize=(16,5)) #创建页面
        (ax1,ax2,ax3) = fig.subplots(1, 3,subplot_kw={'projection': proj}) #子图

        # gl = ax1.gridlines( draw_labels=True,linewidth=1.2, color='k', alpha=0.5, linestyle='--')
        # gl.xlabels_top = False #关闭顶端标签
        # gl.ylabels_right = False #关闭右侧标签  
        ax1.set_extent(extent,crs=ccrs.PlateCarree())
        ax1.add_geometries(reader.geometries(),ccrs.PlateCarree(),edgecolor = 'k',facecolor = 'none')
        cf=ax1.contourf(lon,lat,obs,levels=colorlevel,cmap=rain_map,norm=norm,extend='max',transform=ccrs.PlateCarree())
        cf.cmap.set_over('#0000FF')
        ax1.set_xticks(np.arange(108., 115.5, 2))
        ax1.set_yticks(np.arange(32, 23.625, -2))
        ax1.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax1.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax1.set_title('Ground Truth')
        # ax1.set_adjustable('box')
    
        # gl2 = ax2.gridlines(draw_labels=True,
        # linewidth=1.2, color='k', alpha=0.5, linestyle='--')
        # gl2.xlabels_top = False #关闭顶端标签
        # gl2.ylabels_right = False #关闭右侧标签
        # gl2.ylabels_left = False #关闭右侧标签
        ax2.set_extent(extent,crs=ccrs.PlateCarree())
        ax2.add_geometries(reader.geometries(),ccrs.PlateCarree(),edgecolor = 'k',facecolor = 'none')
        cf2=ax2.contourf(lon,lat,NWP,levels=colorlevel,cmap=rain_map,norm=norm,extend='max',transform=ccrs.PlateCarree())
        cf2.cmap.set_over('#0000FF')
        ax2.set_xticks(np.arange(108., 115.5, 2))
        ax2.set_yticks(np.arange(32, 23.625, -2))
        ax2.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax2.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax2.set_title('ECMWF')
        # ax2.set_adjustable('box')

        # gl3 = ax3.gridlines( draw_labels=True,
        # linewidth=1.2, color='k', alpha=0.5, linestyle='--')
        # gl3.xlabels_top = False #关闭顶端标签
        # gl3.ylabels_right = False #关闭右侧标签
        # gl3.ylabels_left = False #关闭右侧标签
        
        ax3.add_geometries(reader.geometries(),ccrs.PlateCarree(),edgecolor = 'k',facecolor = 'none')
        cf3=ax3.contourf(lon,lat,preds,levels=colorlevel,cmap=rain_map,norm=norm,extend='max',transform=ccrs.PlateCarree())
        cf3.cmap.set_over('#0000FF')
        ax3.set_xticks(np.arange(108., 115.5, 2))
        ax3.set_extent(extent,crs=ccrs.PlateCarree())
        ax3.set_yticks(np.arange(32, 23.625, -2))
        ax3.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax3.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax3.set_title('ODice-PCM')

        plt.subplots_adjust(wspace=0.00)
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax=(ax1,ax2,ax3))
        # cax = divider.append_axes("bottom", size="100%", pad=0.05)
        # labels=['0.1','3.0','10.0','>=20.0','']
        # plt.tight_layout()
        plt.colorbar(cf2,orientation='horizontal',ax=(ax1,ax2,ax3),fraction=0.046, pad=0.08)
        #解决中文显示#
        plt.savefig(os.path.join(save_dir,pf.split('.')[0]+'.eps'), format='eps',dpi=300,bbox_inches = 'tight')
        plt.savefig(os.path.join(save_dir,pf.split('.')[0]+'.png'),dpi=300,bbox_inches = 'tight')
        print('save',os.path.join(save_dir,pf.split('.')[0]+".png"))
        plt.close()
   
        


if __name__=="__main__":
    import pickle
    from config import cfg
    ti=['2021052112_03','2021052800_12','2021060900_05','2021062700_04']
    mname=['unet_grid','Unet_grid_PR_ORBCE','Unet_grid_dice_pixleshuffle','Unet_grid_dice_OR_pixle']
    for fp in mname[1:]:
        save_dir=f'/root/YXX/NP_UNET/datas/output/{fp}/rain/K-Fold'
        # save_dir=os.path.join(cfg.SAVE_ROOT,'K-Fold')
        with open(os.path.join(save_dir,'test_pred.pkl'),'rb') as f:
            dicts=pickle.load(f)
        img_save_dir=f'/root/YXX/NP_UNET/datas/output/{fp}/rain/test_out4'
        if not os.path.exists(img_save_dir):os.mkdir(img_save_dir)
        data_to_dat(dicts,img_save_dir)
    for t in ti:
        ens=[]
        for fp in mname:
            fname=f'/root/YXX/NP_UNET/datas/output/{fp}/rain/test_out4/preds_{t}.npy'
            ens.append(np.load(fname))
        ens=np.hstack(ens)
        ens.tofile(f'/root/YXX/NP_UNET/datas/output/unet_grid/rain/test_out4/merge_{t}.dat')
        print(ens.shape,ens.size)
