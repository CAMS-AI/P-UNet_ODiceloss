"""
@Time    : 2021-12-30 16:00
@Author  : Xiaoxiong
@File    : predict.py
"""
import numpy as np
import sys
sys.path.append("..") 
import pandas as pd
import glob,os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from dataset import get_loaders,get_idx
from tqdm import tqdm
import pickle
import shutil
from config import cfg
from analyze import draw_max_min,draw_rmse_bar,draw_ts,draw_scatter
from utils import ReadStaPoints_test,get_mean_std,create_net,ReadPointsData,get_x_mean_std
from evaluate_rain2 import evaluate
from show_preds import data_to_dat, show_tp



def save_test_predictions(loader, model,y_mean_std,mask=None,
                    station_path='/root/YXX/Racing/datas/test/', 
                    folder="datas/output"):
    folder=os.path.join(folder,'preds')
    if  os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    stations=ReadStaPoints_test(cfg.STATION_OLD)    
    model.eval()
    #if not os.path.exists(os.path.join(folder,'npy')):os.mkdir(os.path.join(folder,'npy'))
    class_values=np.array([0.,1.5,7.,15.,30.])
    with torch.no_grad():
        for idx, (x, fnames) in enumerate(tqdm(loader)):
            x = x.float().to(device=cfg.DEVICE)                      
            preds=model(x)
            _,preds=torch.max(preds.data,dim=1)
            #preds=preds*y_mean_std[1]+y_mean_std[0]
            preds=preds.cpu().numpy()
            preds=np.squeeze(preds)
            preds=class_values[preds]
            if mask is not None:preds=preds*mask
            for i,fp in enumerate(fnames):
                if '.npy' in fp:fp=str.split(fp,'.')[0]
                np.save(os.path.join(folder,fp+'.npy'),preds[i])                
    print('Test Prediction finished!')

def K_fold_mean(K,sname='Pred_precipitation',result_dir=cfg.SAVE_ROOT):
      
    save_dir=os.path.join(cfg.SAVE_ROOT,'K-Fold')
    if not os.path.exists(os.path.join(save_dir,sname)):os.mkdir(os.path.join(save_dir,sname))
    assert os.path.exists(save_dir)
    flist=os.listdir(os.path.join(cfg.TEST_STATION_DIR))
    stations=ReadStaPoints_test(cfg.STATION_OLD)
    class_values=np.array([0.,1.5,7.,15.,30.])
    flist=np.array(flist)
    flist.sort()
    for fn in tqdm(flist):
        txt_dir=os.path.join(os.path.join(save_dir,sname),fn)
        if not os.path.exists(txt_dir):os.mkdir(txt_dir)
        for i in range(9):
            pname=fn+'_'+str(i+1).zfill(2)+'.npy'
            preds=[]
            for j in range(K):
                folder=os.path.join(save_dir,f'fold_{j+1}','preds')
                if not os.path.exists(folder):continue
                if not os.path.exists(os.path.join(folder,pname)):
                    # print('Prediction ',fn,' not exists')
                    continue
                preds.append(np.load(os.path.join(folder,pname)))
                #if len(preds)==0:continue
            preds=np.array(preds).squeeze()
            if preds.ndim>2:
                thresholds = [0.0] + list(cfg.RAIN.THRESHOLDS)
                # print(thresholds)
                preds=torch.from_numpy(preds)
                class_index = torch.zeros_like(preds).long()
                for thre, threshold in enumerate(thresholds):
                    class_index[preds >= threshold] = thre
                count_idx=[]
                for thre in range(len(thresholds)):
                    count_idx.append(torch.sum(class_index==thre,dim=0).unsqueeze(0))
                count_idx=torch.cat(count_idx,dim=0)
                _,p_idx=torch.max(count_idx,dim=0)
                preds=class_values[p_idx]                
                #preds=preds.mean(axis=0)
            points=ReadStaPoints_test(os.path.join(cfg.TEST_STATION_DIR
                                    ,fn,f'ji_loc_inputs_{str(i+1).zfill(2)}.txt'))

            points[:,0]=points[:,0]-(69-64)//2
            points[:,1]=points[:,1]-(73-64)//2
            pdata=ReadPointsData(preds,points)

           # if pdata.max()>20:print(os.path.join(txt_dir,f'pred_{str(i+1).zfill(2)}.txt'))
            df = pd.DataFrame(np.around(pdata.reshape(-1,1),3).astype(str))
            df.to_csv(os.path.join(txt_dir,f'pred_{str(i+1).zfill(2)}.txt'), sep='\t', index=False,header=None)
    
    make_targz(os.path.join(result_dir,f'{sname}.tar.gz'), os.path.join(save_dir,sname))
    print(os.path.join(result_dir,f'{sname}.tar.gz'+' created!'))

def  make_targz(output_filename, source_dir):
    import tarfile
    with tarfile. open (output_filename,  "w:gz" ) as tar:
        tar.add(source_dir, arcname = os.path.basename(source_dir))

def save_val_predictions(loader, model,
                    folder="datas/output"):

    if not os.path.exists(os.path.join(folder,'preds')):os.mkdir(os.path.join(folder,'preds'))
    model.eval()
    # tpidx=get_idx('Total precipitation',cfg.SELECTED_VALS)
    class_values=np.array([0.,1.5,7.,15.,30.])
    ls_preds=[]
    ls_y=[]
    ls_fname=[]
    ls_NWP=[]
    tp=[]
    v_idx=get_idx(['tp'],cfg.VAL_NAMES)
    vals_idx=get_idx(['tp'],cfg.SELECTED_VALS)
    x_mean_std=get_x_mean_std(v_idx).squeeze()
    with torch.no_grad():
        for idx, (x,y,fnames) in enumerate(loader):
            x = x.float().to(cfg.DEVICE)   
            tp=x[:,vals_idx,:,:].squeeze()
            tp=tp*x_mean_std[1]+x_mean_std[0]
            tp[tp<0.1]=0          
            preds=model(x)
            preds=torch.sigmoid(preds)
            preds = torch.sum((preds > 0.5), dim=1)
            y=torch.squeeze(y).to(cfg.DEVICE) 
            ls_fname+=fnames
            ls_y.append(y)
            ls_NWP.append(tp)
            ls_preds.append(preds) 
    
    ls_preds=torch.cat(ls_preds,dim=0)
    ls_y=torch.cat(ls_y,dim=0)
    ls_NWP=torch.cat(ls_NWP,dim=0)
    # ts=evaluate(ls_preds,ls_y)
    
    ts=evaluate(ls_preds,ls_y)
    print("ts_0.1:",ts[0],"ts_3:",ts[1],"ts_10:",ts[2],"ts_20:",ts[3])
    sdir=os.path.join(folder,'evaluattion')
    if not os.path.exists(sdir):os.mkdir(sdir)
    # draw_ts(ts,'ts',os.path.join(folder,'scalars'))
    # ls_preds=class_values[ls_preds.cpu().numpy()]
    data_dic={}
    data_dic['ts']=ts
    data_dic['preds']=ls_preds.cpu().numpy()
    data_dic['obs']=ls_y.cpu().numpy()
    data_dic['NWP']=ls_NWP.cpu().numpy()
    data_dic['name']=ls_fname

    
    # draw_ts(ts_sta,'ts_sta',os.path.join(folder,'scalars'))
    with open(os.path.join(folder,'scalars','val_pred1.pkl'),'wb') as f:
        pickle.dump(data_dic,f,pickle.HIGHEST_PROTOCOL)
    print(os.path.join(folder,'scalars','val_pred1.pkl'),' saved!')
    print('Prediction finished!')

def evaluate_EC():
    test_dir=cfg.DATA_DIR
    test_list=np.array(os.listdir(os.path.join(test_dir,'x')))
    test_list=test_list[(test_list>'2021010100_00.npy')]
    # test_list=test_list[(test_list>'2020010100_00.npy')&(test_list<'2021010100_00.npy')]

    test_loader = get_loaders(
        test_dir,test_list,y_name=cfg.Y_NAME,vals_idx=get_idx(['tp'],cfg.VAL_NAMES),batch_size=cfg.BATCH_SIZE,transform=None,
        num_works=cfg.NUM_WORKERS,shuffle=False)
    x_ls=[]
    y_ls=[]
    for idx, (x,y) in enumerate(tqdm(test_loader)):
        x=x.squeeze()
        x[x<0]=0
        # p=x.numpy()
        # print(p.max(),p.min())
        x_ls.append(x.to(cfg.DEVICE) )               
        y_ls.append(y.to(cfg.DEVICE) )
    x_ls=torch.cat(x_ls,dim=0)
    y_ls=torch.cat(y_ls,dim=0)
    ts=evaluate(x_ls,y_ls,test=True)
    total=torch.numel(y_ls)
    r1=((y_ls>=0)&(y_ls<0.1)).sum()/total
    print('no rain:',r1.item())
    r2=((y_ls>=0.1)&(y_ls<3)).sum()/total
    print('0.1<=rain<3:',r2.item())
    r3=((y_ls>=3)&(y_ls<10)).sum()/total
    print('3<=rain<10:',r3.item())
    r4=((y_ls>=10)&(y_ls<20)).sum()/total
    print('10<=rain<20:',r4.item())
    r5=(y_ls>=20).sum()/total
    r50=(y_ls>=50).sum()/total
    r100=(y_ls>=100).sum()/total
    print('20<=rain:',r5.item())
    print('50<=rain:',r50.item())
    print('100<=rain:',r100.item())
    print(ts)
            


def evaluate_stations():
    points=ReadStaPoints_test(cfg.STATION_LIST)
    save_dir=os.path.join(cfg.SAVE_ROOT,'K-Fold')
    if not os.path.exists(os.path.join(save_dir,'ALL_EVA')):os.mkdir(os.path.join(save_dir,'ALL_EVA'))
    mae=[]

    for i in range(10):
        folder=os.path.join(save_dir,f'fold_{i+1}')
        if not os.path.exists(os.path.join(folder,'scalars','val_pred.pkl')):continue
        with open(os.path.join(folder,'scalars','val_pred.pkl'),'rb') as f:
            data_dic=pickle.load(f)
        mae.append(data_dic['mae'])
    mae=np.vstack(mae)
    mae=np.nanmean(mae,axis=(0,1))
    print('mae-0.8 :',(mae<0.8).sum())
    print('mae0.8-0.9 :',((mae>=0.8)&(mae<0.9)).sum())
    print('mae0.9-1.0 :',((mae>=0.9)&(mae<1.0)).sum())
    print('mae1.-1.1 :',((mae>=1.0)&(mae<1.1)).sum())
    print('mae1.1-1.2 :',((mae>=1.1)&(mae<1.2)).sum())
    print('mae1.2-1.3 :',((mae>=1.2)&(mae<1.3)).sum())
    print('mae-1.3 :',(mae>=1.3).sum())
    draw_scatter(mae[mae<1],points[mae<1],'maelt1',os.path.join(save_dir,'ALL_EVA'))
    draw_scatter(mae[mae>=1],points[mae>=1],'maege1',os.path.join(save_dir,'ALL_EVA'))


        


def test_predict(preds_dir,test_list,test_dir=cfg.TEST_DIR,vals_idx=None,model_path=None,validate=False):
    '''
        调用最佳模型进行测试集计算，并保存结果
        save_test_predictions()为格点站点均保存，站点输出按竞赛要求
        save_predictions()仅保存格点
    ''' 
    print('start to predict...')
    model=create_net(train=False)
    y_mean_std,transform=get_mean_std(cfg.Y_NAME,vals_idx)
    #test_list=os.listdir(os.path.join(cfg.TEST_DIR,"x"))
    test_loader = get_loaders(
        test_dir,test_list,y_name=cfg.Y_NAME,vals_idx=vals_idx,batch_size=cfg.BATCH_SIZE,transform=transform,
        num_works=cfg.NUM_WORKERS,pin_memory=cfg.PIN_MEMORY,shuffle=False,forecast=True,validate=validate)
    model.load_state_dict(torch.load(model_path))
    save_val_predictions(test_loader,model,folder=preds_dir)
    # else:
    #     save_test_predictions(test_loader,model,y_mean_std,station_path=station_path,mask=None,folder=preds_dir)

def predict_K_fold(K,validate=False):
    save_dir=os.path.join(cfg.SAVE_ROOT,'K-Fold')
    test_dir=cfg.DATA_DIR
    data_list=os.listdir(os.path.join(test_dir,'x'))
    data_list=np.array(data_list)
    test_list=data_list[(data_list>'2021010100_00.npy')]
    # test_list=data_list[(data_list>'2021052000_00.npy')&(data_list<'2021063000_00.npy')]
    for i in range(0,5):
        scalar_path=os.path.join(save_dir,f'fold_{i+1}','scalars')
        if not os.path.exists(os.path.join(scalar_path,"scalars.pkl")):continue
        with open(os.path.join(scalar_path,"scalars.pkl"), 'rb') as f:
            dicts=pickle.load(f)
        bestepoch=dicts['best_epoch'] #获取最佳epoch序号
        model_path=os.path.join(save_dir,f'fold_{i+1}','pth',cfg.Y_NAME+f'_{bestepoch}.pth')
        #model_path=os.path.join(save_dir,f'fold_{i+1}','pth',cfg.Y_NAME+f'_{135}.pth')
        preds_dir=os.path.join(save_dir,f'fold_{i+1}')
        print(f'fold_{i+1}')
        # test_list=os.listdir(os.path.join(cfg.TEST_DIR,"x"))
        test_dir=cfg.DATA_DIR
        test_predict(preds_dir,test_list,test_dir,vals_idx=get_idx(cfg.SELECTED_VALS,cfg.VAL_NAMES),model_path=model_path,validate=validate)
    print('finished!')

def predict_single():
    scalar_path=os.path.join(cfg.SAVE_ROOT,'scalars')
    ls_result=[]
    with open(os.path.join(scalar_path,"scalars.pkl"), 'rb') as f:
        dicts=pickle.load(f)
    bestepoch=dicts['best_epoch'] #获取最佳epoch序号

    model_path=os.path.join(cfg.SAVE_ROOT,'pth',cfg.Y_NAME+f'_{bestepoch}.pth')
    #model_path=os.path.join(save_dir,f'fold_{i+1}','pth',cfg.Y_NAME+f'_{138}.pth')
    preds_dir=cfg.SAVE_ROOT
    test_dir=cfg.DATA_DIR
    data_list=os.listdir(os.path.join(test_dir,'x'))
    data_list=np.array(data_list)
    test_list=data_list[(data_list>'2021010100_00.npy')]
    val_list=data_list[(data_list>'2020010100_00.npy')&(data_list<'2021010100_00.npy')]
    # test_predict(preds_dir,val_list,test_dir,vals_idx=get_idx(cfg.SELECTED_VALS,cfg.VAL_NAMES),model_path=model_path,validate=True)
    test_predict(preds_dir,test_list,test_dir,vals_idx=get_idx(cfg.SELECTED_VALS,cfg.VAL_NAMES),model_path=model_path,validate=True) 
    # evaluate_EC()

def test():
    folder=os.path.join(cfg.SAVE_ROOT,'K-Fold')
    ls_result=[]
    for i in range(5):
        print(f'fold_{i+1}')
        scalar_path=os.path.join(folder,f'fold_{i+1}','scalars')
        if not os.path.exists(os.path.join(scalar_path,"scalars.pkl")):continue
        with open(os.path.join(scalar_path,"scalars.pkl"), 'rb') as f:
            dicts=pickle.load(f)
        bestepoch=dicts['best_epoch'] #获取最佳epoch序号
        result=dicts['ts_result']
        ls_result.append(result)
       # print(result)
        print('best_epoch:',bestepoch)
        print('ts_result:',result[bestepoch])
        
    ls_result=np.vstack(ls_result)
    ls_result=ls_result.mean(axis=0)
    draw_ts(ls_result,'ts_result',folder)

def K_fold_mean_test(K):
      
    save_dir=os.path.join(cfg.SAVE_ROOT,'K-Fold')
    assert os.path.exists(save_dir)
    class_values=np.array([0.,1.5,7.,15.,30.])
    preds=[]
    NWP=None
    obs=None
    fname=None
    for j in range(K):
        folder=os.path.join(save_dir,f'fold_{j+1}','scalars')
        if not os.path.exists(folder):continue
        if not os.path.exists(os.path.join(folder,'val_pred1.pkl')):
            # print('Prediction ',fn,' not exists')
            continue
        with open(os.path.join(folder,'val_pred1.pkl'),'rb') as f:
            data_dic=pickle.load(f)
        if NWP is None:
            NWP=data_dic['NWP']
            obs=data_dic['obs']
            fname=data_dic['name']
        preds.append(data_dic['preds'])
        #if len(preds)==0:continue
    preds=np.array(preds).squeeze()
    # mask=np.ones_like(preds)
    # mask[np.isnan(preds)]=np.nan
    # mask=mask.mean(axis=0)
    if preds.ndim>3:
        thresholds = [0.0] + list(cfg.RAIN.THRESHOLDS)
        # print(thresholds)
        preds=torch.from_numpy(preds)
        count_idx=[]
        for thre in range(len(thresholds)):
            count_idx.append(torch.sum(preds==thre,dim=0).unsqueeze(0))
        count_idx=torch.cat(count_idx,dim=0)
        _,preds=torch.max(count_idx,dim=0)  
        # ts=evaluate(preds.to(cfg.DEVICE),torch.from_numpy(obs).to(cfg.DEVICE))
        ts=evaluate(preds.to(cfg.DEVICE),torch.from_numpy(obs).to(cfg.DEVICE))
        print('final_ts:',ts)
        preds=preds.numpy()
    
    preds=class_values[preds]
    
    
    dicts={}
    dicts['fname']=fname
    dicts['preds']=preds
    dicts['NWP']=NWP
    dicts['obs']=obs

    with open(os.path.join(save_dir,'test_pred.pkl'),'wb') as f:
        pickle.dump(dicts,f,pickle.HIGHEST_PROTOCOL)
    return dicts   


if __name__=="__main__":
    # data=np.random.randn(20,12,4,45,45)
    # d1=custom_crop(data,(10,15),(5,5))
    # print(d1.shape)
    # flist=getnamelist()
    # print(flist)
    #test()
    # predict_K_fold(5,validate=True)
    # evaluate_EC()
    # predict_K_fold(10,validate=False)
    # K_fold_mean(10,sname='Pred_precipitation')
    
    # evaluate_all(10)
    #K_fold_mean(10)
    # test()
    # predict_single()
    # with open(os.path.join(cfg.SAVE_ROOT,'scalars','val_pred.pkl'),'rb') as f:
    #     dicts=pickle.load(f)
    # img_save_dir=os.path.join(cfg.SAVE_ROOT,'test_out')
    # if not os.path.exists(img_save_dir):os.mkdir(img_save_dir)
    # show_tp(dicts,img_save_dir)
    # test()
    predict_K_fold(5,validate=True)
    dicts=K_fold_mean_test(5)#对预测值融合
    # img_save_dir=os.path.join(cfg.SAVE_ROOT,'test_out3')
    # if not os.path.exists(img_save_dir):os.mkdir(img_save_dir)
    # show_tp(dicts,img_save_dir)
    # make_targz(os.path.join(cfg.SAVE_ROOT,'test_out3.tar.gz'), os.path.join(cfg.SAVE_ROOT,'test_out3'))
    # print(os.path.join(cfg.SAVE_ROOT,'test_out3.tar.gz')+' created!')

    # evaluate_EC()

    # save_dir=os.path.join(cfg.SAVE_ROOT,'K-Fold')
    # with open(os.path.join(save_dir,'test_pred.pkl'),'rb') as f:
    #     dicts=pickle.load(f)
    # img_save_dir=os.path.join(cfg.SAVE_ROOT,'test_out4')
    # if not os.path.exists(img_save_dir):os.mkdir(img_save_dir)
    # data_to_dat(dicts,img_save_dir)