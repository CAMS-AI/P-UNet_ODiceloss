"""
@Time    : 2021-12-28 15:00
@Author  : Xiaoxiong
@File    : train_and test.py
"""
import shutil
import sys
sys.path.insert(0, '../')
import torch
import numpy as np
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle
from dataset import get_loaders,get_idx
# import random
from config import cfg
# from predict import test_predict
from utils import get_mean_std,create_net
from evaluate_rain import evaluate
# from model.loss import BCELoss,DiceLoss
from Lr_earlystop import EarlyStopping,LRScheduler
torch.manual_seed(666)

def train_run(val_list,train_list,model,
                optimizer, criterion,lr_scheduler,
                scalars_save_dir=None,
                model_save_dir=None,
                predict_save_dir=None,
                vals_idx=None,validate=False,k=None,
                max_epochs=cfg.NUM_EPOCHS,
                iter_show=497,epoch_show=1,
                device=cfg.DEVICE):
    '''
        网络训练与验证
        val_list:验证集
        train_list:训练集
        model,optimizer, criterion,lr_scheduler:模型
        scalars_save_dir:过程参数存储目录,
        model_save_dir:模型pth保存目录
        vals_idx:参与计算变量的对应下标
        predict_save_dir:预报输出存储目录
    '''      
    y_mean_std,transform=get_mean_std(cfg.Y_NAME,vals_idx)
    train_loader = get_loaders(
        cfg.DATA_DIR,train_list,y_name=cfg.Y_NAME,vals_idx=vals_idx,batch_size=cfg.BATCH_SIZE,
        transform=transform,num_works=cfg.NUM_WORKERS,pin_memory=cfg.PIN_MEMORY,shuffle=True)
    val_loader = get_loaders(
        cfg.DATA_DIR,val_list,y_name=cfg.Y_NAME,vals_idx=vals_idx,batch_size=2*cfg.BATCH_SIZE,
        transform=transform,num_works=cfg.NUM_WORKERS,pin_memory=cfg.PIN_MEMORY,shuffle=False)
    # earlystop=EarlyStopping(patience=10, min_delta=0)
    # lr_scheduler=LRScheduler(optimizer, patience=2, min_lr=1e-6, factor=0.5)
    #开始训练
    scaler = torch.cuda.amp.GradScaler()#加快运算
    
    ls_result=[]
    ls_train_loss=[]
    ls_val_loss=[]
    train_loss = 0.0
    bestepoch=0
    best_ts=0 
    POD,FAR=[],[]
    LNS=[]
    for epoch in range(cfg.NUM_EPOCHS):    #cfg.NUM_EPOCHS
        if k is not None:print(f"Fold_{k+1}")
        print(f"{cfg.MNAME}=> epoch:",epoch)
        loop = tqdm(train_loader)
        count=0
        train_loss = 0.0
        model.train()
        for batch_idx, (data, targets) in enumerate(loop):            
            data = data.squeeze().to(device=cfg.DEVICE)     
            targets = targets.squeeze().to(device=cfg.DEVICE) 
            with torch.cuda.amp.autocast():
                output = model(data)
                loss=criterion(output,targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            #梯度截断
            #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())
            train_loss+=loss.item()
            count+=1  
        lr_scheduler.step()        
        train_loss=train_loss/count
        ls_train_loss.append(train_loss)
        print("train_loss:",train_loss)
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            valid_time = 0
            ls_preds=[]
            ls_label=[]
            for valid_data, valid_label in val_loader:
                valid_data = valid_data.squeeze().to(device=cfg.DEVICE)
                valid_label = valid_label.squeeze().to(device=cfg.DEVICE) 
                
                output = model(valid_data)#BCL
                loss=criterion(output,valid_label)
                output=torch.sigmoid(output)
                preds = torch.sum((output > 0.5), dim=1)
                # _,preds=torch.max(output.data,dim=1)
                # preds=torch.sum(preds,dim=1)
                valid_loss += loss.item()               
                valid_label[valid_label<=-9999]=np.nan
                ls_preds.append(preds)
                ls_label.append(torch.squeeze(valid_label))
                valid_time+=1
            valid_loss = valid_loss/valid_time
            ls_preds=torch.cat(ls_preds,dim=0)
            ls_label=torch.cat(ls_label,dim=0)
            result,lns=evaluate(ls_preds,ls_label)
            pod,far=[],[]
            for n in range(lns.shape[0]):
                p=lns[n,0]/(lns[n,0]+lns[n,2])
                f=lns[n,1]/(lns[n,0]+lns[n,1])
                pod.append(p)
                far.append(f)
            POD.append(pod)
            FAR.append(far)
            LNS.append(np.array(lns))
            ts_val=result[0]+1*result[1]+1*result[2]+1*result[3]
            # lr_scheduler(1/ts_val)
            # earlystop(1/ts_val)
            # if earlystop.early_stop:
            #     break
            if best_ts<ts_val:
                best_ts=ts_val
                bestepoch=epoch
                #保存当前epoch模型       
                if model_save_dir is not None:
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    else:
                        shutil.rmtree(model_save_dir)
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f'{cfg.Y_NAME}_{epoch}.pth'))
            print("ts_0.1:",result[0],"ts_3:",result[1],"ts_10:",result[2],"ts_20:",result[3])
            print("valid_loss:",valid_loss)
            ls_result.append(result)
            ls_val_loss.append(valid_loss)

        #保存各类参数,save parameters
        if scalars_save_dir is not None:
            dicts={"train_loss":np.array(ls_train_loss),"valid_loss":np.array(ls_val_loss),"ts_result":np.array(ls_result),
                    "val_list":val_list,"vals_idx":vals_idx,'best_epoch':bestepoch,'pod':POD,'far':FAR,'lns':LNS}
            with open(os.path.join(scalars_save_dir,"scalars.pkl"), 'wb') as f:
                pickle.dump(dicts, f, pickle.HIGHEST_PROTOCOL) #保存文件
                # print("保存：",os.path.join(scalars_save_dir,"scalars.pkl"))


def train_and_test(vals_idx=None,device=cfg.DEVICE,
                    iter_show=497,epoch_show=1,):
    '''
        调用train_run开始进行一次训练
    '''  
    model, optimizer, criterion, lr_scheduler=create_net(device=device)
    data_list=np.array(os.listdir(os.path.join(cfg.DATA_DIR,"x")))
    data_list=data_list[data_list>'2016010100_00.npy']
    ## 载入文件list并打乱，验证集均衡分布
    # random.shuffle(data_list)
    # fs=len(data_list)//5
    val_list=data_list[(data_list>'2020010100_00.npy')&(data_list<'2021010100_00.npy')]
    # val_list=data_list[(data_list>'2019010100_00.npy')&(data_list<'2020010100_00.npy')]
    train_list=data_list[(data_list<'2020010100_00.npy')]
    # train_list=data_list[(data_list<'2019010100_00.npy')|((data_list>'2020010100_00.npy')&(data_list<'2021010100_00.npy'))]

    model_save_dir = os.path.join(cfg.SAVE_ROOT, 'pth')
    if not os.path.exists(model_save_dir):os.mkdir(model_save_dir)
    scalars_save_dir=os.path.join(cfg.SAVE_ROOT,"scalars")
    if not os.path.exists(scalars_save_dir):os.mkdir(scalars_save_dir)
    predict_save_dir=cfg.SAVE_ROOT
    if not os.path.exists(predict_save_dir):os.mkdir(predict_save_dir)

    train_run(val_list,train_list,model,
                optimizer, criterion,lr_scheduler,
                scalars_save_dir=scalars_save_dir,
                model_save_dir=model_save_dir,
                predict_save_dir=predict_save_dir,
                vals_idx=vals_idx,
                device=device)
    print('训练完毕！')


def K_fold_val(K=5,
                vals_idx=None,
                max_epochs=cfg.NUM_EPOCHS,
                iter_show=497,
                epoch_show=1,
                device=cfg.DEVICE
                ): 
    '''
        K折交叉验证
    '''      
    save_dir=os.path.join(cfg.SAVE_ROOT,'K-Fold')
    if not os.path.exists(save_dir):os.mkdir(save_dir)
    ## 载入文件list并打乱，分成K份，存储为K对训练、验证集，后期可以直接载入使用，pkl文件
    if not os.path.exists(cfg.KFOLD_LIST):
        data_list=os.listdir(os.path.join(cfg.DATA_DIR,"x"))
        data_list=np.array(data_list)
        data_list=data_list[(data_list>'2016010100_00.npy')&(data_list<'2021010100_00.npy')]
        
        data_list=list(data_list)
        dls=[]
        for fp in data_list:
            dls.append(fp.split('_')[0])
        dls=list(set(dls))

        data_list=np.array(dls)
        data_list.sort()
        data_list=list(data_list)
        # random.shuffle(data_list)
        fs=len(data_list)//K
        v_ls,tr_ls=[],[]
        for i in range(K):
            if i+1<K:
                val_list=data_list[i*fs:(i+1)*fs]
                train_list=data_list[(i+1)*fs:]
                if i>0 : train_list=data_list[:i*fs]+train_list
            else: 
                val_list=data_list[i*fs:]
                train_list=data_list[:i*fs]
            v_ls.append(val_list)
            tr_ls.append(train_list)
        dicts={
            'val_list':v_ls,
            'train_list':tr_ls
        }
        with open(cfg.KFOLD_LIST,'wb') as f:
            pickle.dump(dicts,f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(cfg.KFOLD_LIST,'rb') as f:
            dicts=pickle.load(f)
        v_ls=dicts['val_list']
        tr_ls=dicts['train_list'] 

    #开始K折交叉验证训练
    for i in range(K):
        print('K-Fold:',i+1)
        # val_list=v_ls[i]
        # train_list=tr_ls[i]
        val_list=[]
        train_list=[]
        for tn in tr_ls[i]:
            for k in range(12):
                t_name=tn+'_'+str(k+1).zfill(2)+'.npy'
                if not os.path.exists(os.path.join(cfg.DATA_DIR,'x',t_name)):continue
                train_list.append(t_name)
        for vn in v_ls[i]:
            for k in range(12):
                v_name=vn+'_'+str(k+1).zfill(2)+'.npy'
                if not os.path.exists(os.path.join(cfg.DATA_DIR,'x',v_name)):continue
                val_list.append(v_name)
        train_list=np.array(train_list)
        val_list=np.array(val_list)
        if not os.path.exists(os.path.join(save_dir, f'fold_{i+1}')):os.mkdir(os.path.join(save_dir, f'fold_{i+1}'))
        model_save_dir = os.path.join(save_dir, f'fold_{i+1}','pth')
        if not os.path.exists(model_save_dir):os.mkdir(model_save_dir)
        scalars_save_dir=os.path.join(save_dir,f'fold_{i+1}',"scalars")
        if not os.path.exists(scalars_save_dir):os.mkdir(scalars_save_dir)
        predict_save_dir=os.path.join(save_dir,f'fold_{i+1}')
        if not os.path.exists(predict_save_dir):os.mkdir(predict_save_dir)

        model, optimizer, criterion, lr_scheduler=None,None,None,None
        model, optimizer, criterion, lr_scheduler=create_net(device=device)
        train_run(val_list,train_list,model,
                optimizer, criterion,lr_scheduler,
                scalars_save_dir=scalars_save_dir,
                model_save_dir=model_save_dir,
                predict_save_dir=predict_save_dir,
                vals_idx=vals_idx,k=i,max_epochs=max_epochs,
                device=device)
    print('K折交叉验证完毕!')



if __name__=="__main__":
    #获取输入变量的下标
    vals_idx=get_idx(cfg.SELECTED_VALS,cfg.VAL_NAMES)
    #  #十折训练
    # train_and_test(vals_idx)
    K_fold_val(5,vals_idx)

    #开始预测，十个模型预测后取平均
    # from predict import predict_K_fold,K_fold_mean
    # predict_K_fold(10,validate=False)
    # K_fold_mean(10,sname='Pred_temperature')