import os
import numpy as np
from config import cfg
import torch

'''
    params:
    pred：预报值
    obsvr：观测值
    k：降水等级,0,1,2,3,4,5,6
'''
def get_N(pred,obsvr): #获取各类型数目
    # pred.ravel()
    # obsvr.ravel()
    
    # if pred.sum()==0 and obsvr.sum()==0:
    #     return None
    temp=pred-obsvr #三种值，0：预报正确，1：空报，-1：漏报
    NA=torch.nansum(pred*obsvr) #命中TP
    NB=torch.nansum((temp==1.))#空报FN
    NC=torch.nansum((temp==-1.))#漏报FP
    ND=torch.nansum(((1-pred)*(1-obsvr)))#无降水预报正确TN
    return [NA,NB,NC,ND]
    
def rain_K(rain,threshold=cfg.RAIN.THRESHOLDS): #将数值转换成等级
    threshold=[0.0]+threshold
    # cp=rain.copy()
    # cp[cp<threshold[0]]=0
    cp=torch.zeros_like(rain)
    for i,thre in enumerate(threshold):
        cp[rain>=thre]=i
    # for i in range(len(threshold)-1):
    #     cp[(cp>=threshold[i])&(cp<threshold[i+1])]=float(i+1)
    # cp[cp>=threshold[-1]]=float(len(threshold))
    return cp

'''
    PO:漏报率
    FAR:空报率
    B:预报偏差
    ETS:公平技巧评分(Equitable Threat Score, ETS)用于衡量对流尺度集合预报的预报效果。
    ETS评分表示在预报区域内满足某降水阈值的降水预报结果相对于满足同样降水阈值的随机预报的预报技巧
'''
def get_Indexs(NA,NB,NC,ND): #计算各评价指标
    if NA+NB+NC==0 :
        TS=torch.tensor(np.nan).to(NA.device)
        ETS=torch.tensor(np.nan).to(NA.device)
    else: 
        TS=NA/(NA+NB+NC)
        R=(NA+NB)*(NA+NC)/(NA+NB+NC+ND)
        ETS=(NA-R)/(NA+NB+NC-R)
    if NA+NC==0 : 
        POD=torch.tensor(np.nan).to(NA.device)
        B=torch.tensor(np.nan).to(NA.device)
    else: 
        POD=NA/(NA+NC)
        B=(NA+NB)/(NA+NC)
    if (NA+NB)!=0:FAR=NB/(NA+NB)
    else: FAR=torch.tensor(np.nan).to(NA.device)
    return [TS.item(),POD.item(),FAR.item(),B.item(),ETS.item()]

def get_TS(NA,NB,NC): #计算各评价指标
    S=torch.tensor(NA+NB+NC)
    TS=S.detach()
    #TS=np.zeros_like(NA)
    # TS=TS*S
    TS[S==0]=np.nan
    TS[S>0]=NA[S>0]/S[S>0]
    return TS

def get_PC(pred,obsvr): #晴雨准确率
    p=np.where(pred>=0.1,1.,0)
    o=np.where(obsvr>=0.1,1.,0)
    NS=get_N(p,o)
    PC=(NS[0]+NS[3])/sum(NS)
    return PC*100


def evaluate(pred,obsvr,test=False,threshold=cfg.RAIN.THRESHOLDS):
    obsvr[obsvr<0.1]=0
    mask=torch.ones_like(obsvr)
    mask[torch.isnan(obsvr)]=np.nan
    
    #pred[pred<0.1]=0
    # rm=rmse(pred,obsvr)
    # ma=mae(pred,obsvr)
    ls=[]
    indexs=[]
    if test: pred=rain_K(pred,threshold)
    obsvr=rain_K(obsvr,threshold)
    for i in range(len(threshold)):
        p=torch.where(pred>=i+1,1.,0.)
        o=torch.where(obsvr>=i+1,1.,0.)
        o=o*mask
        NS=get_N(p,o)
        NN=get_TS(NS[0],NS[1],NS[2])
        print(f'ts:{i}',f'命中{NS[0].item()}：',NS[0].item()/(NS[0].item()+NS[2].item()),
            f'空报{NS[1].item()}',NS[1].item()/(NS[0].item()+NS[1].item()),
            f'漏报{NS[2].item()}',NS[2].item()/(NS[0].item()+NS[2].item()))
        ls.append(NN.cpu().numpy())
        Nindex=get_Indexs(NS[0],NS[1],NS[2],NS[3])
        indexs.append(np.array(Nindex))
        print(f'ts:{i}',f'TS:{Nindex[0]}',
            f'POD:{Nindex[1]}',
            f'FAR:{Nindex[2]}',
            f'BIAS:{Nindex[3]}',
            f'ETS:{Nindex[4]}',)
    ls=np.array(ls)
    return ls

def rmse(pred,obsvr):
    return np.sqrt((np.square(pred-obsvr)).sum()/pred.size)

def mae(pred,obsvr):
    return (np.abs(pred-obsvr)).sum()/pred.size

if __name__=="__main__":
    
    x=np.random.random(size=(64,12,64,64))*100
    y=np.random.random(size=(64,12,64,64))*150
    x[...,0]=0
    y[...,0]=0
    ls=evaluate(x,y)
    print(ls.shape)
    print(ls)

