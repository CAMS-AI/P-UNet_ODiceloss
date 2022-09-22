import sys
sys.path.append("..") 
from ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch


__C = edict()
cfg = __C
#__C.ROOT_DATA=os.path.abspath(os.path.join(os.path.dirname(__file__), '..','data_train')) #原始数据目录下面分为train和test，请将训练数据放置于train下，test数据放置于test下，文件结构与下发的保持一致
__C.ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','datas'))#训练数据存放位置，根目录
__C.ROOT_DATA='/root/YXX/NP_UNET/datas/train' #原始nc数据存放位置
__C.GLOBAL = edict()
__C.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
__C.BATCH_SIZE = 64
__C.NUM_EPOCHS = 30
__C.NUM_WORKERS = 4
__C.PIN_MEMORY = True
__C.Y_NAME = 'rain' #变量名
__C.MNAME = 'Unet_grid_dice_OR_pixle' #模型名


if not os.path.exists(os.path.join(__C.ROOT_PATH,'output')): os.mkdir(os.path.join(__C.ROOT_PATH,'output'))
SAVE_ROOT=os.path.join(__C.ROOT_PATH,'output',__C.MNAME)
if not os.path.exists(SAVE_ROOT): os.mkdir(SAVE_ROOT)
SAVE_ROOT=os.path.join(SAVE_ROOT,__C.Y_NAME)
if not os.path.exists(SAVE_ROOT): os.mkdir(SAVE_ROOT)

__C.SAVE_ROOT = SAVE_ROOT #输出存储目录
# __C.TEST_STATION_DIR = '/root/YXX/Racing/datas/test'
# __C.TRAIN_OLD=os.path.join(__C.ROOT_DATA,'processed/train/')
# __C.TEST_OLD=os.path.join(__C.ROOT_DATA,'processed/test/')

__C.DATA_DIR = os.path.join(__C.ROOT_PATH,"train") #训练数据目录
__C.MEAN_STD_DIR = os.path.join(__C.ROOT_PATH,'m') #参数数据目录
# __C.MEAN_STD_DIR = '/root/YXX/Racing_Swin/datas/m'
__C.STATION_LIST=os.path.join(__C.MEAN_STD_DIR,'stationlist_64.txt')
# __C.STATION_OLD=os.path.join(__C.MEAN_STD_DIR,'stationlist.txt')
# __C.TEST_DIR = '/root/YXX/Racing_Swin/datas/test'
__C.TEST_DIR = os.path.join(__C.ROOT_PATH,"test")#测试数据目录
__C.KFOLD_LIST=os.path.join(__C.MEAN_STD_DIR,'data_list_rain.pkl')#k折验证文件列表
__C.RAIN = edict()
__C.RAIN.THRESHOLDS = [0.1, 3, 10, 20]#降水阈值
__C.BALANCING_WEIGHTS = [ 2, 5, 10, 20]#对应权重
# __C.ALPHA =np.array([0.75,0.3,0.25,0.25,0.25])


#选择变量列表
__C.SELECTED_VALS=['tmp2m', 'dt2m','MSLP', 'windu10', 'windv10', 'tp', 'tcw', 'cape', 't500', 't700',
         't850', 't925', 'gh500', 'gh700', 'gh850', 'gh925', 'u500', 'v500', 'u700', 'v700', 'u850',
         'v850', 'u925', 'v925',  'q500', 'q700', 'q850', 'q925',]

#总变量列表
__C.VAL_NAMES=['tmp2m', 'dt2m', 'MSLP', 'windu10', 'windv10', 'tp', 'tcw', 'cape', 't500', 't700',
         't850', 't925', 'gh500', 'gh700', 'gh850', 'gh925', 'u500', 'v500', 'u700', 'v700', 'u850',
         'v850', 'u925', 'v925',  'q500', 'q700', 'q850', 'q925','height']

__C.MODEL = edict()
__C.MODEL.IN_CHANNEL=len(__C.SELECTED_VALS) #输入通道数
__C.MODEL.OUT_CHANNEL=len(__C.RAIN.THRESHOLDS) #输出通道数
# __C.MODEL.WINDOWSIZE=8 #


