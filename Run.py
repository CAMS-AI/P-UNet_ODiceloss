"""
@Time    : 2022-1-17 19:00
@Author  : Xiaoxiong You
@File    : Run.py
"""

from data_preprocess import NC_to_NPY_train,NC_to_NPY_test
from predict import predict_K_fold,K_fold_mean
from train_and_test import K_fold_val
from dataset import get_idx
from config import cfg
import os

def run_train():
    #预处理原始NC文件，并存为npy个格式,dir为原始测试数据目录，s_dir为存储测试数据目录
    NC_to_NPY_train(dir=os.path.join(cfg.ROOT_DATA,'train'),s_dir=cfg.DATA_DIR)
    #获取输入变量的下标
    vals_idx=get_idx(cfg.SELECTED_VALS,cfg.VAL_NAMES)
    #K折训练
    K_fold_val(5,vals_idx)

def run_test():
    #开始预测
    #dir为原始测试数据目录，s_dir为存储测试数据目录
    NC_to_NPY_test(dir=cfg.TEST_STATION_DIR,s_dir=cfg.TEST_DIR)
    #开始预测，十个模型预测后取平均   
    predict_K_fold(5,validate=False)
    K_fold_mean(5,result_dir=cfg.SAVE_ROOT)

if __name__=="__main__":
    run_train()