本项目为模式降水订正，基于Unet构建，采用diceloss进行训练，此模型在上海气象局举办的第二届人工智能天气预报创新大赛中获得降水赛道冠军。

## 使用说明

本程序包含数据处理、模型训练、验证、预测以及评估

**主程序**train_and_test.py

    vals_idx=get_idx(cfg.SELECTED_VALS,cfg.VAL_NAMES) #挑选变量进行训练
    # train_and_test(vals_idx) #直接开始训练
    K_fold_val(5,vals_idx) #开始K折训练

**数据预处理**内包含了针对nc数据的处理，根据需求自行修改，将处理完的输入数据保存在train下的x文件夹中，实况放在y中

**标准化参数**

> 输入模式数据得参数保存在 x_mean_std.pkl，对应的key为['max','min','mean','std']

> 实况保存在*输出变量名.pkl*中，对应的key为['max','min','mean','std']

**参数设置**config,py

> 主要修改路径，也可对batch_size、阈值等进行调整



**如有疑问请联系作者**

游枭雄，Email:crazytiy@163.com,QQ:58039257


