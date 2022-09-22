from torch import nn
import torch
#from torch._C import long
#from nowcasting.config import cfg
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
sys.path.append("..") 
from config import cfg

class DiceLoss(nn.Module):
    def __init__(self, n_classes,weight=None,thresholds=cfg.RAIN.THRESHOLDS):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self._thresholds=thresholds
        self._weights=None

    def transfer_to_OR(self,target):
        target=target.unsqueeze(1)
        OR_layers=[]
        for thre in self._thresholds:
            OR_layers.append((target>=thre).float())
        OR_layers=torch.cat(OR_layers,dim=1)
        return OR_layers

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(1,self.n_classes+1):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target,  softmax=True):
        weight=self._weights
        if softmax:
            inputs = torch.sigmoid(inputs)
        class_index = torch.zeros_like(target).long()
        thresholds = [0.0]+ list(self._thresholds)
        # print(thresholds)
        for i, threshold in enumerate(thresholds):
            class_index[target >= threshold] = i
        # target = self._one_hot_encoder(class_index)
        target=self.transfer_to_OR(target)
        if weight is None:
            weight = [1] * self.n_classes
        # else: weight=weight[class_index]
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class BCELoss(torch.nn.Module):
    def __init__(self, threshold=0.0, weight=None):
        super(BCELoss, self).__init__()
        self.threshold=threshold
        self.elipson = 0.000001
        self.weight=weight

    def forward(self, predict, target):
        threshold=self.threshold
        if threshold==0.0 :labels=torch.where(target==0.0,1,0)
        else:labels=torch.where(target>=threshold,1,0)  
        mask=torch.ones_like(target)
        mask[target<-9999]=0
        pt = torch.sigmoid(predict) # sigmoide获取概率

        loss=-(self.weight*(pt+self.elipson).log()*labels+(1-labels)*(1-pt+self.elipson).log())
   
        loss =(loss*mask).sum()/(mask>0).sum()
        return loss

class Masked_mae(nn.Module):
    def __init__(self,  NORMAL_LOSS_GLOBAL_SCALE=0.01, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE


    def forward(self, input, target,mask=None): 
        mae=torch.sum(mask.float() * (torch.abs(input-target)))/mask.float().sum()
        return mae


class WeightedCrossEntropyLoss(nn.Module):

    # weight should be a 1D Tensor assigning weight to each of the classes.
    def __init__(self, thresholds=cfg.RAIN.THRESHOLDS, weight=None):
        super().__init__()
        # 每个类别的权重
        self._weight = weight

        # thresholds: 雷达反射率
        self._thresholds = thresholds

    # input: output prob, BCL
    # target: B*L original data, range [0, 1]
    # mask: BL
    def forward(self, input, target, mask):
        #input BSCHW
        # F.cross_entropy should be B*C*L
        # B*S*H*W
       # target = target.squeeze(1)
        class_index = torch.zeros_like(target).long()
        thresholds = [0.0] + list(self._thresholds)
        # print(thresholds)
        for i, threshold in enumerate(thresholds):
            class_index[target >= threshold] = i
        #error = F.cross_entropy(input, class_index,  reduction='none')
        error = F.cross_entropy(input, class_index, self._weight, reduction='none')
        # S*B*1*H*W
        #error = error.unsqueeze(2)
        return torch.sum(error*mask.float())/(mask>0).sum()



