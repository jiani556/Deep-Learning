import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    no_of_classes=len(cls_num_list)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / np.array(effective_num)
    # print(weights)
    weights = (weights / np.sum(weights)) * no_of_classes
    per_cls_weights = torch.tensor(weights).float()
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        weights=self.weight
        no_of_classes=weights.shape[0]
        labels_one_hot = F.one_hot(target, no_of_classes).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        # logpt = nn.functional.cross_entropy(input, target, reduction='none',weight=self.weight)
        # pt = torch.exp(-logpt)
        # # compute the loss
        # loss = ((1 - pt) ** self.gamma) * logpt
        # loss = loss.mean()

        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)  # prevents nans when probability 0
        F_loss = weights * (1 - pt) ** self.gamma * ce_loss
        loss=F_loss.mean()

        # preds = input.view(-1, input.size(-1))
        # self.weight = self.weight.to(preds.device)
        # preds_softmax = F.softmax(preds, dim=1)
        # preds_logsoft = torch.log(preds_softmax)
        # preds_softmax = preds_softmax.gather(1, target.view(-1, 1))
        # preds_logsoft = preds_logsoft.gather(1, target.view(-1, 1))
        # self.weight = self.weight.gather(0, target.view(-1))
        # loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        # loss=loss.mean()
        return loss
