import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from utils.math import rotation_matrix_2d, pitchyaw_to_vector, vector_to_pitchyaw, angular_error


class GazeLoss(nn.Module):
    def __init__(self, gaze_weight, 
                    loss_type:str,
                    head_weight=1.0,
                ):

        super().__init__()
        self.gaze_weight = gaze_weight
        self.head_weight = head_weight
        assert loss_type in ['l1', 'l2', 'angular']
        self.loss_type = loss_type

        self.l1_criterion = nn.L1Loss() 
        self.l2_criterion = nn.MSELoss()
        
    
    def forward(self, pred, label):
        if self.loss_type == 'l1':
            assert pred.shape[-1] == 2 and label.shape[-1]==2, \
            f"the prediction should be in pitchyaw [batch, 2], got pred: {pred.shape}, and label should be in pitchyaw, got label: {label.shape}"
            pred_loss = self.l1_criterion(pred, label)
        elif self.loss_type == 'l2':
            assert pred.shape[-1] == 2 and label.shape[-1]==2, \
            f"the prediction should be in pitchyaw [batch, 2], got pred: {pred.shape}, and label should be in pitchyaw, got label: {label.shape}"
            pred_loss = self.l2_criterion(pred, label)
        elif self.loss_type == 'angular':
            pred_loss = self.gaze_angular_loss(pred, label)
        else:
            print(f'wrong loss type {self.loss_type}')
            exit(0)
        return pred_loss

    @classmethod
    def gaze_angular_loss(self, pred, label):
        loss = angular_error(label, pred)
        return loss.mean()


