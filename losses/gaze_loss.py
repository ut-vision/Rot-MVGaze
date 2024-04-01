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

        # self.l1_criterion = nn.L1Loss() 
        # self.l2_criterion = nn.MSELoss()
    # def forward(self, pred, label):
    #     if self.loss_type == 'l1':
    #         assert pred.shape[-1] == 2 and label.shape[-1]==2, \
    #         f"the prediction should be in pitchyaw [batch, 2], got pred: {pred.shape}, and label should be in pitchyaw, got label: {label.shape}"
    #         pred_loss = self.l1_criterion(pred, label)
    #     elif self.loss_type == 'l2':
    #         assert pred.shape[-1] == 2 and label.shape[-1]==2, \
    #         f"the prediction should be in pitchyaw [batch, 2], got pred: {pred.shape}, and label should be in pitchyaw, got label: {label.shape}"
    #         pred_loss = self.l2_criterion(pred, label)
    #     elif self.loss_type == 'angular':
    #         pred_loss = gaze_angular_loss(pred, label)
    #     else:
    #         print(f'wrong loss type {self.loss_type}')
    #         exit(0)
    #     return pred_loss
    # @classmethod
    # def gaze_angular_loss(self, pred, label):
    #     loss = angular_error(label, pred)
    #     return loss.mean()
        

    def forward(self, pred, label):
        if self.loss_type == 'l1':
            assert pred.shape[-1] == 2 and label.shape[-1]==2, \
            f"the prediction should be in pitchyaw [batch, 2], got pred: {pred.shape}, and label should be in pitchyaw, got label: {label.shape}"
            pred_loss = gaze_l1_loss(pred, label)
        elif self.loss_type == 'l2':
            assert pred.shape[-1] == 2 and label.shape[-1]==2, \
            f"the prediction should be in pitchyaw [batch, 2], got pred: {pred.shape}, and label should be in pitchyaw, got label: {label.shape}"
            pred_loss = gaze_l2_loss(pred, label)
			
        elif self.loss_type == 'angular':
            pred_loss = gaze_angular_loss(pred, label)
        else:
            print(f'wrong loss type {self.loss_type}')
            exit(0)
        return pred_loss


    


def nn_angular_distance(a, b):
	sim = F.cosine_similarity(a, b, eps=1e-6)
	sim = F.hardtanh(sim, -1.0, 1.0)
	return torch.acos(sim) * (180 / np.pi)


def gaze_angular_loss( y_hat, y ):
	y = pitchyaw_to_vector(y)
	y_hat = pitchyaw_to_vector(y_hat)
	loss = nn_angular_distance(y, y_hat)
	return torch.mean(loss)



def gaze_l2_loss(y, y_hat):
    loss = torch.abs(y - y_hat) **2   
    loss = torch.mean(loss) 
    return loss 
		
def gaze_l1_loss(y, y_hat):
    loss = torch.abs(y - y_hat) 
    loss = torch.mean(loss)
    return loss 

