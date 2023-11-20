import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from src.tools.label_transform import rotation_matrix_2d, pitchyaw_to_vector, vector_to_pitchyaw

# def pitchyaw_to_vector(pitchyaws):
# 	if pitchyaws.dim() == 1:
# 		pitchyaws = pitchyaws.unsqueeze(0)  ## (2,) -->  (1, 2)
# 	sin = torch.sin(pitchyaws)
# 	cos = torch.cos(pitchyaws)
# 	return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1).float()


# def vector_to_pitchyaw(vectors):
# 	'''vector shape [n,3]'''
# 	vectors = torch.div(vectors, torch.norm(vectors, dim=1).unsqueeze(dim=1) )
# 	return torch.stack( [torch.asin(vectors[:, 1]), torch.atan2(vectors[:, 0], vectors[:, 2])], dim=1).float() ## stack [ theta, phi]


# def angular_error_torch(a, b):
#     similarity = torch.nn.CosineSimilarity()(a, b)
#     return torch.div(torch.mul(torch.arccos(similarity), 180.0), np.pi)


def nn_angular_distance(a, b):
    a = a.unsqueeze(0) if a.dim()==1 else a
    b = b.unsqueeze(0) if b.dim()==1 else b
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(sim) * (180 / np.pi)


class GazeLoss(nn.Module):
    def __init__(self, gaze_weight, 
                    loss_type:str,
                    head_weight=1.0,
                ):

        super().__init__()
        self.gaze_weight = gaze_weight
        self.head_weight = head_weight
        assert loss_type in ['l1', 'l2', 'angular']
        self.l1_criterion = nn.L1Loss() 
        self.l2_criterion = nn.MSELoss()
        self.loss_type = loss_type
    
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
            # assert pred.shape[-1] == 3 and label.shape[-1]==2, \
            # f"the prediction should be directly in vector [batch, 3], got pred: {pred.shape}, and label should be in pitchyaw, got label: {label.shape}"
            pred_loss = self.gaze_angular_loss(pred, label)
        else:
            print(f'wrong loss type {self.loss_type}')
            exit(0)
        return pred_loss

    @classmethod
    def gaze_angular_loss(self, pred, label):
        if pred.shape[-1] == 2:
            pred = pitchyaw_to_vector(pred)
        label = pitchyaw_to_vector(label)
        loss = nn_angular_distance(label, pred)
        return loss.mean()


    # def forward(self, pred, label,
    #             global_step, 
    #             optimizer_idx=0,
    #             split="train", 
    #             weights=None):

    #     pred_loss = self.criterion(pred, label)
    #     loss =  self.gaze_weight * pred_loss 

    #     angular_error = angular_error_torch(pred.detach(), label.detach())

    #     log = {"{}/total_loss".format(split): loss.clone().detach(), 
    #             "{}/pred_loss".format(split): pred_loss.detach(),
    #             "{}/angular_error".format(split): angular_error.mean().detach(),
    #             }
    #     return loss, log


if __name__ =="__main__":
    pred_gaze = torch.tensor([[0.1, 0.2],
                                [0.1, 0.5]])

    gaze = torch.tensor([[0.5, 0.5], 
                        [ -0.1, 0.5]])

    # angle_error = nn_batch_angular_distance(gaze, pred_gaze)
    # print( 'sted angular error: ', angle_error)

    previous_error = nn_angular_distance(gaze, pred_gaze)
    print( 'previous_error: ', previous_error)

    # pred_v = pitchyaw_to_vector(pred_gaze)
    # v = pitchyaw_to_vector(gaze)
    # previous_error = nn_angular_distance(pred_v, v)
    # print( 'previous_error: ', previous_error)


