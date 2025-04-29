import torch
import torch.nn as nn

class d1_smooth(nn.Module):
    def __init__(self):
        super(d1_smooth, self).__init__()

    def forward(self,flow):
        image_shape = flow.shape
        d1_flow = torch.zeros((image_shape[0], 3) + tuple(image_shape[1:]), dtype=flow.dtype, device=flow.device)
        d1_flow[:, 2, :, :-1, :, :] = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        d1_flow[:, 1, :, :, :-1, :] = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        d1_flow[:, 0, :, :, :, :-1] = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

        d1_loss = torch.mean(torch.sum(torch.abs(d1_flow), dim=2, keepdims=True))
        return d1_loss

class d2_smooth(nn.Module):
    def __init__(self):
        super(d2_smooth, self).__init__()

    def forward(self,flow):
        image_shape = flow.shape
        d2_flow = torch.zeros((image_shape[0], 3) + tuple(image_shape[1:]), dtype=flow.dtype, device=flow.device)
        d2_flow[:, 2, :, :-2, :, :] = (flow[:, :, 2:, :, :] + flow[:, :, :-2, :, :] - 2 * flow[:, :, 1:-1, :, :])  
        d2_flow[:, 1, :, :, :-2, :] = (flow[:, :, :, 2:, :] + flow[:, :, :, :-2, :] - 2 * flow[:, :, :, 1:-1, :]) 
        d2_flow[:, 0, :, :, :, :-2] = (flow[:, :, :, :, 2:] + flow[:, :, :, :, :-2] - 2 * flow[:, :, :, :, 1:-1]) 

        d2_loss = torch.mean(torch.sum(torch.abs(d2_flow), dim=2, keepdims=True))
        return d2_loss
