import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NCC(nn.Module):
    def __init__(self, windows_size=11):
        super().__init__()

        self.dim = 3
        self.num_stab_const = 1e-4 
        
        self.windows_size = windows_size
        
        self.pad = windows_size//2
        self.window_volume = windows_size**self.dim

        self.conv = F.conv3d
    
    def forward(self, I, J):
        try:
            I_sum = self.conv(I, self.sum_filter, padding = self.pad)
        except:
            self.sum_filter = torch.ones([1, 1] + [self.windows_size, ]*self.dim, dtype = I.dtype, device = I.device)
            I_sum = self.conv(I, self.sum_filter, padding = self.pad)

        J_sum = self.conv(J, self.sum_filter, padding = self.pad) # (n, 1, h, w) or (n, 1, d, h, w)
        I2_sum = self.conv(I*I, self.sum_filter, padding = self.pad)
        J2_sum = self.conv(J*J, self.sum_filter, padding = self.pad)
        IJ_sum = self.conv(I*J, self.sum_filter, padding = self.pad)

        cross = torch.clamp(IJ_sum - I_sum*J_sum/self.window_volume, min = self.num_stab_const)
        I_var = torch.clamp(I2_sum - I_sum**2/self.window_volume, min = self.num_stab_const)
        J_var = torch.clamp(J2_sum - J_sum**2/self.window_volume, min = self.num_stab_const)

        cc = cross/((I_var*J_var)**0.5)
        
        return -torch.mean(cc) + 1
class LCC(nn.Module):
    
    def __init__(self, win=None, eps=1e-5):
        super(LCC, self).__init__()
        self.win = win
        self.eps = eps
        
    def forward(self, I, J):
        ndims = I.ndimension() - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        
        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        I = I.to(torch.double)
        J = J.to(torch.double)
        # compute CC squares
        I2 = I.pow(2)
        J2 = J.pow(2)

        IJ = I*J

        # compute filters
        filters = torch.ones([1, 1, *self.win]).double()#Variable()

        win_size = torch.prod(torch.Tensor(self.win), dtype=torch.float)
        if I.is_cuda:#gpu
            filters = filters.cuda()
            win_size = win_size.cuda()
        padding = self.win[0]//2


        I_sum = conv_fn(I, filters, stride=1, padding=padding)
        J_sum = conv_fn(J, filters, stride=1, padding=padding)
        I2_sum = conv_fn(I2, filters, stride=1, padding=padding)
        J2_sum = conv_fn(J2, filters, stride=1, padding=padding)
        IJ_sum = conv_fn(IJ, filters, stride=1, padding=padding)

        u_I = I_sum/win_size
        u_J = J_sum/win_size
        
        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)#1e-5
        return - torch.mean(cc) + 1
    
class GCC(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self, eps=1e-5):
        super(GCC, self).__init__()
        self.eps = eps
        
    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J
        #average value
        I_ave, J_ave= I.mean(), J.mean()
        I2_ave, J2_ave = I2.mean(), J2.mean()
        IJ_ave = IJ.mean()
        
        cross = IJ_ave - I_ave * J_ave
        I_var = I2_ave - I_ave.pow(2)
        J_var = J2_ave - J_ave.pow(2)

        cc = cross / (I_var.sqrt() * J_var.sqrt() + self.eps)#1e-5

        return -1.0 * cc + 1
    
class Grad(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1'):
        super(Grad, self).__init__()
        self.penalty = penalty

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        return grad

class IDloss(nn.Module):
    """
    loss between affine transformation and identity transf.
    """

    def __init__(self, penalty='l1'):
        super(IDloss, self).__init__()
        self.penalty = penalty
        self.id = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, theta):
        if theta.is_cuda:
                ID = Variable(self.id.cuda())
        else:
            ID = Variable(self.id)
        ID = ID.repeat(theta.size(0), 1).view(theta.shape)
        if self.penalty == 'l1':
            loss = torch.mean(torch.abs(theta - ID))
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            loss = torch.mean(torch.pow(theta - ID, 2))
        return loss
