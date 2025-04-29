import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

cmap = plt.cm.seismic

class CustomNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=False):
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        norm_value = np.ma.masked_where((value < 0), value)
        return np.ma.masked_array(cmap((norm_value - self.vmin) / (self.vmax - self.vmin)))


def Get_Jac(displacement):

    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    D = D1 - D2 + D3

    return D


def show_sample_slices(sample, name, index, cmap, vmin, vmax):
    fig, ax = plt.subplots()
    background = np.ones_like(sample) * -1
    norm = CustomNormalize(vmin=vmin, vmax=vmax)
    masked_sample = np.ma.masked_where((sample < 0), sample)
    ax.imshow(np.rot90(background, k=-1), cmap='gray') 
    ax.imshow(np.rot90(masked_sample, k=-1), cmap=cmap, norm=norm) 
    ax.axis('off')
    plt.savefig(f'{name}_{index}.png') 
    plt.show()


def Jac_pic():
    npy_data = np.load('4DCT/flow_val8_plus_case8_lamb10_reso2_noseg.npy')
    npy_data = np.transpose(npy_data, (0, 2, 3, 4, 1))
    jac = Get_Jac(npy_data)
    show_sample_slices(jac[0, :, 141, :], 'Coronal_jac', 141, cmap=cmap, vmin=0, vmax=2)

Jac_pic()
