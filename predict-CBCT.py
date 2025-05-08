
import warp
import numpy as np
import time
from pathlib import Path
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import model
import transform
import utils
import os
import torchvision.transforms as transforms

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#%%
'''dataset loading'''
torch.backends.cudnn.benchmark = True
roottest = r'/yourroot' #put segCADnpy,segnpy,vessel77npy into the same folder
img_size = [256, 256, 128]
modelCAD = model.snetCAD(img_size=img_size).cuda()
modelORG = model.snetORG(img_size=img_size).cuda()
modelENH = model.snetENH(img_size=img_size).cuda()


TRE_ENH, TRE_ORG, TRE_CAD, TRE_final = [], [], [], [10]
TRE_ENHbefore, TRE_ORGbefore, TRE_CADbefore, TRE_finalbefore = [], [], [], [10]

def Get_Jac(displacement):
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    D = D1 - D2 + D3
    return D

timelist = []
jaclist = []
ssim_m_r_list = []
psnr_m_r_list = []
ssim_w_r_list = []
psnr_w_r_list = []
val_index, lamb, it = 8, 10, 1
WEIGHTS_PATH = './weights-adam/'
weights_fnameCAD = 'weights-CE.pth'
weights_fnameORG = 'weights-OG.pth'
weights_fnameENH = 'weights-VE.pth'
for case in range(1, 29):
    Transform = transforms.Compose([transform.ToTensor()])
    pathCAD = roottest + '/segCADnpy/4DLung/'
    mov_fnameCAD = 'case%02d_T00.npy' % case
    ref_fnameCAD = 'case%02d_T50.npy' % case
    movCAD = np.load(pathCAD + mov_fnameCAD)
    refCAD = np.load(pathCAD + ref_fnameCAD)
    movCAD = np.expand_dims(movCAD, 0)  # shape(1, D, H, W)
    refCAD = np.expand_dims(refCAD, 0)
    mov0CAD = Transform(movCAD)
    ref0CAD = Transform(refCAD)
    movCAD = mov0CAD.unsqueeze(0).cuda()
    refCAD = ref0CAD.unsqueeze(0).cuda()

    pathORG = roottest + '/segnpy/4DLung/'
    mov_fnameORG = 'case%02d_T00.npy' % case
    ref_fnameORG = 'case%02d_T50.npy' % case
    movORG = np.load(pathORG + mov_fnameORG)
    refORG = np.load(pathORG + ref_fnameORG)
    movORG = np.expand_dims(movORG, 0)  # shape(1, D, H, W)
    refORG = np.expand_dims(refORG, 0)
    mov0ORG = Transform(movORG)
    ref0ORG = Transform(refORG)
    movORG = mov0ORG.unsqueeze(0).cuda()
    refORG = ref0ORG.unsqueeze(0).cuda()

    pathENH = roottest + '/vessel77npy/4DLung/'
    mov_fnameENH = 'case%02d_T00.npy' % case
    ref_fnameENH = 'case%02d_T50.npy' % case
    movENH = np.load(pathENH + mov_fnameENH)
    refENH = np.load(pathENH + ref_fnameENH)
    movENH = np.expand_dims(movENH, 0)  # shape(1, D, H, W)
    refENH = np.expand_dims(refENH, 0)
    mov0ENH = Transform(movENH)
    ref0ENH = Transform(refENH)
    movENH = mov0ENH.unsqueeze(0).cuda()
    refENH = ref0ENH.unsqueeze(0).cuda()

    startEpochCAD = utils.load_weights(modelCAD, WEIGHTS_PATH + weights_fnameCAD)
    startEpochORG = utils.load_weights(modelORG, WEIGHTS_PATH + weights_fnameORG)
    startEpochENH = utils.load_weights(modelENH, WEIGHTS_PATH + weights_fnameENH)


    flow = torch.zeros([1, 3] + img_size).cuda()
    modelCAD.eval()
    modelORG.eval()
    modelENH.eval()
    warper = warp.SpatialTransformer(img_size).cuda()

    with torch.no_grad():
        warpedCAD_ORG, warpedCAD, flowCAD = modelCAD(movCAD, refCAD, movORG)
        warpedORG_ENH, warpedORG, flowORG = modelORG(warpedCAD_ORG, refORG, flowCAD, movENH)
        _, flowENH = modelENH(warpedORG_ENH, refENH, flowORG, movENH)
        warped = warper(movORG, flowENH)
        flow += flowENH


    m = movORG.data.cpu().numpy()[0, 0]
    r = refORG.data.cpu().numpy()[0, 0]
    w = warped.data.cpu().numpy()[0, 0]
    ssim_m_r = ssim(m, r, data_range=r.max() - r.min())
    psnr_m_r = psnr(m, r, data_range=r.max() - r.min())

    # 计算 w 和 r 之间的 SSIM 和 PSNR
    ssim_w_r = ssim(w, r, data_range=r.max() - r.min())
    psnr_w_r = psnr(w, r, data_range=r.max() - r.min())

    print(f"SSIM between m and r: {ssim_m_r:.4f}")
    print(f"PSNR between m and r: {psnr_m_r:.4f} dB")
    print(f"SSIM between w and r: {ssim_w_r:.4f}")
    print(f"PSNR between w and r: {psnr_w_r:.4f} dB")

    # 将结果添加到列表
    ssim_m_r_list.append(ssim_m_r)
    psnr_m_r_list.append(psnr_m_r)
    ssim_w_r_list.append(ssim_w_r)
    psnr_w_r_list.append(psnr_w_r)

    flow = flow.data.cpu()

    origin = (0.0, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # Save m, r, w as .mha files
    # case_str = 'case%02d' % case
    # output_path = Path('CBCT/')
    # output_path.mkdir(parents=True, exist_ok=True)
    #
    #
    # def save_as_mha(array, filename):
    #     img = sitk.GetImageFromArray(array)
    #     img.SetOrigin(origin)
    #     img.SetDirection(direction)
    #     sitk.WriteImage(img, str(output_path / filename))
    #
    #
    # def save_as_mha2(array, filename):
    #     img = sitk.GetImageFromArray(array)
    #     img.SetOrigin(origin)
    #     img.SetDirection(direction)
    #     sitk.WriteImage(img, str(output_path / filename))
    #
    #
    # transposed_array = np.transpose(flow.squeeze(0).numpy(), (1, 2, 3, 0))
    # save_as_mha(m, f'{case_str}_T00_inhao.mha')
    # save_as_mha(r, f'{case_str}_T50_inhao.mha')
    # save_as_mha(w, f'{case_str}_warped.mha')

    diff1 = torch.from_numpy(w)
    diff0 = torch.from_numpy(m)
    RESULTS_PATH = 'results/'
    flow_folder = RESULTS_PATH + 'flow/'
    Path(flow_folder).mkdir(exist_ok=True)
    flow_fname = 'flow_val%g_plus_case%g_lamb%g_reso2_noseg.npy' % (val_index, case, lamb)

    npy_data = np.transpose(flow.numpy(), (0, 2, 3, 4, 1))
    jac = Get_Jac(npy_data)
    negative_elements_count = np.sum(jac < 0)
    jaclist.append(negative_elements_count)
print("jac", np.mean(jaclist),"±",np.std(jaclist))

# 计算所有案例的平均值
average_ssim_m_r = np.mean(ssim_m_r_list)
average_psnr_m_r = np.mean(psnr_m_r_list)
average_ssim_w_r = np.mean(ssim_w_r_list)
average_psnr_w_r = np.mean(psnr_w_r_list)

std_ssim_m_r = np.std(ssim_m_r_list)
std_psnr_m_r = np.std(psnr_m_r_list)
std_ssim_w_r = np.std(ssim_w_r_list)
std_psnr_w_r = np.std(psnr_w_r_list)

print(f"Average SSIM between m and r: {average_ssim_m_r:.2f} ±{std_ssim_m_r:.2f}")
print(f"Average PSNR between m and r: {average_psnr_m_r:.2f} ±{std_psnr_m_r:.2f}dB")
print(f"Average SSIM between w and r: {average_ssim_w_r:.2f} ±{std_ssim_w_r:.2f}")
print(f"Average PSNR between w and r: {average_psnr_w_r:.2f} ±{std_psnr_w_r:.2f}dB")
# print("Time", np.mean(timelist), "±", np.std(timelist))
print(ssim_w_r_list,psnr_w_r_list)


