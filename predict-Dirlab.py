
import numpy as np
import time
from pathlib import Path
# import SimpleITK as sitk
import torch

import model
import transform
import utils
import os
import torchvision.transforms as transforms
import warp

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''dataset loading'''
torch.backends.cudnn.benchmark = True
roottest = r'/yourroot' # put segCADnpy,segnpy,vessel77npy into the same folder
img_size = [256, 256, 128]
modelCAD = model.snetCAD(img_size=img_size).cuda()
modelORG = model.snetORG(img_size=img_size).cuda()
modelENH = model.snetENH(img_size=img_size).cuda()


TRE_ENH, TRE_ORG, TRE_CAD, TRE_final = [], [], [], [10]
TRE_ENHbefore, TRE_ORGbefore, TRE_CADbefore, TRE_finalbefore = [], [], [], [10]
timelist = []
ssim_m_r_list = []
psnr_m_r_list = []
ssim_w_r_list = []
psnr_w_r_list = []
val_index, lamb, it = 8, 10, 1
WEIGHTS_PATH = './weights-adam/'
weights_fnameCAD = 'weights-CE.pth'
weights_fnameORG = 'weights-OG.pth'
weights_fnameENH = 'weights-VE.pth'
for case in range(1, 11):
    Transform = transforms.Compose([transform.ToTensor()])
    pathCAD = roottest + '/segCADnpy/Dirlab/'
    mov_fnameCAD = 'case%g_T00.npy' % case
    ref_fnameCAD = 'case%g_T50.npy' % case
    movCAD = np.load(pathCAD + mov_fnameCAD)
    refCAD = np.load(pathCAD + ref_fnameCAD)
    movCAD = np.expand_dims(movCAD, 0)  # shape(1, D, H, W)
    refCAD = np.expand_dims(refCAD, 0)
    mov0CAD = Transform(movCAD)
    ref0CAD = Transform(refCAD)
    movCAD = mov0CAD.unsqueeze(0).cuda()
    refCAD = ref0CAD.unsqueeze(0).cuda()

    pathORG = roottest + '/segnpy/Dirlab/'
    mov_fnameORG = 'case%g_T00.npy' % case
    ref_fnameORG = 'case%g_T50.npy' % case
    movORG = np.load(pathORG + mov_fnameORG)
    refORG = np.load(pathORG + ref_fnameORG)
    movORG = np.expand_dims(movORG, 0)  # shape(1, D, H, W)
    refORG = np.expand_dims(refORG, 0)
    mov0ORG = Transform(movORG)
    ref0ORG = Transform(refORG)
    movORG = mov0ORG.unsqueeze(0).cuda()
    refORG = ref0ORG.unsqueeze(0).cuda()

    pathENH = roottest + '/vessel77npy/Dirlab/'
    mov_fnameENH = 'case%g_T00.npy' % case
    ref_fnameENH = 'case%g_T50.npy' % case
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

    warper = warp.SpatialTransformer(img_size).cuda()
    flow = torch.zeros([1, 3] + img_size).cuda()
    modelCAD.eval()
    modelORG.eval()
    modelENH.eval()

    with torch.no_grad():
        warpedCAD_ORG, warpedCAD, flowCAD = modelCAD(movCAD, refCAD, movORG)
        warpedORG_ENH, warpedORG, flowORG = modelORG(warpedCAD_ORG, refORG, flowCAD, movENH)
        # warpedCAD_ORG, flowCAD = modelCAD(movCAD, refCAD, movORG)
        # warpedORG_ENH, flowORG = modelORG(warpedCAD_ORG, refORG, flowCAD, movENH)
        _, flowENH = modelENH(warpedORG_ENH, refENH, flowORG, movENH)

        warped = warper(movORG, flowENH)
        flow += flowENH

    m = movORG.data.cpu().numpy()[0, 0]
    r = refORG.data.cpu().numpy()[0, 0]
    w = warped.data.cpu().numpy()[0, 0]

    # 计算 m 和 r 之间的 SSIM 和 PSNR
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

    # # Save m, r, w as .mha files
    # case_str = 'case%02d' % case
    # output_path = Path('4DCT/')
    # output_path.mkdir(parents=True, exist_ok=True)
    #
    # def save_as_mha(array, filename):
    #     img = sitk.GetImageFromArray(array)
    #     img.SetOrigin(origin)
    #     img.SetDirection(direction)
    #     sitk.WriteImage(img, str(output_path / filename))
    #
    # def save_as_mha2(array, filename):
    #     img = sitk.GetImageFromArray(array)
    #     img.SetOrigin(origin)
    #     img.SetDirection(direction)
    #     sitk.WriteImage(img, str(output_path / filename))
    #
    #
    # transposed_array = np.transpose(flow.squeeze(0).numpy(), (1, 2, 3, 0))
    # print(transposed_array.shape)
    # input()
    # save_as_mha(m, f'{case_str}_T00_inhao.mha')
    # save_as_mha(r, f'{case_str}_T50_inhao.mha')
    # save_as_mha(w, f'{case_str}_warped.mha')
    # save_as_mha2(transposed_array, f'{case_str}_flow.mha')

    # if case == 8:
    #     visual.view_diff(m, r, w)
    diff1 = torch.from_numpy(w)
    diff0 = torch.from_numpy(m)
    RESULTS_PATH = 'results/'
    flow_folder = RESULTS_PATH + 'flow/'
    Path(flow_folder).mkdir(exist_ok=True)
    flow_fname = 'flow_val%g_plus_case%g_lamb%g_reso2_noseg.npy' % (val_index, case, lamb)
    np.save(flow_folder + flow_fname, flow.numpy())
print("Time",np.mean(timelist),"±",np.std(timelist))

# 计算所有案例的平均值
average_ssim_m_r = np.mean(ssim_m_r_list)
average_psnr_m_r = np.mean(psnr_m_r_list)
average_ssim_w_r = np.mean(ssim_w_r_list)
average_psnr_w_r = np.mean(psnr_w_r_list)

std_ssim_m_r = np.std(ssim_m_r_list)
std_psnr_m_r = np.std(psnr_m_r_list)
std_ssim_w_r = np.std(ssim_w_r_list)
std_psnr_w_r = np.std(psnr_w_r_list)

print(f"Average SSIM between m and r: {average_ssim_m_r:.2f}±{std_ssim_m_r:.2f}")
print(f"Average PSNR between m and r: {average_psnr_m_r:.2f}±{std_psnr_m_r:.2f}dB")
print(f"Average SSIM between w and r: {average_ssim_w_r:.2f}±{std_ssim_w_r:.2f}")
print(f"Average PSNR between w and r: {average_psnr_w_r:.2f}±{std_psnr_w_r:.2f}dB")
# print("Time", np.mean(timelist), "±", np.std(timelist))

print(ssim_w_r_list)
print(psnr_w_r_list)

# TRE#############################################################

mean_tot, std_tot = 0, 0
tre_all = {}
for case in range(1, 11):
    val, lamb = 8, 10
    spacing_arr = [[0.97, 0.97, 2.5], [1.16, 1.16, 2.5], [1.15, 1.15, 2.5], [1.13, 1.13, 2.5], [1.10, 1.10, 2.5],
                   [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5], [0.97, 0.97, 2.5]]
    spacing = np.array(spacing_arr[case - 1])
    lmk_path = './mark/Case%gPack/ExtremePhases/' % case  # 路径
    mov_lmk_fname = 'Case%g_300_T00_xyz.txt' % case  # 实验T00到T50进行配准
    ref_lmk_fname = 'Case%g_300_T50_xyz.txt' % case
    flow_folder = './results/flow/'
    flow_fname = 'flow_val%g_plus_case%g_lamb%g_reso2_noseg.npy' % (val, case, lamb)
    flow = np.load(flow_folder + flow_fname)
    flow = torch.Tensor(flow)
    H, W, D = img_size[0], img_size[1], img_size[2]

    mov_lmk = np.loadtxt(lmk_path + mov_lmk_fname)
    ref_lmk = np.loadtxt(lmk_path + ref_lmk_fname)
    offset_arr = [[4, 40, 2], [1, 17, 3], [0, 37, 1], [0, 36, 0], [0, 48, 0], [112, 135, 13], [103, 127, 15],
                  [100, 70, 15], [126, 127, 2], [135, 106, 0]]
    resize_axis = [[257, 167, 92], [261, 195, 105], [260, 178, 99], [255, 172, 98], [255, 181, 105],
                   [327, 198, 104], [342, 208, 110], [307, 237, 113], [266, 202, 74], [256, 232, 105]]
    a = H / resize_axis[case - 1][0]
    b = W / resize_axis[case - 1][1]
    c = D / resize_axis[case - 1][2]
    resize_factor = np.array([a, b, c])
    mov_lmk0 = np.zeros([300, 3])
    ref_lmk0 = np.zeros([300, 3])

    # offset_arr[case - 1] = [offset_arr[case-1][0], offset_arr[case-1][0], offset_arr[case-1][0]+ kk]
    mov_lmk0[:, 0] = (mov_lmk[:, 0] - offset_arr[case - 1][0] - 1) * resize_factor[0]
    mov_lmk0[:, 1] = (mov_lmk[:, 1] - offset_arr[case - 1][1] - 1) * resize_factor[1]
    mov_lmk0[:, 2] = ((mov_lmk[:, 2] - offset_arr[case - 1][2] - 1) * resize_factor[2])
    ref_lmk0[:, 0] = (ref_lmk[:, 0] - offset_arr[case - 1][0] - 1) * resize_factor[0]
    ref_lmk0[:, 1] = (ref_lmk[:, 1] - offset_arr[case - 1][1] - 1) * resize_factor[1]
    ref_lmk0[:, 2] = ((ref_lmk[:, 2] - offset_arr[case - 1][2] - 1) * resize_factor[2])
    ref_lmk_index = np.round(ref_lmk0).astype('int32')  # 取整
    ref_lmk1 = ref_lmk0.copy()
    ref_lmk_index1 = np.zeros([300, 3])
    ref_lmk_index1 = ref_lmk_index  # 预留网格取整后被忽略的小数
    for i in range(300):
        hi, wi, di = ref_lmk_index[i]
        h0, w0, d0 = flow[0, :, hi, wi, di]
        ref_lmk1[i] += [h0, w0, d0]

    spacing1 = spacing
    spacing1 = spacing / resize_factor  # 转换分辨率
    factor1 = np.ones([300, 3])
    factor1 = ref_lmk_index1 / ref_lmk1
    ref_lmk1_xs = (ref_lmk0 - ref_lmk_index1) * factor1  # 计算网格时被忽略的小数
    diff1 = (ref_lmk0 - mov_lmk0) * spacing1
    diff1 = torch.Tensor(diff1)
    tre1 = diff1.pow(2).sum(1).sqrt()
    mean1 = tre1.mean()
    std1 = tre1.std()
    diff1 = (ref_lmk1 - mov_lmk0 + ref_lmk1_xs) * spacing1
    diff1 = torch.Tensor(diff1)
    tre1 = diff1.pow(2).sum(1).sqrt()
    # print(tre1)
    mean1 = tre1.mean()
    std1 = tre1.std()
    mean_tot += mean1
    std_tot += std1
    TRE_ENH.append(mean1)
    print('ENH case%g' % case, mean1, '    case%g' % case, std1)

mean_tot = mean_tot / 10
std_tot = std_tot / 10
print('ENH mean_tot', mean_tot, '    std_tot', std_tot)

