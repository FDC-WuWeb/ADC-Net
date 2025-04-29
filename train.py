import time
import argparse
from pathlib import Path
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import dataset
import model
import transform
import utils
import loss_others
import loss
from itertools import chain

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Training codes')
parser.add_argument('-v', '--val', default=8, type=int,
                    help='the case index of validation')
parser.add_argument('-b', '--batch', default=1, type=int,
                    help='batch size')
parser.add_argument('-l', '--lr', default=0.0001, type=float,
                    help='learning rate')
parser.add_argument('-lD', '--lrD', default=0.00005, type=float,
                    help='learning rateD')
parser.add_argument('-e', '--epoch', default=500, type=int,
                    help='training epochs')
parser.add_argument('-d', '--lamb', default=0.2, type=float,
                    help='lambda, balance the losses.')
parser.add_argument('-w', '--win', default=[5,5,5], type=int,
                    help='window size, in the LCC loss')
parser.add_argument('-i', '--image', default=[256, 256, 128], type=int,
                    help='image size')
parser.add_argument('-p', '--pretrained_model', default=True, type=bool,
                    help='pretrained model')

args = parser.parse_args()
en = 0
att = 0
WEIGHTS_MODEL_PATH = './weights-adam/'
WEIGHTS_NAMECAD = 'weights-CE.pth'
WEIGHTS_NAMEORG = 'weights-OG.pth'
WEIGHTS_NAMEENH = 'weights-VE.pth'
WEIGHTS_PATH = 'weights-adam/'
Path(WEIGHTS_PATH).mkdir(exist_ok=True)
LOSSES_PATH = 'losses/'
Path(LOSSES_PATH).mkdir(exist_ok=True)
RESULTS_PATH = 'results/creatis_flow/'
'''log file'''
f = open(WEIGHTS_PATH + 'README.txt', 'w')
root = r'root'
root2 = './dirlab_32x/'
roottest = r'roottest'
Transform = transforms.Compose([transform.ToTensor()])
train_dset = dataset.Volumes(root,root2, roottest, args.val, train=True, transform=Transform)
val_dset = dataset.Volumes(root,root2, roottest, args.val, train=False, transform=Transform)
train_loader = data.DataLoader(train_dset, args.batch, shuffle=False)
val_loader = data.DataLoader(val_dset, args.batch, shuffle=False)

print("Train dset: %d" %len(train_loader))
print("Val dset: %d" %len(val_loader))

img_size=args.image

'''Train'''
modelCAD = model.snetCAD(img_size=args.image).cuda()
modelORG = model.snetORG(img_size=args.image).cuda()
modelENH = model.snetENH(img_size=args.image).cuda()

optimizer = optim.Adam(params=chain(modelCAD.parameters(),modelORG.parameters(),modelENH.parameters()), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95,
                                                       patience=3, verbose=False, threshold=0.0001,
                                                       threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)
if args.pretrained_model:
    startEpochCAD = utils.load_weights(modelCAD, WEIGHTS_PATH + WEIGHTS_NAMECAD)
    startEpochORG = utils.load_weights(modelORG, WEIGHTS_PATH + WEIGHTS_NAMEORG)
    startEpochENH = utils.load_weights(modelENH, WEIGHTS_PATH + WEIGHTS_NAMEENH)
val = args.val

'''lcc + grad'''
criterion = {'lcc': loss.LCC(args.win).cuda(),
             'ncc': loss.NCC(5).cuda(),
             'mse': torch.nn.MSELoss().cuda(),
             'Grad l1': loss.Grad(penalty='l1').cuda(),
             'Grad l2': loss.Grad(penalty='l2').cuda(),
             'lambda': args.lamb,
             'adver': torch.nn.BCEWithLogitsLoss().cuda(),
             'd1':loss_others.d1_smooth().cuda(),
             'd2':loss_others.d2_smooth().cuda(),
             'diff':loss_others.Diff_loss().cuda()}

for epoch in range(1, args.epoch+1):
    since = time.time()
    ### Train ###
    trn_loss, trn_similarity = utils.train(modelCAD, modelORG, modelENH, modelD, train_loader, optimizer, criterion)
    ### Val ###
    val_loss, val_similarity = utils.val(modelCAD, modelORG, modelENH, val_loader, criterion, epoch)
    scheduler.step(val_loss)

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    ### Checkpoint ###
    utils.save_weightsCAD(en, att, modelCAD, optimizer, val, epoch, val_loss, val_similarity, WEIGHTS_PATH)
    utils.save_weightsORG(en, att, modelORG, optimizer, val, epoch, val_loss, val_similarity, WEIGHTS_PATH)
    utils.save_weightsENH(en, att, modelENH, optimizer, val, epoch, val_loss, val_similarity, WEIGHTS_PATH)

f.close()
