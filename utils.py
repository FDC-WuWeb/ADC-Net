import os
import shutil
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def save_weightsCAD(en,att,model,optimizer,val, epoch, loss, err, WEIGHTS_PATH):
    if en:weights_fname = 'weights-en-val%g-%d-%.3f-%.3fCAD.pth' % (val,epoch, loss, err)
    elif att:weights_fname = 'weights-val%g_plus-att-%d-%.3f-%.3fCAD.pth' % (val, epoch, loss, err)
    else:weights_fname = 'weights-val%g_plus-%d-%.3f-%.3fCAD.pth' % (val, epoch, loss, err)

    torch.save({
        'startEpoch': epoch,
        'loss': loss,
        'error': err,
        'state_dict': model.state_dict(),
        # 'optimizer':optimizer.state_dict()
    }, './weights-adam/' + weights_fname)
    shutil.copyfile('./weights-adam/'+ weights_fname, WEIGHTS_PATH + 'latestCAD.pth')

def save_weightsORG(en,att,model,optimizer,val, epoch, loss, err, WEIGHTS_PATH):
    if en:weights_fname = 'weights-en-val%g-%d-%.3f-%.3fdown1ORG.pth' % (val,epoch, loss, err)
    elif att:weights_fname = 'weights-val%g_plus-att-%d-%.3f-%.3fORG.pth' % (val, epoch, loss, err)
    else:weights_fname = 'weights-val%g_plus-%d-%.3f-%.3fORG.pth' % (val, epoch, loss, err)

    torch.save({
        'startEpoch': epoch,
        'loss': loss,
        'error': err,
        'state_dict': model.state_dict(),
        # 'optimizer': optimizer.state_dict()
    }, './weights-adam/' + weights_fname)
    shutil.copyfile('./weights-adam/'+ weights_fname, WEIGHTS_PATH + 'latestORG.pth')

def save_weightsENH(en,att,model,optimizer,val, epoch, loss, err, WEIGHTS_PATH):
    if en:weights_fname = 'weights-en-val%g-%d-%.3f-%.3fENH.pth' % (val,epoch, loss, err)
    elif att:weights_fname = 'weights-val%g_plus-att-%d-%.3f-%.3fENH.pth' % (val, epoch, loss, err)
    else:weights_fname = 'weights-val%g_plus-%d-%.3f-%.3fENH.pth' % (val, epoch, loss, err)
    # weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
        'startEpoch': epoch,
        'loss': loss,
        'error': err,
        'state_dict': model.state_dict(),
    }, './weights-adam/' + weights_fname)
    shutil.copyfile('./weights-adam/'+ weights_fname, WEIGHTS_PATH + 'latestENH.pth')

def load_weights(model, fpath):
    weights = torch.load(fpath, map_location='cpu')
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'],strict=False)
    return startEpoch

def adjust_lr(lr, decay, optimizer, cur_epoch, n_epochs):
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train(modelCAD, modelORG, modelENH, modelD, trn_loader, optimizer, criterion):
    modelCAD.train()
    modelORG.train()
    modelENH.train()
    modelD.train()
    a = 0.01
    b = 0.01
    c = 0.98
    m = 1
    trn_loss, trn_ncc, trn_mse = 0, 0, 0
    for i, [movCAD, refCAD, movORG, refORG, movENH, refENH] in enumerate(trn_loader):
        movCAD = movCAD.cuda()
        refCAD = refCAD.cuda()

        movORG = movORG.cuda()
        refORG = refORG.cuda()

        movENH = movENH.cuda()
        refENH = refENH.cuda()

        optimizer.zero_grad()

        warpedCAD_ORG, warpedCAD, flowCAD = modelCAD(movCAD, refCAD, movORG)
        warpedORG_ENH, warpedORG, flowORG = modelORG(warpedCAD_ORG, refORG, flowCAD, movENH)
        warpedENH, flowENH = modelENH(warpedORG_ENH, refENH, flowORG, movENH)

        optimizer.zero_grad()

        loss1CAD = criterion['ncc'](warpedCAD, refCAD)
        loss1ORG = criterion['ncc'](warpedORG, refORG)
        loss1ENH = criterion['ncc'](warpedENH, refENH)
        loss2CAD = criterion['mse'](warpedCAD, refCAD)
        loss2ORG = criterion['mse'](warpedORG, refORG)
        loss2ENH = criterion['mse'](warpedENH, refENH)
        loss3 = criterion['d2'](flowENH)

        loss = a*(loss1CAD+ m*loss2CAD) + b*(loss1ORG+m*loss2ORG) + c*(loss1ENH+m*loss2ENH) + 3*loss3

        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        
        trn_loss += loss.item()
        trn_ncc += (loss1CAD.item()*a + loss1ORG.item()*b + loss1ENH.item()*c)
        trn_mse += (loss2CAD.item()*a + loss2ORG.item()*b + loss1ENH.item()*c)


    trn_loss /= len(trn_loader)
    trn_ncc /= len(trn_loader)
    trn_mse /= len(trn_loader)
    trn_similarity = trn_ncc + trn_mse
    return trn_loss, trn_similarity

def val(modelCAD, modelORG, modelENH,val_loader, criterion, epoch):
    modelCAD.eval()
    modelORG.eval()
    modelENH.eval()
    a = 0.01
    b = 0.01
    c = 0.98
    m = 1
    val_loss, val_lcc, val_mse = 0, 0, 0
    for movCAD, refCAD, movORG, refORG, movENH, refENH in val_loader:
        movCAD = movCAD.cuda()
        refCAD = refCAD.cuda()

        movORG = movORG.cuda()
        refORG = refORG.cuda()

        movENH = movENH.cuda()
        refENH = refENH.cuda()

        with torch.no_grad():
            warpedCAD_ORG, warpedCAD, flowCAD = modelCAD(movCAD, refCAD, movORG)
            warpedORG_ENH, warpedORG, flowORG = modelORG(warpedCAD_ORG, refORG, flowCAD, movENH)
            warpedENH, flowENH = modelENH(warpedORG_ENH, refENH, flowORG, movENH)

            loss1CAD = criterion['ncc'](warpedCAD, refCAD)
            loss1ORG = criterion['ncc'](warpedORG, refORG)
            loss1ENH = criterion['ncc'](warpedENH, refENH)
            loss2CAD = criterion['mse'](warpedCAD, refCAD)
            loss2ORG = criterion['mse'](warpedORG, refORG)
            loss2ENH = criterion['mse'](warpedENH, refENH)

            loss3 = criterion['d2'](flowENH)
            loss = a*(loss1CAD+m*loss2CAD) + b*(loss1ORG+m*loss2ORG) + c*(loss1ENH+m*loss2ENH) + 3*loss3

        torch.cuda.empty_cache()
        
        val_loss += loss.item()
        val_lcc += (loss1CAD.item()*a + loss1ORG.item()*b + loss1ENH.item()*c)
        val_mse += (loss2CAD.item()*a + loss2ORG.item()*b + loss1ENH.item()*c)

    val_loss /= len(val_loader)
    val_lcc /= len(val_loader)
    val_mse /= len(val_loader)
    val_similarity = val_lcc + val_mse

    return val_loss, val_similarity


