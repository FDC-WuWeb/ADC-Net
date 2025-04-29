import torch
import torch.nn as nn
import torch.nn.functional as F
import warp
import ViP3d
class conv_down(nn.Module):
    def __init__(self, inChan, outChan, down=True, pool_kernel=2):
        super(conv_down, self).__init__()
        self.down = down
        self.conv = nn.Sequential(
            nn.Conv3d(inChan, outChan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(outChan),
            nn.LeakyReLU(0.2),
            nn.Conv3d(outChan, outChan, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(outChan),
            nn.LeakyReLU(0.2)
        )
        self.pool = nn.AvgPool3d(pool_kernel)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        if self.down:
            x = self.pool(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class gateconv(nn.Module):
    def __init__(self, inChan, outChan, down=True, pool_kernel=2):
        super(gateconv, self).__init__()
        self.down = down
        self.conv = nn.Sequential(
            nn.Conv3d(inChan, outChan, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x

class NetCAD(nn.Module):
    def __init__(self,  nfea=[2,16,32,64,64,64,128,64,32,3]):
        super(NetCAD, self).__init__()

        self.down1 = conv_down(nfea[0], nfea[1])
        self.down2 = conv_down(nfea[1], nfea[2])
        self.down3 = conv_down(nfea[2], nfea[3])

        self.same1 = conv_down(nfea[3], nfea[4], down=False)
        self.same2 = conv_down(nfea[4], nfea[5], down=False)
        self.same3 = conv_down(nfea[5], nfea[6], down=False)
        self.same4 = conv_down(nfea[6], nfea[7], down=False)

        self.channelAttention1 = SELayer(nfea[1])
        self.channelAttention2 = SELayer(nfea[2])
        self.channelAttention3 = SELayer(nfea[3])

        self.channelAttention4 = SELayer(nfea[4])
        self.channelAttention5 = SELayer(nfea[5])
        self.channelAttention6 = SELayer(nfea[6])
        self.channelAttention7 = SELayer(nfea[7])

        self.ViP4 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)
        self.ViP5 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)
        self.ViP6 = ViP3d.WeightedPermuteMLP(128,8,8,4,seg_dim=4,res=True)
        self.ViP7 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)

        self.outconv = nn.Conv3d(
                64, nfea[9], kernel_size=1, stride=1, padding=0, bias=True)

        self.gateconv1 = gateconv(128,3)
        self.gateconv2 = gateconv(3,3)
        self.gateconv3 = gateconv(3,3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, x):

        x = self.down1(x)
        x = self.channelAttention1(x)

        x = self.down2(x)
        x = self.channelAttention2(x)

        x = self.down3(x)
        x = self.channelAttention3(x)

        x = self.same1(x)
        x = self.ViP4(x)
        x = self.channelAttention4(x)

        x = self.same2(x)
        x = self.ViP5(x)
        x = self.channelAttention5(x)

        x = self.same3(x)
        x = self.ViP6(x)
        x = self.channelAttention6(x)
        x_ra = x

        x = self.same4(x)
        x = self.ViP7(x)
        x = self.channelAttention7(x)

        x = self.outconv(x)
        x_flow = x
        x_ra = self.gateconv1(x_ra)
        x = self.gateconv2(x)
        x = self.relu(x + x_ra)
        x = self.gateconv3(x)
        x = self.sigmoid(x)
        x = x * x_flow
        x = F.interpolate(x,scale_factor=8,mode="trilinear", align_corners= True)
        return x

class NetORG(nn.Module):
    def __init__(self,  nfea=[2,16,32,64,64,64,128,64,32,3]):
        super(NetORG, self).__init__()

        self.down1 = conv_down(nfea[0], nfea[1])
        self.down2 = conv_down(nfea[1], nfea[2])
        self.down3 = conv_down(nfea[2], nfea[3])

        self.same1 = conv_down(nfea[3], nfea[4], down=False)
        self.same2 = conv_down(nfea[4], nfea[5], down=False)
        self.same3 = conv_down(nfea[5], nfea[6], down=False)
        self.same4 = conv_down(nfea[6], nfea[7], down=False)

        self.channelAttention1 = SELayer(nfea[1])
        self.channelAttention2 = SELayer(nfea[2])
        self.channelAttention3 = SELayer(nfea[3])

        self.channelAttention4 = SELayer(nfea[4])
        self.channelAttention5 = SELayer(nfea[5])
        self.channelAttention6 = SELayer(nfea[6])
        self.channelAttention7 = SELayer(nfea[7])

        self.ViP4 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)
        self.ViP5 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)
        self.ViP6 = ViP3d.WeightedPermuteMLP(128,8,8,4,seg_dim=4,res=True)
        self.ViP7 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)

        self.outconv = nn.Conv3d(
                64, nfea[9], kernel_size=1, stride=1, padding=0, bias=True)

        self.gateconv1 = gateconv(128,3)
        self.gateconv2 = gateconv(3,3)
        self.gateconv3 = gateconv(3,3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, x):

        x = self.down1(x)
        x = self.channelAttention1(x)

        x = self.down2(x)
        x = self.channelAttention2(x)

        x = self.down3(x)
        x = self.channelAttention3(x)

        x = self.same1(x)
        x = self.ViP4(x)
        x = self.channelAttention4(x)

        x = self.same2(x)
        x = self.ViP5(x)
        x = self.channelAttention5(x)

        x = self.same3(x)
        x = self.ViP6(x)
        x = self.channelAttention6(x)
        x_ra = x

        x = self.same4(x)
        x = self.ViP7(x)
        x = self.channelAttention7(x)

        x = self.outconv(x)
        x_flow = x
        x_ra = self.gateconv1(x_ra)
        x = self.gateconv2(x)
        x = self.relu(x + x_ra)
        x = self.gateconv3(x)
        x = self.sigmoid(x)
        x = x * x_flow
        x = F.interpolate(x,scale_factor=8,mode="trilinear", align_corners= True)
        return x


class NetENH(nn.Module):
    def __init__(self,  nfea=[2,16,32,64,64,64,128,64,32,3]):

        self.down1 = conv_down(nfea[0], nfea[1])
        self.down2 = conv_down(nfea[1], nfea[2])
        self.down3 = conv_down(nfea[2], nfea[3])

        self.same1 = conv_down(nfea[3], nfea[4], down=False)
        self.same2 = conv_down(nfea[4], nfea[5], down=False)
        self.same3 = conv_down(nfea[5], nfea[6], down=False)
        self.same4 = conv_down(nfea[6], nfea[7], down=False)

        self.channelAttention1 = SELayer(nfea[1])
        self.channelAttention2 = SELayer(nfea[2])
        self.channelAttention3 = SELayer(nfea[3])

        self.channelAttention4 = SELayer(nfea[4])
        self.channelAttention5 = SELayer(nfea[5])
        self.channelAttention6 = SELayer(nfea[6])
        self.channelAttention7 = SELayer(nfea[7])

        self.ViP4 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)
        self.ViP5 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)
        self.ViP6 = ViP3d.WeightedPermuteMLP(128,8,8,4,seg_dim=4,res=True)
        self.ViP7 = ViP3d.WeightedPermuteMLP(64,8,8,4,seg_dim=4,res=True)

        self.outconv = nn.Conv3d(
                64, nfea[9], kernel_size=1, stride=1, padding=0, bias=True)

        self.gateconv1 = gateconv(128,3)
        self.gateconv2 = gateconv(3,3)
        self.gateconv3 = gateconv(3,3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, x):

        x = self.down1(x)
        x = self.channelAttention1(x)

        x = self.down2(x)
        x = self.channelAttention2(x)

        x = self.down3(x)
        x = self.channelAttention3(x)

        x = self.same1(x)
        x = self.ViP4(x)
        x = self.channelAttention4(x)

        x = self.same2(x)
        x = self.ViP5(x)
        x = self.channelAttention5(x)

        x = self.same3(x)
        x = self.ViP6(x)
        x = self.channelAttention6(x)
        x_ra = x

        x = self.same4(x)
        x = self.ViP7(x)
        x = self.channelAttention7(x)

        x = self.outconv(x)
        x_flow = x
        x_ra = self.gateconv1(x_ra)
        x = self.gateconv2(x)
        x = self.relu(x + x_ra)
        x = self.gateconv3(x)
        x = self.sigmoid(x)
        x = x * x_flow
        x = F.interpolate(x,scale_factor=8,mode="trilinear", align_corners= True)
        return x

class snetCAD(nn.Module):
    def __init__(self,  img_size=[256,256,128]):
        super(snetCAD, self).__init__()
        self.net = NetCAD()
        self.warper = warp.SpatialTransformer(img_size)

    def forward(self, movCAD, refCAD, movORG):
        input0 = torch.cat((movCAD, refCAD), 1)
        flowCAD = self.net(input0)
        warpedCAD = self.warper(movCAD,flowCAD) #uncomment for training
        warpedCAD_ORG = self.warper(movORG, flowCAD)
        return warpedCAD_ORG, warpedCAD, flowCAD  #uncomment for training
        # return warpedCAD_ORG, flowCAD   #uncomment for predict

class snetORG(nn.Module):
    def __init__(self,  img_size=[256,256,128]):
        super(snetORG, self).__init__()
        self.net = NetORG()
        self.warper = warp.SpatialTransformer(img_size)

    def forward(self, warpedCAD_ORG, refORG, flowCAD, movENH):
        input0 = torch.cat((warpedCAD_ORG, refORG), 1)
        flowres_ORG = self.net(input0)
        flowORG = flowCAD + flowres_ORG
        warpORG = self.warper(warpedCAD_ORG,flowORG) #uncomment for training
        warpedORG_ENH = self.warper(movENH, flowORG)
        return warpedORG_ENH, warpORG, flowORG #uncomment for training
        # return warpedORG_ENH, flowORG #uncomment for predict

class snetENH(nn.Module):
    def __init__(self,  img_size=[256,256,128]):
        super(snetENH, self).__init__()
        self.net = NetENH()
        self.warper = warp.SpatialTransformer(img_size)

    def forward(self, warpedORG_ENH, refENH, flowORG, movENH):
        input0 = torch.cat((warpedORG_ENH, refENH), 1)
        flowres_ENH = self.net(input0)
        flowENH = flowORG + flowres_ENH
        warpedENH = self.warper(movENH, flowENH)
        return warpedENH, flowENH

