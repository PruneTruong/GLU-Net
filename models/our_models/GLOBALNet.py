import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
from models.feature_backbones.VGG_features import VGGPyramid
from .mod import CMDTop
from models.our_models.mod import OpticalFlowEstimator, FeatureL2Norm, \
    CorrelationVolume, deconv, conv, predict_flow, unnormalise_and_convert_mapping_to_flow
import torch.nn.functional as F
os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory 
try:
	from models.correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python

import numpy as np


class GLOBALNet_model(nn.Module):
    '''
    GLOBAL-Net model
    '''
    def __init__(self, evaluation, div=1.0, refinement=True, batch_norm=True, pyramid_type='VGG'):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warping

        """
        super(GLOBALNet_model, self).__init__()

        self.div = div
        self.refinement = refinement
        self.pyramid_type = pyramid_type
        if pyramid_type == 'VGG':
            nbr_features = [512, 512, 512, 256, 128, 64, 64]
        else:
            # this is PWC-Net backbone
            nbr_features = [196, 128, 96, 64, 32, 16, 3]

        self.leakyRELU = nn.LeakyReLU(0.1)
        self.corr = CorrelationVolume()
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()

        dd = np.cumsum([128,128,96,64,32])
        # weights for decoder at different levels

        nd = 16*16 # global correlation
        od = nd + 2
        self.decoder4 = CMDTop(in_channels=od, bn=batch_norm)
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        nd = nbr_features[-4]*2 # concatenating the features
        od = nd + 2
        self.decoder3 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        nd = nbr_features[-3]*2 # concatenating the features
        od = nd + 4
        self.decoder2 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 

        # weights for refinement module
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        if pyramid_type == 'VGG':
            self.pyramid = VGGPyramid()
        else:
            raise ValueError("No other back-bone implemented, please choose VGG")

        self.evaluation = evaluation

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        if float(torch.__version__[:3]) >= 1.3:
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        else:
            output = nn.functional.grid_sample(x, vgrid)
        return output

    def forward(self, im_target, im_source, w_original=256, h_original=256):
        # im_target is target image ==> corresponds to all the indices 1,
        # im_source is source image ==> corresponds to all the indices 2

        b, _, h_original, w_original = im_target.size()
        div=self.div
        if self.pyramid_type == 'VGG':
            im1_pyr = self.pyramid(im_target)
            im2_pyr = self.pyramid(im_source)
            c14=im1_pyr[-3]
            c24=im2_pyr[-3]
            c13=im1_pyr[-4]
            c23=im2_pyr[-4]
            c12=im1_pyr[-5]
            c22=im2_pyr[-5]
        else:
            raise ValueError("No other back-bone implemented, please choose VGG")

        # 16x16
        corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))
        corr4 = self.l2norm(F.relu(corr4))
        b, c, h, w = corr4.size()
        ratio_x = w / float(w_original)
        ratio_y = h / float(h_original)
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)

        # conversion to flow
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4) / self.div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        up_flow4 = self.deconv4(flow4)

        # 32x32
        ratio_x = up_flow4.shape[3] / float(w_original)
        ratio_y = up_flow4.shape[2] / float(h_original)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = self.warp(c23, up_flow_4_warping)
        # concatenate features now
        concat3 = torch.cat((c13, warp3, up_flow4), 1)
        x3, res_flow3 = self.decoder3(concat3)
        flow3 = res_flow3 + up_flow4
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x3)

        # 64x64
        ratio_x = up_flow3.shape[3] / float(w_original)
        ratio_y = up_flow3.shape[2] / float(h_original)
        up_flow_3_warping = up_flow3 * div
        up_flow_3_warping[:, 0, :, :] *= ratio_x
        up_flow_3_warping[:, 1, :, :] *= ratio_y
        warp2 = self.warp(c22, up_flow_3_warping)
        concat2 = torch.cat((c12, warp2, up_flow3, up_feat3), 1)
        x, res_flow2 = self.decoder2(concat2)
        flow2 = res_flow2 + up_flow3

        if self.refinement:
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
            flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.evaluation:
            return flow2
        else:
            return [flow4, flow3, flow2]



