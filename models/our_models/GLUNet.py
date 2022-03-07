import torch
import torch.nn as nn
import math
import os
import sys
import torch.nn.functional as F
from models.feature_backbones.VGG_features import VGGPyramid
from models.feature_backbones.ResNet_features import ResNetPyramid
from .mod import CMDTop
from models.our_models.mod import OpticalFlowEstimatorNoDenseConnection, OpticalFlowEstimator, FeatureL2Norm, \
    CorrelationVolume, deconv, conv, predict_flow, unnormalise_and_convert_mapping_to_flow, warp
from models.our_models.consensus_network_modules import MutualMatching, NeighConsensus, FeatureCorrelation
from models.correlation import correlation # the custom cost volume layer
import numpy as np
from .bilinear_deconv import BilinearConvTranspose2d


class GLUNet_model(nn.Module):
    '''
    GLU-Net
    '''

    def __init__(self, evaluation, div=1.0, iterative_refinement=False,
                 refinement_at_all_levels=False, refinement_at_adaptive_reso=True,
                 batch_norm=True, pyramid_type='VGG', md=4, upfeat_channels=2, dense_connection=True,
                 consensus_network=False, cyclic_consistency=True, decoder_inputs='corr_flow_feat'):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(GLUNet_model, self).__init__()
        self.div=div
        self.pyramid_type = pyramid_type
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.l2norm = FeatureL2Norm()
        self.iterative_refinement = iterative_refinement #only during evaluation

        # where to put the refinement networks
        self.refinement_at_all_levels = refinement_at_all_levels
        self.refinement_at_adaptive_reso = refinement_at_adaptive_reso

        # definition of the inputs to the decoders
        self.decoder_inputs = decoder_inputs
        self.dense_connection = dense_connection
        self.upfeat_channels = upfeat_channels

        # improvement of the global correlation
        self.cyclic_consistency=cyclic_consistency
        self.consensus_network = consensus_network
        if self.cyclic_consistency:
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
        elif consensus_network:
            ncons_kernel_sizes = [3, 3, 3]
            ncons_channels = [10, 10, 1]
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
            # normalisation is applied in code here
            self.NeighConsensus = NeighConsensus(use_cuda=True,
                                                 kernel_sizes=ncons_kernel_sizes,
                                                 channels=ncons_channels)
        else:
            self.corr = CorrelationVolume()

        dd = np.cumsum([128,128,96,64,32])
        # 16x16
        nd = 16*16 # global correlation
        od = nd + 2
        self.decoder4 = CMDTop(in_channels=od, bn=batch_norm)

        # initialize the deconv to bilinear weights speeds up the training significantly
        self.deconv4 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        # self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # 32x32
        nd = (2*md+1)**2 # constrained correlation, 4 pixels on each side
        if self.decoder_inputs == 'corr_flow_feat':
            od = nd + 2
        elif self.decoder_inputs == 'corr':
            od = nd
        elif self.decoder_inputs == 'corr_flow':
            od = nd + 2
        if dense_connection:
            self.decoder3 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = od + dd[4]
        else:
            self.decoder3 = OpticalFlowEstimatorNoDenseConnection(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = 32

        # weights for refinement module
        if self.refinement_at_all_levels or self.refinement_at_adaptive_reso:
            self.dc_conv1 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
            self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=batch_norm)
            self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=batch_norm)
            self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=batch_norm)
            self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
            self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
            self.dc_conv7 = predict_flow(32)

        # 1/8 of original resolution
        nd = (2*md+1)**2 # constrained correlation, 4 pixels on each side
        if self.decoder_inputs == 'corr_flow_feat':
            od = nd + 2
        elif self.decoder_inputs == 'corr':
            od = nd
        elif self.decoder_inputs == 'corr_flow':
            od = nd + 2
        if dense_connection:
            self.decoder2 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = od + dd[4]
        else:
            self.decoder2 = OpticalFlowEstimatorNoDenseConnection(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = 32
        if self.decoder_inputs == 'corr_flow_feat':
            self.upfeat2 = deconv(input_to_refinement, self.upfeat_channels, kernel_size=4, stride=2, padding=1)

        # initialize the deconv to bilinear weights speeds up the training significantly
        self.deconv2 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        # self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        if refinement_at_all_levels:
            # weights for refinement module
            self.dc_conv1_level2 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
            self.dc_conv2_level2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=batch_norm)
            self.dc_conv3_level2 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=batch_norm)
            self.dc_conv4_level2 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=batch_norm)
            self.dc_conv5_level2 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
            self.dc_conv6_level2 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
            self.dc_conv7_level2 = predict_flow(32)

        # 1/4 of original resolution
        nd = (2*md+1)**2 # constrained correlation, 4 pixels on each side
        if self.decoder_inputs == 'corr_flow_feat':
            od = nd + self.upfeat_channels + 2
        elif self.decoder_inputs == 'corr':
            od = nd
        elif self.decoder_inputs == 'corr_flow':
            od = nd + 2
        if dense_connection:
            self.decoder1 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = od+dd[4]
        else:
            self.decoder1 = OpticalFlowEstimatorNoDenseConnection(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = 32

        self.l_dc_conv1 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.l_dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=batch_norm)
        self.l_dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=batch_norm)
        self.l_dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=batch_norm)
        self.l_dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.l_dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.l_dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        if pyramid_type == 'ResNet':
            self.pyramid = ResNetPyramid()
        else:
            self.pyramid = VGGPyramid()

        self.evaluation=evaluation

    def pre_process_data(self, source_img, target_img, device, apply_flip=False):
        '''

        :param source_img:
        :param target_img:
        :param apply_flip:
        :param device:
        :return:
        '''

        # img has shape bx3xhxw
        b, _, h_original, w_original = target_img.shape
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])

        # original resolution
        if h_original < 256:
            int_preprocessed_height = 256
        else:
            int_preprocessed_height = int(math.floor(int(h_original / 8.0) * 8.0))

        if w_original < 256:
            int_preprocessed_width = 256
        else:
            int_preprocessed_width = int(math.floor(int(w_original / 8.0) * 8.0))

        if apply_flip:
            # if apply flip, horizontally flip the target images
            target_img_original = target_img
            target_img = []
            for i in range(b):
                transformed_image = np.fliplr(target_img_original[i].cpu().permute(1,2,0).numpy())
                target_img.append(transformed_image)

            target_img = torch.from_numpy(np.uint8(target_img)).permute(0, 3, 1, 2)

        source_img_copy = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area').byte()
        target_img_copy = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area').byte()
        source_img_copy = source_img_copy.float().div(255.0)
        target_img_copy = target_img_copy.float().div(255.0)
        mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

        # resolution 256x256
        source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                          size=(256, 256),
                                                          mode='area').byte()
        target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                          size=(256, 256),
                                                          mode='area').byte()

        source_img_256 = source_img_256.float().div(255.0)
        target_img_256 = target_img_256.float().div(255.0)
        source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])

        ratio_x = float(w_original)/float(int_preprocessed_width)
        ratio_y = float(h_original)/float(int_preprocessed_height)

        return source_img_copy.to(device), target_img_copy.to(device), source_img_256.to(device), target_img_256.to(device), \
               ratio_x, ratio_y, h_original, w_original

    def flipping_condition(self, im_source_base, im_target_base, device):

        # should only happen during evaluation
        target_image_is_flipped = False # for training
        if not self.evaluation:
            raise ValueError('Flipping condition should only happen during evaluation')
        else:
            list_average_flow = []
            false_true = [False, True]
            for apply_flipping in false_true:
                im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, h_base, w_base = \
                    self.pre_process_data(im_source_base, im_target_base, apply_flip=apply_flipping, device=device)
                b, _, h_256, w_256 = im_target_256.size()

                with torch.no_grad():
                    # pyramid, 256 reso
                    im1_pyr_256 = self.pyramid(im_target_256)
                    im2_pyr_256 = self.pyramid(im_source_256)
                    c14 = im1_pyr_256[-3]
                    c24 = im2_pyr_256[-3]

                flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
                average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                list_average_flow.append(average_flow.item())
            target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
            if target_image_is_flipped:
                list_average_flow = []
                # if previous way found that target is flipped with respect to the source ==> check that the
                # other way finds the same thing
                # ==> the source becomes the target and the target becomes source
                for apply_flipping in false_true:
                    im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, h_base, w_base = \
                        self.pre_process_data(im_target_base, im_source_base, apply_flip=apply_flipping, device=device)
                    b, _, h_256, w_256 = im_target_256.size()

                    with torch.no_grad():
                        # pyramid, 256 reso
                        im1_pyr_256 = self.pyramid(im_target_256)
                        im2_pyr_256 = self.pyramid(im_source_256)
                        c14 = im1_pyr_256[-3]
                        c24 = im2_pyr_256[-3]

                    flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
                    average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                    list_average_flow.append(average_flow.item())
                target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
                # if the right direction found that it is flipped, either the other direction finds the same,
                # then it is flipped, otherwise it isnt flipped

        # found out if better to flip the target image or not, now pre-process the new source and target images
        self.target_image_is_flipped = target_image_is_flipped
        im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, \
        h_original, w_original = self.pre_process_data(im_source_base, im_target_base,
                                                       apply_flip=target_image_is_flipped, device=device)
        return im_source.to(device).contiguous(), im_target.to(device).contiguous(), \
               im_source_256.to(device).contiguous(), im_target_256.to(device).contiguous(), \
               ratio_x, ratio_y, h_original, w_original

    def coarsest_resolution_flow(self, c14, c24, h_256, w_256):
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)
        b = c24.shape[0]
        if self.cyclic_consistency:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        elif self.consensus_network:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4d = self.NeighConsensus(corr4d)
            corr4d = MutualMatching(corr4d)  # size is [b, 1, hsource, wsource, htarget, wtarget]
            corr4 = corr4d.squeeze(1).view(c24.shape[0], c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        else:
            corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))
        corr4 = self.l2norm(F.relu(corr4))
        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4) / self.div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        return flow4

    def forward(self, im_target, im_source, im_target_256, im_source_256):
        # all indices 1 refer to target images
        # all indices 2 refer to source images

        b, _, h_full, w_full = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()
        div = self.div

        # extract pyramid features
        with torch.no_grad():
            im1_pyr = self.pyramid(im_target, eigth_resolution=True)
            im2_pyr = self.pyramid(im_source, eigth_resolution=True)
            c11 = im1_pyr[-2] # size original_res/4xoriginal_res/4
            c21 = im2_pyr[-2]
            c12 = im1_pyr[-1] # size original_res/8xoriginal_res/8
            c22 = im2_pyr[-1]

            # pyramid, 256 reso
            im1_pyr_256 = self.pyramid(im_target_256)
            im2_pyr_256 = self.pyramid(im_source_256)
            c13 = im1_pyr_256[-4]
            c23 = im2_pyr_256[-4]
            c14 = im1_pyr_256[-3]
            c24 = im2_pyr_256[-3]

        # RESOLUTION 256x256
        # level 16x16
        flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
        up_flow4 = self.deconv4(flow4)

        # level 32x32
        ratio_x = 32.0 / float(w_256)
        ratio_y = 32.0 / float(h_256)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = warp(c23, up_flow_4_warping)
        # constrained correlation now
        corr3 = correlation.FunctionCorrelation(tensorFirst=c13, tensorSecond=warp3)
        corr3 = self.leakyRELU(corr3)
        if self.decoder_inputs == 'corr_flow_feat':
            corr3 = torch.cat((corr3, up_flow4), 1)
        elif self.decoder_inputs == 'corr':
            corr3 = corr3
        elif self.decoder_inputs == 'corr_flow':
            corr3 = torch.cat((corr3, up_flow4), 1)
        x3, res_flow3 = self.decoder3(corr3)
        flow3 = res_flow3 + up_flow4
        # flow 3 refined (at 32x32 resolution)
        if self.refinement_at_adaptive_reso or self.refinement_at_all_levels:
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x3))))
            flow3 = flow3 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.iterative_refinement and self.evaluation:
            # from 32x32 resolution, if upsampling to 1/8*original resolution is too big,
            # do iterative upsampling so that gap is always smaller than 2.
            R_w = float(w_full)/8.0/32.0
            R_h = float(h_full)/8.0/32.0
            if R_w > R_h:
                R = R_w
            else:
                R = R_h

            minimum_ratio = 3.0
            nbr_extra_layers = max(0, int(round(np.log(R/minimum_ratio)/np.log(2))))

            if nbr_extra_layers == 0:
                flow3[:, 0, :, :] *= float(w_full) / float(256)
                flow3[:, 1, :, :] *= float(h_full) / float(256)
                # ==> put the upflow in the range [Horiginal x Woriginal]
            else:
                # adding extra layers
                flow3[:, 0, :, :] *= float(w_full) / float(256)
                flow3[:, 1, :, :] *= float(h_full) / float(256)
                for n in range(nbr_extra_layers):
                    ratio = 1.0 / (8.0 * 2 ** (nbr_extra_layers - n ))
                    up_flow3 = F.interpolate(input=flow3, size=(int(h_full * ratio), int(w_full * ratio)),
                                             mode='bilinear',
                                             align_corners=False)
                    c23_bis = torch.nn.functional.interpolate(c22, size=(int(h_full * ratio), int(w_full * ratio)), mode='area')
                    c13_bis = torch.nn.functional.interpolate(c12, size=(int(h_full * ratio), int(w_full * ratio)), mode='area')
                    warp3 = warp(c23_bis, up_flow3 * div * ratio)
                    corr3 = correlation.FunctionCorrelation(tensorFirst=c13_bis, tensorSecond=warp3)
                    corr3 = self.leakyRELU(corr3)
                    if self.decoder_inputs == 'corr_flow_feat':
                        corr3 = torch.cat((corr3, up_flow3), 1)
                    elif self.decoder_inputs == 'corr':
                        corr3 = corr3
                    elif self.decoder_inputs == 'corr_flow':
                        corr3 = torch.cat((corr3, up_flow3), 1)
                    x, res_flow3 = self.decoder2(corr3)
                    flow3 = res_flow3 + up_flow3

            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_full / 8.0), int(w_full / 8.0)), mode='bilinear',
                                     align_corners=False)
        else:
            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_full / 8.0), int(w_full / 8.0)), mode='bilinear',
                                     align_corners=False)
            up_flow3[:, 0, :, :] *= float(w_full) / float(256)
            up_flow3[:, 1, :, :] *= float(h_full) / float(256)
            # ==> put the upflow in the range [Horiginal x Woriginal]

        # level 1/8 of original resolution
        ratio = 1.0 / 8.0
        warp2 = warp(c22, up_flow3*div*ratio)
        corr2 = correlation.FunctionCorrelation(tensorFirst=c12, tensorSecond=warp2)
        corr2 = self.leakyRELU(corr2)
        if self.decoder_inputs == 'corr_flow_feat':
            corr2 = torch.cat((corr2, up_flow3), 1)
        elif self.decoder_inputs == 'corr':
            corr2 = corr2
        elif self.decoder_inputs == 'corr_flow':
            corr2 = torch.cat((corr2, up_flow3), 1)
        x2, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3
        if self.refinement_at_all_levels:
            x = self.dc_conv4_level2(self.dc_conv3_level2(self.dc_conv2_level2(self.dc_conv1_level2(x2))))
            flow2 = flow2 + self.dc_conv7_level2(self.dc_conv6_level2(self.dc_conv5_level2(x)))

        up_flow2 = self.deconv2(flow2)
        if self.decoder_inputs == 'corr_flow_feat':
            up_feat2 = self.upfeat2(x2)

        # level 1/4 of original resolution
        ratio = 1.0 / 4.0
        warp1 = warp(c21, up_flow2*div*ratio)
        corr1 = correlation.FunctionCorrelation(tensorFirst=c11, tensorSecond=warp1)
        corr1 = self.leakyRELU(corr1)
        if self.decoder_inputs == 'corr_flow_feat':
            corr1 = torch.cat((corr1, up_flow2, up_feat2), 1)
        elif self.decoder_inputs == 'corr':
            corr1 = corr1
        if self.decoder_inputs == 'corr_flow':
            corr1 = torch.cat((corr1, up_flow2), 1)
        x, res_flow1 = self.decoder1(corr1)
        flow1 = res_flow1 + up_flow2
        x = self.l_dc_conv4(self.l_dc_conv3(self.l_dc_conv2(self.l_dc_conv1(x))))
        flow1 = flow1 + self.l_dc_conv7(self.l_dc_conv6(self.l_dc_conv5(x)))

        if self.evaluation:
            return flow1
        else:
            return [flow4, flow3], [flow2, flow1]
