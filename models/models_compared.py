
from os import path as osp
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from models.our_models.GLUNet import GLUNet_model
from models.our_models.Semantic_GLUNet import SemanticGLUNet_model
from models.our_models.GLOCALNet import GLOCALNet_model
from models.our_models.GLOBALNet import GLOBALNet_model
from models.our_models.LOCALNet import LOCALNet_model
from datasets.util import convert_mapping_to_flow, convert_flow_to_mapping
from torchvision import transforms


class GLU_Net:
    def __init__(self, model_type='DPED_CityScape_ADE', path_pre_trained_models='pre_trained_models/',
                 apply_flipping_condition=False, pyramid_type='VGG', iterative_refinement=True,
                 feature_concatenation=False, decoder_inputs='corr_flow_feat', up_feat_channels=2,
                 cyclic_consistency=True, consensus_network=False, dense_connections=True):

        self.apply_flipping_condition = apply_flipping_condition
        # semantic glu-net
        if feature_concatenation:
            net = SemanticGLUNet_model(batch_norm=True, pyramid_type=pyramid_type,
                                       div=1.0, evaluation=True, consensus_network=consensus_network,
                                       iterative_refinement=iterative_refinement)

            if consensus_network:
                checkpoint_fname = osp.join(path_pre_trained_models, 'Semantic_GLUNet_' + model_type + '.pth')
            else:
                raise ValueError('there are no saved weights for this configuration')

        else:
            net = GLUNet_model(batch_norm=True,
                                pyramid_type=pyramid_type,
                                div=1.0, evaluation=True,
                                refinement_at_adaptive_reso=True,
                                decoder_inputs=decoder_inputs,
                                upfeat_channels=up_feat_channels,
                                dense_connection=dense_connections,
                                cyclic_consistency=cyclic_consistency,
                                consensus_network=consensus_network,
                                iterative_refinement=iterative_refinement)

            if cyclic_consistency and dense_connections and decoder_inputs == 'corr_flow_feat' and up_feat_channels == 2:
                checkpoint_fname = osp.join(path_pre_trained_models, 'GLUNet_' + model_type + '.pth')
            else:
                raise ValueError('there are no saved weights for this configuration')

        if not osp.isfile(checkpoint_fname):
            raise ValueError('check the snapshots path, checkpoint is {}'.format(checkpoint_fname))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            net.load_state_dict(torch.load(checkpoint_fname))
        except:
            net.load_state_dict(torch.load(checkpoint_fname)['state_dict'])

        print("loaded the weights")
        net.eval()
        self.net = net.to(device) # load on GPU

    def estimate_flow(self, source_img, target_img, device, mode='channel_first'):
        if self.apply_flipping_condition:
            source_img_copy, target_img_copy, source_img_256, target_img_256, ratio_x, ratio_y, h_original, w_original\
                = self.net.flipping_condition(source_img, target_img, device)
            estimated_flow = self.net(target_img_copy, source_img_copy,
                                      target_img_256, source_img_256)

            flow_original_reso = torch.nn.functional.interpolate(input=estimated_flow, size=(h_original, w_original),
                                                                 mode='bilinear', align_corners=False)
            flow_original_reso[:, 0, :, :] *= ratio_x
            flow_original_reso[:, 1, :, :] *= ratio_y

            if self.net.target_image_is_flipped:
                flipped_mapping = convert_flow_to_mapping(flow_original_reso, output_channel_first=True).permute(0, 2, 3, 1).cpu().numpy()
                b = flipped_mapping.shape[0]
                mapping_per_batch = []
                for i in range(b):
                    map = np.copy(np.fliplr(flipped_mapping[i]))
                    mapping_per_batch.append(map)

                mapping = torch.from_numpy(np.float32(mapping_per_batch)).permute(0, 3, 1, 2).to(device)
                flow_original_reso = convert_mapping_to_flow(mapping, device)

        else:
            source_img_copy, target_img_copy, source_img_256, target_img_256, ratio_x, ratio_y, h_original, w_original\
                = self.net.pre_process_data(source_img, target_img, device)
            estimated_flow = self.net(target_img_copy, source_img_copy,
                                      target_img_256, source_img_256)

            flow_original_reso = torch.nn.functional.interpolate(input=estimated_flow, size=(h_original, w_original),
                                                                 mode='bilinear', align_corners=False)
            flow_original_reso[:, 0, :, :] *= ratio_x
            flow_original_reso[:, 1, :, :] *= ratio_y

        if mode == 'channel_first':
            return flow_original_reso
        else:
            return flow_original_reso.permute(0,2,3,1)


class GLOCAL_Net:
    def __init__(self, model_type='default', path_pre_trained_models='pre_trained_models/',
                 constrained_corr=True, global_corr=True, residual=True,
                 decoder_inputs='flow_and_feat', refinement_32=False):

        self.fixed_input=True
        if global_corr:
            if constrained_corr:
                net = GLOCALNet_model(evaluation=True, residual=residual, input_decoder=decoder_inputs,
                                      div=1, refinement=True, batch_norm=True, refinement_32=refinement_32)
                checkpoint_fname = osp.join(path_pre_trained_models, 'GLOCALNet_' + model_type + '.pth')

            else:
                net = GLOBALNet_model(evaluation=True, div=1, refinement=True, batch_norm=True)
                checkpoint_fname = osp.join(path_pre_trained_models, 'GLOBALNet_' + model_type + '.pth')
        else:
            self.fixed_input=False
            net = LOCALNet_model(evaluation=True, div=1, refinement=True, batch_norm=True)
            checkpoint_fname = osp.join(path_pre_trained_models, 'LOCALNet_' + model_type + '.pth')

        # check the checkpoints
        if not osp.isfile(checkpoint_fname):
            raise ValueError('check the snapshots path')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            net.load_state_dict(torch.load(checkpoint_fname))
        except:
            net.load_state_dict(torch.load(checkpoint_fname)['state_dict'])

        print("loaded the weights")
        net.eval()
        self.net = net.to(device) # load on GPU

    def pre_process_data(self, source_img, target_img, device):
        # img has shape bx3xhxw
        b, _, h_scale, w_scale = target_img.shape
        if self.fixed_input:
            int_preprocessed_height = 256
            int_preprocessed_width = 256
        else:
            int_preprocessed_height = int(math.floor(math.ceil(h_scale / 16.0) * 16.0))
            int_preprocessed_width = int(math.floor(math.ceil(w_scale / 16.0) * 16.0))

        source_img_copy = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area')
        target_img_copy = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area')
        source_img_copy = source_img_copy.float().div(255.0)
        target_img_copy = target_img_copy.float().div(255.0)
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])
        mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
        '''
        # to get exactly same results as in paper, but not on gpu
        source_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))
        target_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))

        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])
        normTransform = transforms.Normalize(mean_vector, std_vector)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((int_preprocessed_height, int_preprocessed_width), interpolation=2),
                                        transforms.ToTensor(),
                                        normTransform])
        for i in range(source_img.shape[0]):
            source_img_copy[i] = transform(source_img[i])
            target_img_copy[i] = transform(target_img[i])
        '''

        ratio_x = float(w_scale)/float(int_preprocessed_width)
        ratio_y = float(h_scale)/float(int_preprocessed_height)
        return source_img_copy.to(device), target_img_copy.to(device), ratio_x, ratio_y

    def estimate_flow(self, source_img, target_img, device, mode='channel_first'):
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        source_img_copy, target_img_copy, ratio_x, ratio_y = self.pre_process_data(source_img.clone().detach(),
                                                                                   target_img.clone().detach(),
                                                                                   device)

        estimated_flow = torch.nn.functional.interpolate(input=self.net(target_img_copy, source_img_copy),
                                                         size=(h_scale, w_scale),
                                                         mode='bilinear', align_corners=False)

        estimated_flow[:, 0, :, :] *= ratio_x
        estimated_flow[:, 1, :, :] *= ratio_y
        # shape is Bx2xHxW here
        if mode == 'channel_first':
            return estimated_flow
        else:
            return estimated_flow.permute(0,2,3,1)
