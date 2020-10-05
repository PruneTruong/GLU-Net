import numpy as np
import os
import torch
from models.models_compared import GLOCAL_Net, GLU_Net
from utils.evaluate import calculate_epe_and_pck_per_dataset
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import datasets
from utils.image_transforms import ArrayToTensor
from tqdm import tqdm
from utils.io import writeFlow
import torch.nn as nn


dataset_names = sorted(name for name in datasets.__all__)
model_type = ['GLUNet', 'SemanticGLUNet', 'LOCALNet', 'GLOBALNet', 'GLOCALNet']
pre_trained_model_type = ['DPED_CityScape_ADE', 'tokyo', 'chairs-things',  'flying-chairs']


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Evaluation code')
# Paths
parser.add_argument('--data_dir', metavar='DIR', type=str,
                    help='path to folder containing images and flows for validation')
parser.add_argument('--datasets', metavar='DATASET', default='HPatchesDataset',
                    choices=dataset_names,
                    help='datasets type : ' +' | '.join(dataset_names))
parser.add_argument('--hpatches_original_size', default=False, type=boolean_string,
                    help='use hpatches_original_size? default is False')
parser.add_argument('--model', type=str, default='GLUNet',
                    help='Model to use', choices=model_type)
parser.add_argument('--flipping_condition', dest='flipping_condition', default=False, type=boolean_string,
                    help='apply flipping condition during eval ? default is false')
parser.add_argument('--pre_trained_models', nargs='+', choices=pre_trained_model_type,
                    help='name of pre trained models, can have several ones')
parser.add_argument('--save', default=False, type=boolean_string,
                    help='save the flow files ? default is False')
parser.add_argument('--save_dir', type=str, default='evaluation/',
                    help='path to directory to save the text files and results')
parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')
args = parser.parse_args()

torch.cuda.empty_cache()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

save_dict = {}
if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

pre_trained_models = args.pre_trained_models

# define the image processing parameters, the actual pre-processing is done within the model functions
input_images_transform = transforms.Compose([ArrayToTensor(get_float=False)])  # only put channel first
gt_flow_transform = transforms.Compose([ArrayToTensor()])  # only put channel first
co_transform = None

for pre_trained_model_type in pre_trained_models:
    print('model: ' + args.model + ', pre-trained model: ' + pre_trained_model_type)
    with torch.no_grad():
        # define the network to use
        if args.model == 'GLUNet':
            network = GLU_Net(model_type=pre_trained_model_type,
                              consensus_network=False,
                              cyclic_consistency=True,
                              iterative_refinement=True,
                              apply_flipping_condition=args.flipping_condition)

        elif args.model == 'SemanticGLUNet':
            network = GLU_Net(model_type=pre_trained_model_type,
                              feature_concatenation=True,
                              cyclic_consistency=False,
                              consensus_network=True,
                              iterative_refinement=True,
                              apply_flipping_condition=args.flipping_condition)

        elif args.model == 'GLOCALNet':
            network = GLOCAL_Net(model_type=pre_trained_model_type, constrained_corr=True, global_corr=True)

        elif args.model == 'GLOBALNet':
            network = GLOCAL_Net(model_type=pre_trained_model_type, constrained_corr=False, global_corr=True)

        elif args.model == 'LOCALNet':
            network = GLOCAL_Net(model_type=pre_trained_model_type, constrained_corr=True, global_corr=False)

        # choosing the different dataset !
        name_to_save = args.model + '_' + args.datasets

        # no ground truth available, saving the flow file
        if args.datasets == 'DatasetNoGT':
            # only qualitative, no quantitative value because no gt
            test_set = datasets.__dict__[args.datasets](args.data_dir, first_image_transform=input_images_transform,
                                                        second_image_transform=input_images_transform)  # only test

            test_dataloader = DataLoader(test_set,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=1)

            path_to_save = os.path.join(args.save_dir, '{}_{}'.format(name_to_save, pre_trained_model_type))
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            for i_batch, mini_batch in pbar:
                source_img = mini_batch['source_image']
                target_img = mini_batch['target_image']
                h_g, w_g, ch_g = mini_batch['image_shape']

                flow_estimated = network.estimate_flow(source_img, target_img, device, mode='channel_first')
                # if flow is smaller than the input image size, resize it and scale it accordingly
                if flow_estimated.shape[2] != h_g or flow_estimated.shape[3] != w_g:
                    ratio_h = float(h_g) / float(flow_estimated.shape[2])
                    ratio_w = float(w_g) / float(flow_estimated.shape[3])
                    flow_estimated = nn.functional.interpolate(flow_estimated, size=(h_g, w_g), mode='bilinear',
                                                               align_corners=False)
                    flow_estimated[:, 0, :, :] *= ratio_w
                    flow_estimated[:, 1, :, :] *= ratio_h
                assert flow_estimated.shape[2] == h_g and flow_estimated.shape[3] == w_g

                flow_est_x = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 0]
                flow_est_y = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 1]

                writeFlow(np.dstack([flow_est_x[0].cpu().numpy(), flow_est_y[0].cpu().numpy()]),
                           'batch_{}'.format(i_batch), path_to_save)

        else:
            # Datasets with ground-truth flow fields available

            # HPATCHES dataset
            threshold_range = np.linspace(0.002, 0.2, num=50)
            if args.datasets=='HPatchesdataset':
                number_of_scenes = 5 + 1
                list_of_outputs = []

                # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
                for id, k in enumerate(range(2, number_of_scenes + 2)):
                    if id == 5:
                        # looks at the total of all scenes
                        test_set = datasets.HPatchesdataset(args.data_dir,
                                                            os.path.join('datasets/csv_files/',
                                                                         'hpatches_all.csv'.format(k)),
                                                            input_images_transform, gt_flow_transform, co_transform,
                                                            original_size=args.hpatches_original_size)
                        path_to_save = os.path.join(args.save_dir, 'all', '{}_{}'.format(name_to_save,
                                                                                         pre_trained_model_type))
                    else:
                        # looks at each scene individually
                        test_set = datasets.HPatchesdataset(args.data_dir,
                                                            os.path.join('datasets/csv_files/',
                                                                         'hpatches_1_{}.csv'.format(k)),
                                                            input_images_transform, gt_flow_transform, co_transform,
                                                            original_size=args.hpatches_original_size)
                        path_to_save = os.path.join(args.save_dir, 'scene_{}'.format(k), '{}_{}'.format(name_to_save,
                                                                                                        pre_trained_model_type))

                    test_dataloader = DataLoader(test_set,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=8)
                    if not os.path.isdir(path_to_save) and args.save:
                        os.makedirs(path_to_save)

                    output_scene = calculate_epe_and_pck_per_dataset(test_dataloader, network, device, threshold_range,
                                                                     path_to_save, save=args.save)
                    list_of_outputs.append(output_scene)

                output = {'scene_1': list_of_outputs[0], 'scene_2': list_of_outputs[1], 'scene_3': list_of_outputs[2],
                          'scene_4': list_of_outputs[3], 'scene_5': list_of_outputs[4], 'all': list_of_outputs[5]}

            else:
                # OTHER DATASETS (kitti, tss..)
                train_set, test_set = datasets.__dict__[args.datasets](args.data_dir, source_image_transform=input_images_transform,
                                                                       target_image_transform=input_images_transform,
                                                                       flow_transform=gt_flow_transform, split=0)  # only test

                test_dataloader = DataLoader(test_set,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=8)
                print('test set contains {} image pairs'.format(test_set.__len__()))

                path_to_save = os.path.join(args.save_dir, '{}_{}'.format(name_to_save, pre_trained_model_type))
                if not os.path.isdir(path_to_save) and args.save:
                    os.makedirs(path_to_save)

                # OPTICAL FLOW DATASET
                if args.datasets == 'KITTI_occ' or args.datasets == 'KITTI_noc':
                    output = calculate_epe_and_pck_per_dataset(test_dataloader, network, device, threshold_range,
                                                               path_to_save, compute_F1=True, save=args.save)
                else:
                    output = calculate_epe_and_pck_per_dataset(test_dataloader, network, device, threshold_range,
                                                               path_to_save, save=args.save)

            save_dict['{}'.format(pre_trained_model_type)]=output

            with open('{}/{}.txt'.format(args.save_dir, 'metrics_{}'.format(name_to_save)), 'w') as outfile:
                    json.dump(save_dict, outfile, ensure_ascii=False, separators=(',', ':'))
                    print('written to file ')


