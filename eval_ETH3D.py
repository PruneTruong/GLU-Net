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
from datasets.ETH3D_interval import ETH_interval

dataset_names = sorted(name for name in datasets.__all__)
model_type = ['GLUNet', 'SemanticGLUNet', 'LOCALNet', 'GLOBALNet', 'GLOCALNet']
pre_trained_model_type = ['DPED_CityScape_ADE', 'tokyo', 'chairs-things',  'flying-chairs']


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Evaluation code for ETH3D')
# Paths
parser.add_argument('--data_dir', metavar='DIR', type=str,
                    help='path to folder containing images and flows for validation')
parser.add_argument('--model', type=str, default='GLUNet',
                    help='Model to use', choices=model_type)
parser.add_argument('--pre_trained_models', nargs='+', choices=pre_trained_model_type,
                    help='name of pre trained models, can have several ones')
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

# ETH3D dataset information
dataset_names = ['lakeside', 'sand_box', 'storage_room', 'storage_room_2', 'tunnel', 'delivery_area', 'electro',
                 'forest', 'playground', 'terrains']
rates = list(range(3, 16, 2))


for pre_trained_model_type in pre_trained_models:
    print('model: ' + args.model + ', pre-trained model: ' + pre_trained_model_type)
    with torch.no_grad():
        # define the network to use
        if args.model == 'GLUNet':
            network = GLU_Net(model_type=pre_trained_model_type,
                              consensus_network=False,
                              cyclic_consistency=True,
                              iterative_refinement=True)

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
        name_to_save = args.model + '_' + 'ETH3D'
        threshold_range = np.linspace(0.002, 0.2, num=50)

        dict_results = {}
        for rate in rates:
            print('Computing results for interval {}...'.format(rate))
            dict_results['rate_{}'.format(rate)] = {}
            list_of_outputs_per_rate = []
            for name_dataset in dataset_names:
                print('looking at dataset {}...'.format(name_dataset))
                test_set = ETH_interval(root=args.data_dir,
                                        path_list=os.path.join(args.data_dir, 'info_ETH3D_files',
                                                               '{}_every_5_rate_of_{}'.format(name_dataset, rate)),
                                        source_image_transform=input_images_transform,
                                        target_image_transform=input_images_transform,
                                        flow_transform=gt_flow_transform,
                                        co_transform=co_transform)  # only test
                test_dataloader = DataLoader(test_set,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=8)
                print(test_set.__len__())
                output = calculate_epe_and_pck_per_dataset(test_dataloader, network, device, threshold_range)
                
                # to save the intermediate results
                # dict_results['rate_{}'.format(rate)][name_dataset] = output
                list_of_outputs_per_rate.append(output)

            # average over all datasets for this particular rate of interval
            avg = {'final_eape': np.mean([list_of_outputs_per_rate[i]['final_eape'] for i in range(len(dataset_names))]),
                   'pck_thresh_1_average_per_image': np.mean([list_of_outputs_per_rate[i]
                                                              ['pck_thresh_1_average_per_image'] for i in range(len(dataset_names))]),
                   'pck_thresh_5_average_per_image': np.mean([list_of_outputs_per_rate[i]
                                                              ['pck_thresh_5_average_per_image'] for i in range(len(dataset_names))])
                   }
            dict_results['rate_{}'.format(rate)]['avg'] = avg

    # save the dictionnary for this particular pre trained model
    save_dict['{}'.format(pre_trained_model_type)]=dict_results

with open('{}/{}.txt'.format(args.save_dir, 'metrics_{}'.format(name_to_save)), 'w') as outfile:
        json.dump(save_dict, outfile, ensure_ascii=False, separators=(',', ':'))
        print('written to file ')


