import os.path
import glob
from .listdataset import ListDataset
from datasets.util import split2list
from utils import co_flow_and_images_transforms
from imageio import imread
from .listdataset import load_flo
import numpy as np
import cv2

'''
extracted from https://github.com/ClementPinard/FlowNetPytorch/tree/master/datasets
Dataset routines for MPI Sintel.
http://sintel.is.tue.mpg.de/
clean version imgs are without shaders, final version imgs are fully rendered
The datasets is not very big, you might want to only pretrain on it for flownet
'''


def make_dataset(dataset_dir, split, dataset_type='clean'):
    flow_dir = 'flow'
    assert(os.path.isdir(os.path.join(dataset_dir,flow_dir)))
    img_dir = dataset_type
    assert(os.path.isdir(os.path.join(dataset_dir,img_dir)))

    images = []
    for flow_map in sorted(glob.glob(os.path.join(dataset_dir,flow_dir,'*','*.flo'))):
        flow_map = os.path.relpath(flow_map,os.path.join(dataset_dir,flow_dir))

        scene_dir, filename = os.path.split(flow_map)
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split('_')
        frame_nb = int(frame_nb)
        img1 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb + 1))
        img2 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb))
        # img2 is target, which corresponds to the first image for sintel
        flow_map = os.path.join(flow_dir, flow_map)
        if not (os.path.isfile(os.path.join(dataset_dir,img1)) and os.path.isfile(os.path.join(dataset_dir,img2))):
            continue
        images.append([[img1,img2],flow_map])

    return split2list(images, split, default_split=0.87)


def mpisintel_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flo = os.path.join(root,path_flo)

    invalid_mask_dir = 'invalid'
    occlusion_mask_dir = 'occlusions'

    scene_dir, filename = os.path.split(path_flo)
    flow, scene_dir = os.path.split(scene_dir)
    filename = filename[:-4]

    path_invalid_mask = os.path.join(invalid_mask_dir, scene_dir, '{}.png'.format(filename))
    invalid_mask = cv2.imread(os.path.join(root, path_invalid_mask), 0).astype(np.uint8)
    valid_mask = (invalid_mask == 0)

    # if want to remove occluded regions
    path_occlusion_mask = os.path.join(occlusion_mask_dir, scene_dir, '{}.png'.format(filename))
    occluded_mask = cv2.imread(os.path.join(root, path_occlusion_mask), 0).astype(np.uint8)
    noc_mask = (occluded_mask == 0).astype(np.uint8)

    return [imread(img).astype(np.uint8) for img in imgs], load_flo(flo), valid_mask.astype(np.uint8)


def mpi_sintel_clean(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_dataset(root, split, 'clean')
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform,
                                co_transform=co_transform, loader=mpisintel_loader, mask=True)
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform,
                               co_transform=co_transform, loader=mpisintel_loader, mask=True)
    return train_dataset, test_dataset


def mpi_sintel_final(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                     co_transform=None, split=None):
    train_list, test_list = make_dataset(root, split, 'final')
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform,
                                co_transform=co_transform, loader=mpisintel_loader, mask=True)
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform,
                               co_transform=co_transform, loader=mpisintel_loader, mask=True)

    return train_dataset, test_dataset


def mpi_sintel_both(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                    co_transform=None, test_image_transform=None, split=None):
    '''load images from both clean and final folders.
    We cannot shuffle input, because it would very likely cause data snooping
    for the clean and final frames are not that different'''
    #assert(isinstance(split, str)), 'To avoid data snooping, you must provide a static list of train/val when dealing with both clean and final.'
    ' Look at Sintel_train_val.txt for an example'
    train_list1, test_list1 = make_dataset(root, split, 'clean')
    train_list2, test_list2 = make_dataset(root, split, 'final')
    train_dataset = ListDataset(root, train_list1 + train_list2, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform,
                                co_transform=co_transform, loader=mpisintel_loader, mask=True)
    if test_image_transform is None:
        test_dataset = ListDataset(root, test_list1 + test_list2, source_image_transform=source_image_transform,
                                   target_image_transform=target_image_transform,
                                   flow_transform=flow_transform,
                                   co_transform=co_flow_and_images_transforms.CenterCrop((384, 1024)),
                                   loader=mpisintel_loader, mask=True)
    else:
        test_dataset = ListDataset(root, test_list1 + test_list2, source_image_transform=test_image_transform,
                                   target_image_transform=test_image_transform,
                                   flow_transform=flow_transform,
                                   co_transform=co_flow_and_images_transforms.CenterCrop((384,1024)),
                                   loader=mpisintel_loader, mask=True)

    return train_dataset, test_dataset
