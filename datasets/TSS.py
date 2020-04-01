from __future__ import division
import os.path
from datasets.listdataset import ListDataset
from datasets.util import split2list
import numpy as np
from datasets.util import load_flo
try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for TSS which uses 16bit PNG images", ImportWarning)


def pad_to_same_shape(im1, im2, flow, mask):
    # pad to same shape
    if len(im1.shape) == 2:
        im1 = np.dstack([im1,im1,im1])

    if len(im2.shape) == 2:
        im2 = np.dstack([im2,im2,im2])

    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)
    # value so that they are not represented when plottung gt (value of 0 would
    # represent them), nan when interpolating is not good
    flow = cv2.copyMakeBorder(flow, 0, pad_y_2, 0, pad_x_2, borderType = cv2.BORDER_REPLICATE)
    mask = cv2.copyMakeBorder(mask, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)
    return im1, im2, flow, mask


def make_dataset(dir, split):
    images = []
    dir_list = [f for f in os.listdir(os.path.join(dir)) if
                os.path.isdir(os.path.join(dir, f))]
    for image_dir in sorted(dir_list):
        if image_dir in ['FG3DCar', 'JODS', 'PASCAL']:
            folders_list = [f for f in os.listdir(os.path.join(dir, image_dir)) if
                            os.path.isdir(os.path.join(dir, image_dir, f))]
            for folders in sorted(folders_list):
                img_dir = os.path.join(image_dir, folders)

                # the flow is taken both ways !
                img1 = os.path.join(img_dir, 'image1.png')
                img2 = os.path.join(img_dir, 'image2.png')
                flow_map = os.path.join(img_dir, 'flow2.flo')
                images.append([[img1, img2], flow_map])

                img1 = os.path.join(img_dir, 'image2.png')
                img2 = os.path.join(img_dir, 'image1.png') # target
                flow_map = os.path.join(img_dir, 'flow1.flo')
                images.append([[img1, img2], flow_map])
        else:
            img_dir = image_dir
            # the flow is taken both ways
            img1 = os.path.join(img_dir, 'image1.png')  # path to image_1
            img2 = os.path.join(img_dir, 'image2.png')  # path to image_3, they say image 10 is the reference
            flow_map = os.path.join(img_dir, 'flow2.flo')
            images.append([[img1, img2], flow_map])

            img1 = os.path.join(img_dir, 'image2.png')
            img2 = os.path.join(img_dir, 'image1.png')
            flow_map = os.path.join(img_dir, 'flow1.flo')
            images.append([[img1, img2], flow_map])

    return split2list(images, split, default_split=0.9)


def TSS_flow_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root, path) for path in path_imgs]

    flo = os.path.join(root, path_flo)
    flow = load_flo(flo)
    base_path = os.path.dirname(path_flo)

    # getting the mask
    image_number = path_flo[-5] # getting the mask number, either 1 or 2 depending which image is the target !
    path_mask = os.path.join(root, base_path, 'mask{}.png'.format(image_number))
    mask = cv2.imread(path_mask, 0)/255 # before it was 255, we want mask in range 0,1

    images = [cv2.imread(img)[:,:,::-1].astype(np.uint8) for img in imgs]
    source_size = images[0].shape # threshold is max size of source image for pck

    # we want both images to be of the same size to give to network
    im1, im2, flow, mask = pad_to_same_shape(images[0], images[1], flow, mask)

    return [im1, im2], flow, mask.astype(np.uint8), source_size


def TSS(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
        co_transform=None, split=None):
    train_list, test_list = make_dataset(root, split)
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform,
                                loader=TSS_flow_loader, mask=True, size=True)
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform,
                               co_transform=co_transform,
                               loader=TSS_flow_loader, mask=True, size=True)
    return train_dataset, test_dataset
