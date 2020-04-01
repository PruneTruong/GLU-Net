import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np
import cv2
import pandas as pd


def get_gt_correspondence_mask(flow):

    # 1 = convert flow to mapping
    h,w = flow.shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (flow[:,:,0]+X).astype(np.float32)
    map_y = (flow[:,:,1]+Y).astype(np.float32)

    # 2 = compute mask, check that it is not normalised to -1 and 1 and numpy array
    mask_x = np.logical_and(map_x>0, map_x< w)
    mask_y = np.logical_and(map_y>0, map_y< h)
    mask = np.logical_and(mask_x, mask_y).astype(np.uint8)
    return mask


def default_loader(root, path_imgs):
    image1 = cv2.imread(os.path.join(root, path_imgs[0]))
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)# it is black and white, convert to BGR
    else:
        image1 = image1[:,:,::-1] # convert to RGB from BGR

    image2 = cv2.imread(os.path.join(root, path_imgs[1]))
    if len(image2.shape)  == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)# it is black and white, convert to BGR
    else:
        image2 = image2[:,:,::-1] # convert to RGB from BGR
    return image1.astype(np.uint8), image2.astype(np.uint8)


def make_dataset(dir):
    '''Will search for pairs of images in the order in which they appear  '''
    images = []
    #name_images=[f for f in sorted(os.listdir(dir)) if f.endswith('.png') or f.endswith('.ppm') or f.endswith('.jpg') or f.endswith('.jpeg')]
    name_images = [f for f in sorted(os.listdir(dir))]
    print(len(name_images))
    i=0
    while i < len(name_images)-1:
        img1 = name_images[i]
        img2 = name_images[i+1]
        images.append([img1, img2])
        i += 2
    print(images)
    return images


class DatasetNoGT(data.Dataset):
    def __init__(self, root, path_csv=None, source_image_transform=None, target_image_transform=None,
                 loader=default_loader):

        self.root = root
        if isinstance(path_csv, str):
            # it is a string, must be read from csv
            self.path_list = pd.read_csv(path_csv)
            self.csv = True
        elif isinstance(path_csv, list):
            # a list is directly given
            self.path_list = path_csv
            self.csv = False
        elif path_csv is None:
            self.path_list = make_dataset(root)
            self.csv = False
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.loader = loader

    def __getitem__(self, index):
        if self.csv:
            inputs = self.path_list.iloc[index] #maybe colum indexer , check if this is a list
        else:
            inputs = self.path_list[index]
        im1, im2 = self.loader(self.root, inputs)

        # pad to same shape both images with zero
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
        im1 = cv2.copyMakeBorder(im1, pad_y_1, 0, pad_x_1, 0, cv2.BORDER_CONSTANT)
        im2 = cv2.copyMakeBorder(im2, pad_y_2, 0, pad_x_2, 0, cv2.BORDER_CONSTANT)
        shape = im1.shape

        if self.source_image_transform is not None:
            im1 = self.source_image_transform(im1)
        if self.target_image_transform is not None:
            im2 = self.target_image_transform(im2)

        return {'source_image': im1,
                'target_image': im2,
                'image_shape': shape
                }
        # attention here this is flow and not correspondence !! and it is nor normalised

    def __len__(self):
        return len(self.path_list)
