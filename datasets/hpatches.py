'''
Extracted from https://github.com/AaltoVision/DGC-Net and modified
'''


from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
from os import path as osp
import torch
from datasets.util import convert_mapping_to_flow


def HPatchesdataset(root, csv, image_transform=None, flow_transform=None,
                    co_transform=None, original_size=False):
    test_dataset = HPatchesDataset(root, csv, image_transform, flow_transform, co_transform, original_size=original_size)
    return test_dataset # because no training dataset, only test


class HPatchesDataset(Dataset):
    """HPatches datasets (for evaluation)
    Args:
        root: filepath to the datasets (full resolution)
        path_list: path to csv file with ground-truth data
        image_transforms: image transformations (data preprocessing)
        flow_transforms: transforms for the flow
        co_transform: transform to apply to both images and flow fields
        original_size: bool to keep original_size, overrides image_size below
        image_size: size (tuple) of the output images
    Output:
        source_image: source image
        target_image: target image
        correspondence_map: pixel correspondence map
            between source and target views
        mask: valid/invalid correspondences
    """
    def __init__(self,
                 root,
                 path_list,
                 image_transform,
                 flow_transform,
                 co_transform,
                 original_size=False,
                 image_size=(240, 240)):

        self.root = root
        self.path_list = path_list
        self.df = pd.read_csv(path_list)
        self.image_transform = image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform
        self.image_size = image_size
        self.original_size = original_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        obj = str(data.obj)

        obj_dir=osp.join('{}'.format(self.root), obj)
        im1_id, im2_id = str(data.im1), str(data.im2)

        h_ref_orig, w_ref_orig = data.Him.astype('int'), data.Wim.astype('int')
        h_trg_orig, w_trg_orig, _ = \
            cv2.imread(osp.join(obj_dir, im2_id + '.ppm'), -1).shape
        if self.original_size:
            h_scale, w_scale = h_trg_orig, w_trg_orig
        else:
            h_scale, w_scale = self.image_size[0], self.image_size[1]

        H = data[5:].astype('double').values.reshape((3, 3))

        # As gt homography is calculated for (h_orig, w_orig) images,
        # we need to map it to (h_scale, w_scale), that is 240x240
        # H_scale = S * H * inv(S)

        S1 = np.array([[w_scale / w_ref_orig, 0, 0],
                       [0, h_scale / h_ref_orig, 0],
                       [0, 0, 1]])
        S2 = np.array([[w_scale / w_trg_orig, 0, 0],
                       [0, h_scale / h_trg_orig, 0],
                       [0, 0, 1]])

        H_scale = np.dot(np.dot(S2, H), np.linalg.inv(S1))

        # inverse homography matrix
        Hinv = np.linalg.inv(H_scale)

        # estimate the grid
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                           np.linspace(0, h_scale - 1, h_scale))
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.dot(Hinv, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        Xwarp=XwarpHom / (ZwarpHom + 1e-8)
        Ywarp=YwarpHom / (ZwarpHom + 1e-8)
        # and now the grid
        grid_gt = torch.stack([Xwarp.view(h_scale, w_scale),
                               Ywarp.view(h_scale, w_scale)], dim=-1)

        # mask
        mask = grid_gt[:, :, 0].ge(0) & grid_gt[:, :, 0].le(w_scale-1) & \
               grid_gt[:, :, 1].ge(0) & grid_gt[:, :, 1].le(h_scale-1)

        img1 = \
            cv2.resize(cv2.imread(osp.join(self.root,
                                           obj,
                                           im1_id + '.ppm'), -1),
                       (h_scale, w_scale))
        img2 = \
            cv2.resize(cv2.imread(osp.join(self.root,
                                           obj,
                                           im2_id + '.ppm'), -1),
                       (h_scale, w_scale))
        _, _, ch = img1.shape
        if ch == 3:
            img1_tmp = cv2.imread(osp.join(self.root,
                                           obj,
                                           im1_id + '.ppm'), -1)
            img2_tmp = cv2.imread(osp.join(self.root,
                                           obj,
                                           im2_id + '.ppm'), -1)
            img1 = cv2.cvtColor(cv2.resize(img1_tmp,
                                           (w_scale, h_scale)),
                                cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.resize(img2_tmp,
                                           (w_scale,h_scale)),
                                cv2.COLOR_BGR2RGB)

        gt_flow = convert_mapping_to_flow(grid_gt, output_channel_first=False)
        inputs = [img1, img2]

        # global transforms
        if self.co_transform is not None:
            inputs, gt_flow = self.co_transform(inputs, gt_flow)
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.image_transform is not None:
            inputs[0] = self.image_transform(inputs[0])
            inputs[1] = self.image_transform(inputs[1])
        if self.flow_transform is not None:
            gt_flow = self.flow_transform(gt_flow)

        return {'source_image': inputs[0],
                'target_image': inputs[1],
                'flow_map': gt_flow,
                'correspondence_mask': mask,  # mask_x and mask_y
                'source_image_size': img1.shape
                }

