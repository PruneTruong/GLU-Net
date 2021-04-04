from tqdm import tqdm
import torch
import numpy as np
from utils.io import writeFlow
import torch.nn as nn


def epe(input_flow, target_flow, mean=True):
    """
    End-point-Error computation
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
    Output:
        Averaged end-point-error (value)
    """
    EPE = torch.norm(target_flow - input_flow, p=2, dim=1)
    if mean:
        EPE = EPE.mean()
    return EPE


def correct_correspondences(input_flow, target_flow, alpha, img_size):
    """
    Computation PCK, i.e number of the pixels within a certain threshold
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
        alpha: threshold
        img_size: image size
    Output:
        PCK metric
    """
    # input flow is shape (BxHgtxWgt,2)
    dist = torch.norm(target_flow - input_flow, p=2, dim=1)
    # dist is shape BxHgtxWgt
    pck_threshold = alpha * img_size
    mask = dist.le(pck_threshold) # Computes dist ≤ pck_threshold element-wise (element then equal to 1)
    return mask.sum().item()


def F1_kitti_2015(input_flow, target_flow, tau=[3.0, 0.05]):
    """
    Computation number of outliers
    for which error > 3px(tau[0]) and error/magnitude(ground truth flow) > 0.05(tau[1])
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
        alpha: threshold
        img_size: image size
    Output:
        PCK metric
    """
    # input flow is shape (BxHgtxWgt,2)
    dist = torch.norm(target_flow - input_flow, p=2, dim=1)
    gt_magnitude = torch.norm(target_flow, p=2, dim=1)
    # dist is shape BxHgtxWgt
    mask = dist.gt(3.0) & (dist/gt_magnitude).gt(0.05)
    return mask.sum().item()


def calculate_epe_and_pck_per_dataset(test_dataloader, network, device, threshold_range, path_to_save=None,
                                      compute_F1=False, save=False):
    aepe_array = []
    pck_alpha_0_05_over_image = []
    pck_thresh_1_over_image = []
    pck_thresh_5_over_image = []
    F1 = 0.0

    n_registered_pxs = 0.0
    array_n_correct_correspondences = np.zeros(threshold_range.shape, dtype=np.float32)

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i_batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        mask_gt = mini_batch['correspondence_mask'].to(device)
        flow_gt = mini_batch['flow_map'].to(device)
        if flow_gt.shape[1] != 2:
            # shape is BxHxWx2
            flow_gt = flow_gt.permute(0,3,1,2)
        bs, ch_g, h_g, w_g = flow_gt.shape

        flow_estimated = network.estimate_flow(source_img, target_img, device, mode='channel_first')

        # torch tensor of shape Bx2xH_xW_, will be the same types (cuda or cpu) depending on the device
        # H_ and W_ could be smaller than the ground truth flow (ex DCG Net takes only 240x240 images)
        if flow_estimated.shape[2] != h_g or flow_estimated.shape[3] != w_g:
            '''
            the estimated flow is downscaled (the original images were downscaled before 
            passing through the network)
            as it is the case with DCG Net, the estimate flow will have shape 240x240
            it needs to be upscaled to the same size as flow_target_x and rescaled accordingly:
            '''
            ratio_h = float(h_g) / float(flow_estimated.shape[2])
            ratio_w = float(w_g) / float(flow_estimated.shape[3])
            flow_estimated = nn.functional.interpolate(flow_estimated, size=(h_g, w_g), mode='bilinear',
                                                       align_corners=False)
            flow_estimated[:, 0, :, :] *= ratio_w
            flow_estimated[:, 1, :, :] *= ratio_h
        assert flow_estimated.shape == flow_gt.shape

        flow_target_x = flow_gt.permute(0, 2, 3, 1)[:, :, :, 0]
        flow_target_y = flow_gt.permute(0, 2, 3, 1)[:, :, :, 1]
        flow_est_x = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 0]  # B x h_g x w_g
        flow_est_y = flow_estimated.permute(0, 2, 3, 1)[:, :, :, 1]

        flow_target = \
            torch.cat((flow_target_x[mask_gt].unsqueeze(1),
                       flow_target_y[mask_gt].unsqueeze(1)), dim=1)
        flow_est = \
            torch.cat((flow_est_x[mask_gt].unsqueeze(1),
                       flow_est_y[mask_gt].unsqueeze(1)), dim=1)
        # flow_target_x[mask_gt].shape is (number of pixels), then with unsqueze(1) it becomes (number_of_pixels, 1)
        # final shape is (B*H*W , 2), B*H*W is the number of registered pixels (according to ground truth masks)

        # let's calculate EPE per batch
        aepe = epe(flow_est, flow_target)  # you obtain the mean per pixel
        aepe_array.append(aepe.item())

        # let's calculate PCK values
        img_size = max(mini_batch['source_image_size'][0], mini_batch['source_image_size'][1]).float().to(device)
        alpha_0_05 = correct_correspondences(flow_est, flow_target, alpha=0.05, img_size=img_size)
        px_1 = correct_correspondences(flow_est, flow_target, alpha=1.0/float(img_size), img_size=img_size) # threshold of 1 px
        px_5 = correct_correspondences(flow_est, flow_target, alpha=5.0/float(img_size), img_size=img_size) # threshold of 5 px

        # percentage per image is calculated for each
        pck_alpha_0_05_over_image.append(alpha_0_05/flow_target.shape[0])
        pck_thresh_1_over_image.append(px_1/flow_target.shape[0])
        pck_thresh_5_over_image.append(px_5/flow_target.shape[0])

        # PCK curve for different thresholds ! ATTENTION, here it is over the whole dataset and not per image
        n_registered_pxs += flow_target.shape[0]  # also equal to number of correspondences that should be correct
        # according to ground truth mask
        for t_id, threshold in enumerate(threshold_range):
            array_n_correct_correspondences[t_id] += correct_correspondences(flow_est,
                                                                             flow_target,
                                                                             alpha=threshold,
                                                                             img_size=img_size)
            # number of correct pixel correspondence below a certain threshold, added for each batch

        if compute_F1:
            F1 += F1_kitti_2015(flow_est, flow_target)

        if save:
            writeFlow(np.dstack([flow_est_x[0].cpu().numpy(), flow_est_y[0].cpu().numpy()]),
            'batch_{}'.format(i_batch), path_to_save)

    output = {'final_eape': np.mean(aepe_array),
              'pck_alpha_0_05_average_per_image': np.mean(pck_alpha_0_05_over_image),
              'pck_thresh_1_average_per_image': np.mean(pck_thresh_1_over_image),
              'pck_thresh_5_average_per_image': np.mean(pck_thresh_5_over_image),
              'alpha_threshold': threshold_range.tolist(),
              'pixel_threshold': np.round(threshold_range * img_size.cpu().numpy(), 2).tolist(),
              'pck_per_threshold_over_dataset': np.float32(array_n_correct_correspondences /
                                                           (n_registered_pxs + 1e-6)).tolist()}

    if compute_F1:
        output['kitti2015-F1'] = F1 / float(n_registered_pxs)
    return output
