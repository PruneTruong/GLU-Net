import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.pixel_wise_mapping import remap_using_flow_fields
from utils_training.multiscale_loss import multiscaleEPE, realEPE, sparse_max_pool
from matplotlib import pyplot as plt


def pre_process_data(source_img, target_img, device):
    '''
    Pre-processes source and target images before passing it to the network
    :param source_img: Torch tensor Bx3xHxW
    :param target_img: Torch tensor Bx3xHxW
    :param device: cpu or gpu
    :return:
    source_img_copy: Torch tensor Bx3xHxW, source image scaled to 0-1 and mean-centered and normalized
                     using mean and standard deviation of ImageNet
    target_img_copy: Torch tensor Bx3xHxW, target image scaled to 0-1 and mean-centered and normalized
                     using mean and standard deviation of ImageNet
    source_img_256: Torch tensor Bx3x256x256, source image rescaled to 256x256, scaled to 0-1 and mean-centered and normalized
                    using mean and standard deviation of ImageNet
    target_img_256: Torch tensor Bx3x256x256, target image rescaled to 256x256, scaled to 0-1 and mean-centered and normalized
                    using mean and standard deviation of ImageNet
    '''
    # img has shape bx3xhxw
    b, _, h_scale, w_scale = target_img.shape
    mean_vector = np.array([0.485, 0.456, 0.406])
    std_vector = np.array([0.229, 0.224, 0.225])

    # original resolution
    source_img_copy = source_img.float().to(device).div(255.0)
    target_img_copy = target_img.float().to(device).div(255.0)
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

    return source_img_copy, target_img_copy, source_img_256, target_img_256


def plot_during_training(save_path, epoch, batch, apply_mask,
                         h_original, w_original, h_256, w_256,
                         source_image, target_image, source_image_256, target_image_256, div_flow,
                         flow_gt_original, flow_gt_256, output_net,  output_net_256, mask=None, mask_256=None):
    # resolution original
    flow_est_original = F.interpolate(output_net, (h_original, w_original),
                                      mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_target_x = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)
    image_1 = (source_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2 = (target_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt = remap_using_flow_fields(image_1.numpy(),
                                          flow_target_x.cpu().numpy(),
                                          flow_target_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())

    # resolution 256x256
    flow_est_256 = F.interpolate(output_net_256, (h_256, w_256),
                                 mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_target_x_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x_256.shape == flow_target_x_256.shape

    image_1_256 = (source_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2_256 = (target_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt_256 = remap_using_flow_fields(image_1_256.numpy(),
                                              flow_target_x_256.cpu().numpy(),
                                              flow_target_y_256.cpu().numpy())
    remapped_est_256 = remap_using_flow_fields(image_1_256.numpy(), flow_est_x_256.cpu().numpy(),
                                               flow_est_y_256.cpu().numpy())

    fig, axis = plt.subplots(2, 5, figsize=(20, 20))
    axis[0][0].imshow(image_1.numpy())
    axis[0][0].set_title("original reso: \nsource image")
    axis[0][1].imshow(image_2.numpy())
    axis[0][1].set_title("original reso: \ntarget image")
    if apply_mask:
        mask = mask.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask = np.ones((h_original, w_original))
    axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
    axis[0][2].set_title("original reso: \nmask applied during training")
    axis[0][3].imshow(remapped_gt)
    axis[0][3].set_title("original reso : \nsource remapped with ground truth")
    axis[0][4].imshow(remapped_est)
    axis[0][4].set_title("original reso: \nsource remapped with network")
    axis[1][0].imshow(image_1_256.numpy())
    axis[1][0].set_title("reso 256: \nsource image")
    axis[1][1].imshow(image_2_256.numpy())
    axis[1][1].set_title("reso 256:\ntarget image")
    if apply_mask:
        mask_256 = mask_256.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask_256 = np.ones((h_256, w_256))
    axis[1][2].imshow(mask_256, vmin=0.0, vmax=1.0)
    axis[1][2].set_title("reso 256: \nmask applied during training")
    axis[1][3].imshow(remapped_gt_256)
    axis[1][3].set_title("reso 256: \nsource remapped with ground truth")
    axis[1][4].imshow(remapped_est_256)
    axis[1][4].set_title("reso 256: \nsource remapped with network")
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                bbox_inches='tight')
    plt.close(fig)


def train_epoch(net,
                optimizer,
                train_loader,
                device,
                epoch,
                train_writer,
                div_flow=1.0,
                save_path=None,
                loss_grid_weights=None,
                apply_mask=False,
                robust_L1_loss=False,
                sparse=False):
    """
    Training epoch script
    Args:
        net: model architecture
        optimizer: optimizer to be used for traninig `net`
        train_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        train_writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
        robust_L1_loss: bool on the loss to use
        sparse: bool on sparsity of ground truth flow field
    Output:
        running_total_loss: total training loss

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.
    """
    n_iter = epoch*len(train_loader)
    net.train()

    running_total_loss = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()

        # pre-process the data
        source_image, target_image, source_image_256, target_image_256 = pre_process_data(mini_batch['source_image'],
                                                                                          mini_batch['target_image'],
                                                                                          device=device)

        output_net_256, output_net_original = net(target_image, source_image, target_image_256, source_image_256)

        # At original resolution
        flow_gt_original = mini_batch['flow_map'].to(device)
        if flow_gt_original.shape[1] != 2:
            # shape is bxhxwx2
            flow_gt_original = flow_gt_original.permute(0,3,1,2)
        bs, _, h_original, w_original = flow_gt_original.shape
        weights_original = loss_grid_weights[-len(output_net_original):]

        # at 256x256 resolution, b, _, 256, 256
        if sparse:
            flow_gt_256 = sparse_max_pool(flow_gt_original, (256, 256))
        else:
            flow_gt_256 = F.interpolate(flow_gt_original, (256, 256),
                                        mode='bilinear', align_corners=False)
        flow_gt_256[:,0,:,:] *= 256.0/float(w_original)
        flow_gt_256[:,1,:,:] *= 256.0/float(h_original)
        bs, _, h_256, w_256 = flow_gt_256.shape
        weights_256 = loss_grid_weights[:len(output_net_256)]

        # calculate the loss, depending on mask conditions
        if apply_mask:
            mask = mini_batch['correspondence_mask'].to(device)  # bxhxw, torch.uint8
            Loss = multiscaleEPE(output_net_original, flow_gt_original, weights=weights_original, sparse=sparse,
                                 mean=False, mask=mask, robust_L1_loss=robust_L1_loss)
            if sparse:
                mask_256 = sparse_max_pool(mask.unsqueeze(1).float(), (256, 256)).squeeze(1).byte() # bx256x256
            else:
                mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                           align_corners=False).squeeze(1).byte() # bx256x256
            Loss += multiscaleEPE(output_net_256, flow_gt_256, weights=weights_256, sparse=sparse,
                                 mean=False, mask=mask_256, robust_L1_loss=robust_L1_loss)
        else:
            Loss = multiscaleEPE(output_net_original, flow_gt_original, weights=weights_original, sparse=False,
                                 mean=False, robust_L1_loss=robust_L1_loss)
            Loss += multiscaleEPE(output_net_256, flow_gt_256, weights=weights_256, sparse=False,
                                 mean=False, robust_L1_loss=robust_L1_loss)

        Loss.backward()
        optimizer.step()

        if i < 4:  # log first output of first batches
            if apply_mask:
                plot_during_training(save_path, epoch, i, apply_mask,
                                     h_original, w_original, h_256, w_256,
                                     source_image, target_image, source_image_256, target_image_256, div_flow,
                                     flow_gt_original, flow_gt_256, output_net=output_net_original[-1],
                                     output_net_256=output_net_256[-1],
                                     mask=mask, mask_256=mask_256)
            else:
                plot_during_training(save_path, epoch, i, apply_mask,
                                     h_original, w_original, h_256, w_256,
                                     source_image, target_image, source_image_256, target_image_256, div_flow,
                                     flow_gt_original, flow_gt_256, output_net=output_net_original[-1],
                                     output_net_256=output_net_256[-1])

        running_total_loss += Loss.item()
        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        n_iter += 1
        pbar.set_description(
                'training: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                             Loss.item()))
    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(net,
                   val_loader,
                   device,
                   epoch,
                   save_path,
                   div_flow=1,
                   loss_grid_weights=None,
                   apply_mask=False,
                   sparse=False,
                   robust_L1_loss=False):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        train_writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
        robust_L1_loss: bool on the loss to use
        sparse: bool on sparsity of ground truth flow field
    Output:
        running_total_loss: total validation loss,
        EPE_0, EPE_1, EPE_2, EPE_3: EPEs corresponding to each level of the network (after upsampling
        the estimated flow to original resolution and scaling it properly to compare to ground truth).

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.

    """

    net.eval()
    if loss_grid_weights is None:
        loss_grid_weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        EPE_array = torch.zeros([len(loss_grid_weights), len(val_loader)], dtype=torch.float32, device=device)
        for i, mini_batch in pbar:
            source_image, target_image, source_image_256, target_image_256 = pre_process_data(
                mini_batch['source_image'],
                mini_batch['target_image'],
                device=device)

            output_net_256, output_net_original = net(target_image, source_image, target_image_256, source_image_256)

            # at original size
            flow_gt_original = mini_batch['flow_map'].to(device)
            if flow_gt_original.shape[1] != 2:
                # shape is bxhxwx2
                flow_gt_original = flow_gt_original.permute(0, 3, 1, 2)
            bs, _, h_original, w_original = flow_gt_original.shape
            mask_gt = mini_batch['correspondence_mask'].to(device)
            weights_original = loss_grid_weights[-len(output_net_original):]

            # at 256x256 resolution, b, _, 256, 256
            if sparse:
                flow_gt_256 = sparse_max_pool(flow_gt_original, (256, 256))
            else:
                flow_gt_256 = F.interpolate(flow_gt_original, (256, 256),
                                            mode='bilinear', align_corners=False)
            flow_gt_256[:, 0, :, :] *= 256.0 / float(w_original)
            flow_gt_256[:, 1, :, :] *= 256.0 / float(h_original)
            bs, _, h_256, w_256 = flow_gt_256.shape
            weights_256 = loss_grid_weights[:len(output_net_256)]

            if apply_mask:
                mask = mini_batch['correspondence_mask'].to(device)  # bxhxw, torch.uint8
                Loss = multiscaleEPE(output_net_original, flow_gt_original,
                                     weights=weights_original, sparse=sparse,
                                     mean=False, mask=mask, robust_L1_loss=robust_L1_loss)
                if sparse:
                    mask_256 = sparse_max_pool(mask.unsqueeze(1).float(), (256, 256)).squeeze(1).byte()  # bx256x256
                else:
                    mask_256 = F.interpolate(mask.unsqueeze(1).float(), (256, 256), mode='bilinear',
                                             align_corners=False).squeeze(1).byte()  # bx256x256
                Loss += multiscaleEPE(output_net_256, flow_gt_256, weights=weights_256,
                                      sparse=sparse,
                                      mean=False, mask=mask_256, robust_L1_loss=robust_L1_loss)
            else:
                Loss = multiscaleEPE(output_net_original, flow_gt_original,
                                     weights=weights_original, sparse=False,
                                     mean=False, robust_L1_loss=robust_L1_loss)
                Loss += multiscaleEPE(output_net_256, flow_gt_256, weights=weights_256, sparse=False,
                                      mean=False, robust_L1_loss=robust_L1_loss)

            # calculating the validation EPE
            for index_reso_original in range(len(output_net_original)):
                EPE = div_flow * realEPE(output_net_original[-(index_reso_original+1)], flow_gt_original, mask_gt, sparse=sparse)
                EPE_array[index_reso_original, i] = EPE

            for index_reso_256 in range(len(output_net_256)):
                EPE = div_flow * realEPE(output_net_256[-(index_reso_256+1)], flow_gt_original, mask_gt,
                                        ratio_x=float(w_original) / float(256.0),
                                        ratio_y=float(h_original) / float(256.0),
                                        sparse=sparse)
                EPE_array[(len(output_net_original) + index_reso_256), i] = EPE
            # must be both in shape Bx2xHxW

            if i < 4:
                # log first output of first batches
                if apply_mask:
                    plot_during_training(save_path, epoch, i, apply_mask,
                                         h_original, w_original, h_256, w_256,
                                         source_image, target_image, source_image_256, target_image_256, div_flow,
                                         flow_gt_original, flow_gt_256, output_net=output_net_original[-1],
                                         output_net_256=output_net_256[-1],
                                         mask=mask, mask_256=mask_256)
                else:
                    plot_during_training(save_path, epoch, i, apply_mask,
                                         h_original, w_original, h_256, w_256,
                                         source_image, target_image, source_image_256, target_image_256, div_flow,
                                         flow_gt_original, flow_gt_256, output_net=output_net_original[-1],
                                         output_net_256=output_net_256[-1])

            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                             Loss.item()))
        mean_epe = torch.mean(EPE_array, dim=1)

    return running_total_loss / len(val_loader), mean_epe[0].item(), mean_epe[1].item(), mean_epe[2].item(), mean_epe[3].item()
