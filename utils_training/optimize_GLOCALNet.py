import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.pixel_wise_mapping import remap_using_flow_fields
from utils_training.multiscale_loss import multiscaleEPE, realEPE
from matplotlib import pyplot as plt


def plot_during_training(save_path, epoch, batch, h, w, mini_batch, div_flow, flow_gt, output_net, mask_gt=None):
    flow_est = F.interpolate(output_net, (h, w), mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_target_x = div_flow * flow_gt.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
    image_1 = (mini_batch['source_image'][0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt = remap_using_flow_fields(image_1.numpy(),
                                          flow_target_x.cpu().numpy(),
                                          flow_target_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())

    fig, axis = plt.subplots(2, 3, figsize=(20, 20))
    axis[0][0].imshow(image_1.numpy())
    axis[0][0].set_title("source image")
    axis[0][1].imshow((mini_batch['target_image'][0].cpu() * std_values +
                       mean_values).clamp(0, 1).permute(1, 2, 0).numpy())
    axis[0][1].set_title("target image")
    if mask_gt is not None:
        mask = mask_gt.cpu().numpy()[0].astype(np.float32)
    else:
        mask = np.ones((h, w))
    axis[0][2].imshow(mask, vmin=0, vmax=1)
    axis[0][2].set_title('mask applied during training')
    axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
    axis[1][0].imshow(remapped_gt)
    axis[1][0].set_title("source remapped with ground truth")
    axis[1][1].imshow(remapped_est)
    axis[1][1].set_title("source remapped with network")
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
                apply_mask=False):
    """
    Training epoch script
    Args:
        net: model architecture
        optimizer: optimizer to be used for traninig `net`
        train_loader: dataloader
        device: `cpu` or `gpu`
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        train_writer: for tensorboard
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
    Output:
        running_total_loss: total training loss

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.
    """
    n_iter = epoch*len(train_loader)
    # everywhere when they say flow it is actuallt mapping
    net.train()
    running_total_loss = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        flow_gt = mini_batch['flow_map'].to(device)
        if flow_gt.shape[1] != 2:
            flow_gt.permute(0,3,1,2)
        bs, _, h, w = flow_gt.shape

        output_net = net(mini_batch['target_image'].to(device),
                         mini_batch['source_image'].to(device))

        if apply_mask:
            mask_gt = mini_batch['correspondence_mask'].to(device)  # bxhxw, torch.uint8
            if mask_gt.shape[1] != h or mask_gt.shape[2] != w:
                # mask_gt does not have the proper shape
                mask_gt = F.interpolate(mask_gt.float().unsqueeze(1), (h, w), mode='bilinear',
                                        align_corners=False).squeeze(1).byte()  #bxhxw
                mask_gt = mask_gt.bool() if float(torch.__version__[:3]) >= 1.1 else mask_gt.byte()

            Loss = multiscaleEPE(output_net, flow_gt, weights=loss_grid_weights,
                                 sparse=False, mean=False, mask=mask_gt)
        else:
            Loss = multiscaleEPE(output_net, flow_gt, weights=loss_grid_weights, sparse=False,
                                 mean=False)
        Loss.backward()
        optimizer.step()

        if i < 4:
            # log first output of first batches
            if apply_mask:
                plot_during_training(save_path, epoch, i, h, w, mini_batch, div_flow, flow_gt, output_net[-1],  mask_gt)
            else:
                plot_during_training(save_path, epoch, i, h, w, mini_batch, div_flow, flow_gt, output_net[-1])

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
                   apply_mask=False):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
        device: `cpu` or `gpu`
        epoch: epoch number for plotting
        div_flow: multiplicative factor to apply to the estimated flow
        save_path: path to folder to save the plots
        loss_grid_weights: weight coefficients for each level of the feature pyramid
        apply_mask: bool on whether or not to apply a mask for the loss
    Output:
        running_total_loss: total validation loss,
        mean_EPE: EPE corresponding to highest level of the network (after upsampling
        the estimated flow to original resolution to compare to ground truth).

        here output of the network at every level is flow interpolated but not scaled.
        we only use the ground truth flow as highest resolution and downsample it without scaling.

    """

    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        aepe_array=[]
        for i, mini_batch in pbar:
            output_net = net(mini_batch['target_image'].to(device),
                             mini_batch['source_image'].to(device))

            flow_gt = mini_batch['flow_map'].to(device)
            if flow_gt.shape[1] != 2:
                flow_gt.permute(0, 3, 1, 2)
            bs, _, h, w = flow_gt.shape
            mask_gt = mini_batch['correspondence_mask'].to(device)  # bxhxw, torch.uint8

            if mask_gt.shape[1] != h or mask_gt.shape[2] != w:
                # mask_gt does not have the proper shape
                mask_gt = F.interpolate(mask_gt.float().unsqueeze(1), (h, w), mode='bilinear',
                                        align_corners=False).squeeze(1).byte()  # bxhxw
                mask_gt = mask_gt.bool() if float(torch.__version__[:3]) >= 1.1 else mask_gt.byte()

            if apply_mask:
                Loss = multiscaleEPE(output_net, flow_gt, weights=loss_grid_weights,
                                     sparse=False, mean=False, mask=mask_gt)
            else:
                Loss = multiscaleEPE(output_net, flow_gt, weights=loss_grid_weights,
                                     sparse=False, mean=False)

            EPE = div_flow * realEPE(output_net[-1], flow_gt, mask_gt, sparse=False)
            aepe_array.append(EPE.item())
            # must be both in shape Bx2xHxW

            if i < 3:
                # log first output of first batches
                if apply_mask:
                    plot_during_training(save_path, epoch, i, h, w, mini_batch, div_flow, flow_gt, output_net[-1],
                                         mask_gt)
                else:
                    plot_during_training(save_path, epoch, i, h, w, mini_batch, div_flow, flow_gt, output_net[-1])

            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                             Loss.item()))
        mean_epe = np.mean(aepe_array)

    return running_total_loss / len(val_loader), mean_epe
