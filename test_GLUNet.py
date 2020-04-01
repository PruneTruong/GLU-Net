import os
import torch
from models.models_compared import GLU_Net
import argparse
import imageio
from matplotlib import pyplot as plt
from utils.pixel_wise_mapping import remap_using_flow_fields
import cv2


def pad_to_same_shape(im1, im2):
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

    return im1, im2


parser = argparse.ArgumentParser(description='Test GLUNet on a pair of images')
parser.add_argument('--path_source_image', type=str,
                    help='Path to the source image.')
parser.add_argument('--path_target_image', type=str,
                    help='Path to the target image.')
parser.add_argument('--write_dir', type=str,
                    help='Directory where to write output figure.')
parser.add_argument('--pre_trained_models_dir', type=str, default='pre_trained_models/',
                    help='Directory containing the pre-trained-models.')
parser.add_argument('--pre_trained_model', type=str, default='DPED_CityScape_ADE',
                    help='Name of the pre-trained-model.')
args = parser.parse_args()

torch.cuda.empty_cache()
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu

if not os.path.exists(args.path_source_image):
    raise ValueError('The path to the source image you provide does not exist ! ')
if not os.path.exists(args.path_target_image):
    raise ValueError('The path to the target image you provide does not exist ! ')

if not os.path.isdir(args.write_dir):
    os.makedirs(args.write_dir)
try:
    source_image = imageio.imread(args.path_source_image)
    target_image = imageio.imread(args.path_target_image)
    source_image, target_image = pad_to_same_shape(source_image, target_image)
except:
    raise ValueError('It seems that the path for the images you provided does not work ! ')

with torch.no_grad():
    network = GLU_Net(path_pre_trained_models=args.pre_trained_models_dir,
                      model_type=args.pre_trained_model,
                      consensus_network=False,
                      cyclic_consistency=True,
                      iterative_refinement=True,
                      apply_flipping_condition=False)

    # convert numpy to torch tensor and put it in right shape
    source_image_ = torch.from_numpy(source_image).permute(2,0,1).unsqueeze(0)
    target_image_ = torch.from_numpy(target_image).permute(2,0,1).unsqueeze(0)
    # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
    # specific pre-processing (/255 and rescaling) are done within the function.

    # pass both images to the network, it will pre-process the images and ouput the estimated flow in dimension 1x2xHxW
    estimated_flow = network.estimate_flow(source_image_, target_image_, device, mode='channel_first')
    warped_source_image = remap_using_flow_fields(source_image, estimated_flow.squeeze()[0].cpu().numpy(),
                                                  estimated_flow.squeeze()[1].cpu().numpy())

    fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(30, 30))
    axis1.imshow(source_image)
    axis1.set_title('Source image')
    axis2.imshow(target_image)
    axis2.set_title('Target image')
    axis3.imshow(warped_source_image)
    axis3.set_title('Warped source image according to estimated flow by GLU-Net')
    fig.savefig(os.path.join(args.write_dir, 'Warped_source_image.png'),
                bbox_inches='tight')
    plt.close(fig)
