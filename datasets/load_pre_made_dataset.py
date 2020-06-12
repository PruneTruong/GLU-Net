import os.path
import glob
from .listdataset import ListDataset
from datasets.util import split2list


def make_dataset(dir, split=None, dataset_name=None):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm  in folder images and
      [name]_flow.flo' in folder flow '''
    images = []
    '''
    if get_mapping:
        flow_dir = 'mapping'
        # flow_dir is actually mapping dir in that case, it is always normalised to [-1,1]
    '''
    flow_dir = 'flow'
    image_dir = 'images'

    # Make sure that the folders exist
    if not os.path.isdir(dir):
        raise ValueError("the training directory path that you indicated does not exist ! ")
    if not os.path.isdir(os.path.join(dir, flow_dir)):
        raise ValueError("the training directory path that you indicated does not contain the flow folder ! Check your directories.")
    if not os.path.isdir(os.path.join(dir, image_dir)):
        raise ValueError("the training directory path that you indicated does not contain the images folder ! Check your directories.")

    for flow_map in sorted(glob.glob(os.path.join(dir, flow_dir, '*_flow.flo'))):
        flow_map = os.path.join(flow_dir, os.path.basename(flow_map))
        root_filename = os.path.basename(flow_map)[:-9]
        img1 = os.path.join(image_dir, root_filename + '_img_1.jpg') # source image
        img2 = os.path.join(image_dir, root_filename + '_img_2.jpg') # target image
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            continue
        if dataset_name is not None:
            images.append([[os.path.join(dataset_name, img1),
                            os.path.join(dataset_name, img2)],
                           os.path.join(dataset_name, flow_map)])
        else:
            images.append([[img1, img2], flow_map])
    return split2list(images, split, default_split=0.97)


def PreMadeDataset(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                   co_transform=None, split=None):
    # that is only reading and loading the data and applying transformations to both datasets
    if isinstance(root, list):
        train_list=[]
        test_list=[]
        for sub_root in root:
            _, dataset_name = os.path.split(sub_root)
            sub_train_list, sub_test_list = make_dataset(sub_root, split, dataset_name=dataset_name)
            train_list.extend(sub_train_list)
            test_list.extend(sub_test_list)
        root = os.path.dirname(sub_root)
    else:
        train_list, test_list = make_dataset(root, split)
    print('Loading dataset at {}'.format(root))

    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform,
                                flow_transform=flow_transform, co_transform=co_transform)
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform,
                               flow_transform=flow_transform, co_transform=co_transform)

    return train_dataset, test_dataset
