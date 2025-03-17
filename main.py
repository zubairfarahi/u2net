import torch
from config.config import load_configuration
from train import train_model


import os
import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from data_augmentation import (
    RandomBackground, RandomRotation, RandomPerspective, RandomEqualize,
    RandomHorizontalFlip, RandomGrayscale, RandomCrop, RandomPadding,
    RandomGaussianBlur, RandomHorizontalMotionBlur, RandomVerticalMotionBlur,
    RandomPosterize, RandomSharpness, RandomColorJitter, RandomPixelation,
    RandomJpegCompression)
from data_loader import SalObjDataset, RescaleT, ToTensor
from model import U2NET, U2NETP
from torchvision.transforms import v2
import torch.distributed as dist
import torch.nn as nn

from utils import count_trainable_params

def get_label(name_list, data_dir, label_dir):
    label_ext = '.png'
    dataset = []
    for img_path in name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]

        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        dataset.append(data_dir + label_dir + imidx + label_ext)
    return dataset

def train_val_data_loader(config):
    """
    Prepare data loaders for training and validation datasets.
    """
    image_ext = '.jpg'

    all_img_name_list = glob.glob(os.path.join(config['data_train_path'], 'originals', '*' + image_ext))
    all_val_img_list = glob.glob(os.path.join(config['data_val_path'], 'originals', '*' + image_ext))

    tra_lbl_name_list = get_label(all_img_name_list, config['data_train_path'], 'masks/')
    val_lbl_name_list = get_label(all_val_img_list, config['data_val_path'], 'masks/')

    print("---")
    print("train images: ", len(all_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    print("---")
    print("validation images: ", len(all_val_img_list))
    print("validation labels: ", len(val_lbl_name_list))
    print("---")

    train_salobj_dataset = SalObjDataset(
        img_name_list=all_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomPadding(config['debug_mode'], config['debug_path']),
            RescaleT(320),
            RandomBackground(config['debug_mode'], config['debug_path']),
            RandomHorizontalFlip(config['debug_mode'], config['debug_path']),
            v2.RandomChoice(transforms=[RandomPerspective(config['debug_mode'], config['debug_path'], probability=0.4),RandomRotation(45, config['debug_mode'], config['debug_path'], probability=0.6)]),
            v2.RandomChoice(transforms=[RandomPosterize(config['debug_mode'], config['debug_path']),RandomColorJitter(config['debug_mode'], config['debug_path'], probability=0.8), RandomEqualize(config['debug_mode'], config['debug_path'], probability=0.6)]),
            v2.RandomChoice(transforms=[RandomGaussianBlur(config['debug_mode'], config['debug_path']),RandomSharpness(config['debug_mode'], config['debug_path'], probability=0.7)]),
            v2.RandomChoice(transforms=[RandomJpegCompression(config['debug_mode'], config['debug_path'])]),
            RandomHorizontalMotionBlur(config['debug_mode'], config['debug_path']), 
            RandomVerticalMotionBlur(config['debug_mode'], config['debug_path']),
            RandomPixelation(config['debug_mode'], config['debug_path']),
            RandomGrayscale(config['debug_mode'], config['debug_path']),
            RandomCrop(288, config['debug_mode'], config['debug_path'], probability=0.2),
            ToTensor()
        ])
    )

    val_salobj_dataset = SalObjDataset(
        img_name_list=all_val_img_list,
        lbl_name_list=val_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            ToTensor()
        ])
    )
    train_salobj_dataloader = DataLoader(train_salobj_dataset, batch_size=config['batch_size_train'], shuffle=True, num_workers=1)
    val_salobj_dataloader = DataLoader(val_salobj_dataset, batch_size=config['batch_size_train'], shuffle=False, num_workers=1)

    return train_salobj_dataloader, val_salobj_dataloader


# def initialize_model(config, device):
#     """
#     Initialize the model based on type and load pretrained weights if available.
#     """
#     print(torch.cuda.is_available())

#     if config['distributed']:
#         dist.init_process_group(backend='nccl')

#     if config['model_type'] == 'u2net':
#         net = U2NET(3, 1)
#     elif config['model_type'] == 'u2netp':
#         net = U2NETP(3, 1)

#     if torch.cuda.is_available():
#         net.to(device)

#     if config['distributed']:
#         net = nn.parallel.DistributedDataParallel(net, device_ids=[device])

#     return net

def initialize_model(config, device):
    """
    Initialize the model based on type and load pretrained weights if available.
    """
    print(torch.cuda.is_available())
    if config['model_type'] == 'u2net':
        net = U2NET(3, 1)
    elif config['model_type'] == 'u2netp':
        net = U2NETP(3, 1)
    if torch.cuda.is_available():
        net.to(device)
    
    # fine tune complet model
    
    if config['pretrained_model_path'] != "" and os.path.isfile(config['pretrained_model_path']):
        print("Loading pretrained weights from", config['pretrained_model_path'])
        net.load_state_dict(torch.load(config['pretrained_model_path'], map_location=torch.device('cuda')), strict=False)
        # net.load_state_dict(torch.load(config['pretrained_model_path'], map_location=device))
        

    # Freezeing some layers
    
    # if config['cuda'] == 2:    
    #     print(f'Before freezing, the number of trainable parameters: {count_trainable_params(net)}')
    #     for stage in [net.stage1, net.stage2]:
    #         for param in stage.parameters():
    #             param.requires_grad = False

    #         print(f'After freezing, the number of trainable parameters: {count_trainable_params(net)}')
    
    #     return net
    # else:
    #     return net
    # if config['cuda'] == 2:    
    #     print(f'Before freezing, the number of trainable parameters: {count_trainable_params(net)}')
    #     for stage in [net.stage4, net.stage5]:
    #         for param in stage.parameters():
    #             param.requires_grad = False

    #         print(f'After freezing, the number of trainable parameters: {count_trainable_params(net)}')
    
    #     return net
    
    if config['cuda'] == 2:    
        print(f'Before freezing, the number of trainable parameters: {count_trainable_params(net)}')
        for stage in [net.stage5]:
            for param in stage.parameters():
                param.requires_grad = False

            print(f'After freezing, the number of trainable parameters: {count_trainable_params(net)}')
    
        return net
    else:
        return net


def main():
    config = load_configuration()
    torch.cuda.set_device(config['cuda'])
    print("Current device:", torch.cuda.current_device())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = train_val_data_loader(config)
    net = initialize_model(config, device)

    train_model(net, train_loader, val_loader, config)

if __name__ == '__main__':
    main()
