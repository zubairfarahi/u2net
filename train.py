# Imports for system and file handling
import time

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid
import os

# Local module imports
from scheduler import instance_1, instance_2, instance_3, instance_0
from utils import visualize_filters, top_3_min_loss, get_lr, generate_dir_name, freeze_and_unfreeze_layers, count_trainable_params
from multi_model_tests import process_image, load_image, save_combined_output
from loss import custom_loss_0, muti_bce_logits_loss_fusion, muti_bce_loss_fusion
from torch.utils.tensorboard import SummaryWriter

# torch.manual_seed(123)





def validate(net, dataloader, batch_size_train, config):
    net.eval()

    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    ite_num = 0

    for i, data in enumerate(dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)

        with torch.no_grad():
            outputs = net(inputs_v)
            if config['cuda'] == 0:
                loss2, loss = custom_loss_0(*outputs, labels_v)
            elif config['cuda'] == 1:
                loss2, loss = custom_loss_0(*outputs, labels_v)
            elif config['cuda'] == 2:
                loss2, loss = custom_loss_0(*outputs, labels_v)
            elif config['cuda'] == 3:
                loss2, loss = muti_bce_loss_fusion(*outputs, labels_v)
            
            running_loss += loss.item()
            running_tar_loss += loss2.item()
        print("[batch: %5d/%5d, ite: %d] validation loss: %3f, tar: %3f \n" % ((i + 1) * batch_size_train, len(dataloader.dataset), ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

    return running_loss / ite_num4val, running_tar_loss / ite_num4val


def perform_testing(net, epoch, min_val_loss, min_val_loss_filename, config):

    if min_val_loss_filename:            
        start_time = time.time()
        
        net.load_state_dict(torch.load(min_val_loss_filename))
        print(f"Testing with best model of this interval: {min_val_loss_filename}")
        
        test_loader, img_name_list = load_image(config["input"])

        filename_without_extension = f"epoch_{epoch}_loss_{min_val_loss:.6f}"
        output_dir = os.path.join(config['output'], filename_without_extension)

        os.makedirs(output_dir, exist_ok=True)
        for j, data_test in enumerate(test_loader):
            image = data_test['image']
            original_label = data_test.get('label')
            pred, mean = process_image(net, image)
            save_combined_output(img_name_list[j], pred, min_val_loss_filename, original_label, output_dir)
        
        print("Testing Time:", int(time.time() - start_time), "seconds\n")

    return float('inf'), ""

def train_model(net, train_loader, val_loader, config):
    """
    Main training loop including validation.
    """
    print(f"lr: {config['lr']}, betas: {config['betas']}, eps: {config['eps']}, weight: {config['weight_decay']}")
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], betas=config['betas'], eps=config['eps'], weight_decay=config['weight_decay'])
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'], betas=config['betas'], eps=config['eps'], weight_decay=config['weight_decay'])
    # optimizer = optim.SGD(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    if config['cuda'] == 0:
        scheduler = instance_0(optimizer, len(train_loader), config['epoch_num'])
    elif config['cuda'] == 1:
        scheduler = instance_1(optimizer)
    elif config['cuda'] == 2:
        scheduler = instance_2(optimizer, len(train_loader), config['epoch_num'])
    elif config['cuda'] == 3:
        scheduler = instance_3(optimizer, len(train_loader), config['epoch_num'])

    

    writer_loss = SummaryWriter(log_dir=f"runs/{generate_dir_name(config['cuda'])}/loss")
    writer_loss.add_graph(net, next(iter(train_loader))['image'].cuda())
    

    # writer_filters_and_featuremaps = SummaryWriter(log_dir=f"runs/{generate_dir_name(config['cuda'])}/filters_and_featuremaps")
    

    
    # num_stages = 6
    # epochs_per_stage = 100

    # stage_lrs = {
    #     'stage1': 0.001,
    #     'stage2': 0.0001,
    #     'stage3': 0.00001,
    # }

    # net = freeze_and_unfreeze_layers(net, [])
    # unfreeze_stages = []
    
    # # Set the initial learning rates for all layers
    # for name, param in net.named_parameters():
    #     if any(stage in name for stage in ['stage4', 'stage5', 'stage6']):
    #         param.requires_grad = True
    #         param.lr = config['lr']
    #     else:
    #         param.requires_grad = False
    
    track_loss = []

    min_val_loss = float('inf')
    min_val_loss_filename = ""
    # epoch_counter = 0
    # interval = 25
    iters = len(train_loader)
    for epoch in range(config['epoch_num']):
        ite_num = 0
        ite_num4val = 0
        running_loss = 0.0
        running_tar_loss = 0.0
        
        # for g in optimizer.param_groups:
        #     print(g['lr'])
        
        # current_stage = (epoch // epochs_per_stage) + 1
        # if current_stage > num_stages:
        #     current_stage = num_stages

        # if epoch < 10:
        #     new_unfreeze_stages = [f'stage{stage}' for stage in range(4, 7)]
        # elif epoch < 20:
        #     new_unfreeze_stages = ['stage1'] + [f'stage{stage}' for stage in range(4, 7)]
        # elif epoch < 30:
        #     new_unfreeze_stages = ['stage1', 'stage2'] + [f'stage{stage}' for stage in range(4, 7)]
        # else:
        #     new_unfreeze_stages = [f'stage{stage}' for stage in range(1, 7)]

        # if new_unfreeze_stages != unfreeze_stages:
        #     unfreeze_stages = new_unfreeze_stages
        #     net = freeze_and_unfreeze_layers(net, unfreeze_stages)
            
        #     # param_groups = []
        #     # for name, param in net.named_parameters():
        #     #     stage_name = next((stage for stage in stage_lrs.keys() if stage in name), None)
        #     #     if stage_name is None:
        #     #         continue  # Skip layers that don't belong to any stage
        #     #     param_groups.append({'params': [param], 'lr': stage_lrs[stage_name]})

        #     # optimizer = optim.Adam(param_groups, betas=config['betas'], eps=config['eps'], weight_decay=config['weight_decay'])



        #     print(f'Unfreezing Layers: {unfreeze_stages}')
        #     # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'], betas=config['betas'], eps=config['eps'], weight_decay=config['weight_decay'])



        
        start_time = time.time()
        for i, data in enumerate(train_loader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)

            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            if config['cuda'] == 0:
                loss2, loss = custom_loss_0(d0, d1, d2, d3, d4, d5, d6, labels_v)
            elif config['cuda'] == 1:
                loss2, loss = custom_loss_0(d0, d1, d2, d3, d4, d5, d6, labels_v)
            elif config['cuda'] == 2:
                loss2, loss = custom_loss_0(d0, d1, d2, d3, d4, d5, d6, labels_v)
            elif config['cuda'] == 3:
                loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            
            running_loss += loss.item()
            running_tar_loss += loss2.item()
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f \n" % (
                epoch + 1, config['epoch_num'], (i + 1) * config['batch_size_train'], len(train_loader.dataset), ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if i % 100 == 0:
                writer_loss.add_image('train/inputs', make_grid(inputs_v.data.cpu(), nrow=8, normalize=True), epoch * len(train_loader) + i)
                writer_loss.add_image('train/labels', make_grid(labels_v.data.cpu(), nrow=8, normalize=True), epoch * len(train_loader) + i)

        print("Training Epoch Loop Time:", int(time.time()-start_time), "seconds\n")

        start_time = time.time()
        val_loss, val_tar_loss = validate(net, val_loader, config['batch_size_train'], config)

        
        torch.save(net.state_dict(), config['model_dir'] + 'u2net' + "_bce_epoch_%d_train_%3f_tar_%3f_val_loss_%3f_val_tar_%3f.pth" % (
            epoch+1, running_loss / ite_num4val, running_tar_loss / ite_num4val,  val_loss, val_tar_loss, ))


        track_loss.append('u2net' + "_bce_epoch_%d_train_%3f_tar_%3f_val_loss_%3f_val_tar_%3f.pth" % (
            epoch+1, running_loss / ite_num4val, running_tar_loss / ite_num4val, val_loss, val_tar_loss))

        print("Evaluation Epoch Loop Time:", int(time.time()-start_time), "seconds\n")
        
        if config['testing_flag'] and val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_filename = config['model_dir'] + 'u2net' + "_bce_epoch_%d_train_%3f_tar_%3f_val_loss_%3f_val_tar_%3f.pth" % (
            epoch+1, running_loss / ite_num4val, running_tar_loss / ite_num4val,  val_loss, val_tar_loss, )
            perform_testing(net, epoch, min_val_loss, min_val_loss_filename, config)
            # min_val_loss = float('inf')
            # min_val_loss_filename = ""
        
        # print(scheduler.get_lr()[0])
        
        writer_loss.add_scalar("Learning Rate", get_lr(optimizer), epoch)

        writer_loss.add_scalars("Loss", {"train_loss": running_loss / ite_num4val,
                                         "validation_loss": val_loss}, epoch+1)



        print(f"\nEpoch {epoch + 1}, LR={get_lr(optimizer)} Train Loss: {running_loss / ite_num4val}, Train Tar Loss: {running_tar_loss / ite_num4val},Val Loss: {val_loss}, Val Tar Loss: {val_tar_loss}\n")
        # if config["visualize_filters"]:
        #     visualize_filters(net, writer_filters_and_featuremaps, epoch=epoch)
        
    top_3_min_loss(track_loss, config['model_dir'], config['model_dest'])
    writer_loss.flush()
    writer_loss.close()
    # writer_filters_and_featuremaps.flush()
    # writer_filters_and_featuremaps.close()