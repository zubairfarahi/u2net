import torch.nn as nn
import torch
import numpy as np
import cv2
bce_loss = nn.BCELoss(reduction='mean')
bce_logits_loss = nn.BCEWithLogitsLoss(reduction='mean')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss

def muti_bce_logits_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_logits_loss(d0,labels_v)
	loss1 = bce_logits_loss(d1,labels_v)
	loss2 = bce_logits_loss(d2,labels_v)
	loss3 = bce_logits_loss(d3,labels_v)
	loss4 = bce_logits_loss(d4,labels_v)
	loss5 = bce_logits_loss(d5,labels_v)
	loss6 = bce_logits_loss(d6,labels_v)
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

     
	return loss0, loss

def threshold_image(image, threshold=0.5):
    thresholded_image = (image > threshold).type(torch.int)
    return thresholded_image

def custom_loss_0(d0, d1, d2, d3, d4, d5, d6, labels_v):
    batch_size = len(labels_v)

    # Calculate BCE loss for each layer
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)


    # Calculate incorrect pixels using the custom threshold
    thresholded_d0 = threshold_image(d0)
    thresholded_label = threshold_image(labels_v)
    
    incorrect_pixels0 = torch.sum(thresholded_d0 != thresholded_label)
   
    # Sum up the incorrect pixels across all layers
    total_incorrect_pixels =  torch.log(incorrect_pixels0 / batch_size)
    
    # Combine the individual losses
    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + total_incorrect_pixels
    
    # Print information for debugging or monitoring
    print("BCE Losses: l0: {:.3f}, l1: {:.3f}, l2: {:.3f}, l3: {:.3f}, l4: {:.3f}, l5: {:.3f}, l6: {:.3f}".format(
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(),
        loss4.data.item(), loss5.data.item(), loss6.data.item()))
    print("Incorrect pixels in batch: {:.3f}".format(
        total_incorrect_pixels.item()))

    return loss0, total_loss
