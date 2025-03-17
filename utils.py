# Standard Library Imports
import os
import shutil
import re
import random

# Third-Party Imports
import numpy as np
import cv2
from PIL import Image, ImageFile
from torchvision.transforms import v2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime

BG_DIR = "/home/zfarahi/jupyter-base/u2net/training_data/set-bg"
backgrounds = os.listdir(BG_DIR)

# Color Range for Conversion to Black and White
lower_hue  = np.array([5,5,5])
upper_hue = np.array([255,255,255])

# Image Processing Functions

def to_black_white(img):
    """Convert an image to black and white based on HSV thresholding."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    img[mask > 0] = (255, 255, 255)
    return img

def resize(image, intended_size):
    """Resize an image to the intended size while maintaining aspect ratio."""
    to_height, to_width = intended_size
    to_ratio = to_height / to_width
    cutted_image = cut_by_ratio(image, to_ratio)
    resized_image = cv2.resize(cutted_image, (to_width, to_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def cut_by_ratio(image, response_ratio):
    """Cut an image to match a specific aspect ratio."""
    height, width, _ = image.shape
    image_ratio = height / width

    if image_ratio > response_ratio:
        cut_height = int((height - width * response_ratio) / 2)
        image = image[cut_height:height-cut_height, :]
    elif image_ratio < response_ratio:
        cut_width = int((width - height / response_ratio) / 2)
        image = image[:, cut_width:width-cut_width]

    return image

def create_background(rgb_color, width, height):
    background = Image.new('RGB', (width, height), rgb_color)
    return background

def find_dominant_color(image, k=3):
    # Load the image
    # image = Image.open(image_path)
    w, h = image.size
    image = image.resize((150, 150)) # optional, to reduce time
    # Convert image to np.array
    pixels = np.array(image)
    # Reshape the array to be a list of RGB values
    pixels = pixels.reshape(-1, 3)
    # Perform k-means clustering to find the most dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get the cluster labels for each pixel in the image
    labels = kmeans.labels_

    # Find the most common cluster label
    most_common_label = np.bincount(labels).argmax()

    # The dominant color is the cluster center with the most members
    dominant_color = kmeans.cluster_centers_[most_common_label]
    dominant_colors = tuple(map(int, dominant_color))
    background = create_background(dominant_colors, w, h)
    return background



# Composite Image Creation
def combined(image, matte, bg):
    """Combine image and matte with a background."""
    transparent = Image.new(bg.mode, bg.size)
    transparent.paste(image, (0, 0))
    mask = matte.convert("L")
    transparent = transparent.convert("RGBA")
    transparent.putalpha(mask)
    bg.paste(transparent, (0, 0), transparent)
    bg = bg.convert('RGB')
    return bg

def random_background(image, mask):
    nr1 = random.randint(0, len(backgrounds)-1)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    bg1 = Image.open(os.path.join(BG_DIR,backgrounds[nr1]))
    
    bg1 = v2.CenterCrop(size=(320,320))(bg1)
    image = combined(image, mask, bg1)
    return image

def extract_losses(filename):
    """Extract loss values from a filename."""
    match = re.match(r'.*val_loss_(\d+\.\d+)_val_tar_(\d+\.\d+)\.pth', filename)
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)

def get_files_and_losses(list_of_filenames):
    """Get a list of files and their associated losses."""
    files_losses = [(filename, *extract_losses(filename)) for filename in list_of_filenames if extract_losses(filename)[0] is not None]
    return files_losses

def print_top_3_min_loss(files_losses):
    """Print and return the top 3 files with the minimum loss."""
    sorted_files = sorted(files_losses, key=lambda x: x[1] + x[2])[:3]
    for filename, _, _ in sorted_files:
        print(f"Moving {filename}...")
    return [filename for filename, _, _ in sorted_files]

def move_files(files, source_dir, dest_dir):
    """Move files from source to destination directory."""
    for filename in files:
        shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))

def top_3_min_loss(filenames, source_directory, destination_directory):
    """Identify and move the top 3 models with minimum loss."""
    files_losses = get_files_and_losses(filenames)
    if files_losses:
        top_3_files = print_top_3_min_loss(files_losses)
        move_files(top_3_files, source_directory, destination_directory)
    else:
        print("No files found or no valid losses extracted.")

def get_lr(optimizer):
    """
    Get the current learning rate from the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def generate_dir_name(instance_number):
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    dir_name = f"instance {instance_number} {current_time}"
    return dir_name

def freeze_and_unfreeze_layers(net, unfreeze_stages=[]):
    """
    Freeze all layers initially, and then unfreeze specified stages.
    
    Args:
        net (nn.Module): The model to freeze/unfreeze layers.
        unfreeze_stages (list): A list of stages to unfreeze.
    """
    # Freeze all layers initially
    for param in net.parameters():
        param.requires_grad = False
    
    # Unfreeze specified stages
    for stage in unfreeze_stages:
        for param in getattr(net, stage).parameters():
            param.requires_grad = True
    
    print(f"After freezing/unfreezing, the number of trainable parameters: {count_trainable_params(net)}")
    return net

def count_trainable_params(model):
    """
    Count the number of trainable parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def visualize_filters(module, writer, epoch, prefix=''):

    for name, sub_module in module.named_children():
        new_prefix = f'{prefix}{name}.' if prefix else name

        if isinstance(sub_module, nn.Conv2d):
            filters = sub_module.weight.data.cpu().numpy()
            num_filters = filters.shape[0]

            fig, axs = plt.subplots(1, num_filters, figsize=(20, 5))
            if num_filters == 1:
                axs = [axs]

            for i, ax in enumerate(axs):
                ax.imshow(filters[i, 0], cmap='gray')
                ax.axis('off')
            
            writer.add_figure(f"{new_prefix}/filters", fig, global_step=epoch)
            plt.close(fig)
        else:
            # Recursively call this function for non-Conv2d modules
            visualize_filters(sub_module, writer, epoch, new_prefix)

