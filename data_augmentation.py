# Standard library imports
import random
import numbers
import numpy as np
import cv2
from collections.abc import Sequence
from skimage import io
from typing import List, Tuple

# Third-party imports
from PIL import Image, ImageOps, ImageFilter
import torch
from skimage import transform
from skimage.transform import rotate
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.transforms import functional as F, InterpolationMode

# Local imports
from utils import random_background , find_dominant_color, combined
from debug import DebuggerParams, Debugger

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:            
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
		debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
		Debugger(self.debug_mode, self.debug_path).save_image(debug_params, img)
		Debugger(self.debug_mode, self.debug_path).save_image(debug_params, lbl)
		return {'imidx':imidx, 'image':img,'label':lbl}


class RandomCrop(object):
    def __init__(self, output_size, debug_mode, debug_path, probability=0.5):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.debug_mode = debug_mode
        self.debug_path = debug_path
        self.probability = probability
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        w, h = image.size
       
        new_w, new_h = self.output_size

        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        if torch.rand(1) >= self.probability:
            image = image.crop((left, top, left + new_w, top + new_h))
            label = label.crop((left, top, left + new_w, top + new_h))
            
            image = image.resize((w, h), resample=Image.BICUBIC)
            label = label.resize((w, h), resample=Image.NEAREST)
        
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}

class RandomRotation(object):
    def __init__(self, degrees, debug_mode, debug_path, resample=False, expand=False, 
                 center=None, fill=None, probability=0.3):
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill
        self.degrees = (-degrees, degrees)
        self.probability = probability
        self.debug_mode = debug_mode
        self.debug_path = debug_path

    def __call__(self, sample):
        imidx, image, mask = sample['imidx'], sample['image'], sample['label']
        
        if torch.rand(1) >= self.probability:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            image = F.rotate(image, angle, self.resample, self.expand, self.center, 255)
            mask = F.rotate(mask, angle, self.resample, self.expand, self.center, 0)
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, mask)
        return {'imidx':imidx,'image':image, 'label':mask}


class RandomHorizontalFlip(object):
    def __init__(self, debug_mode, debug_path, probability=0.5):
        self.debug_mode = debug_mode
        self.debug_path = debug_path
        self.probability = probability
        
    def __call__(self, sample):
        imidx, image, mask = sample['imidx'], sample['image'], sample['label']
        
        if torch.rand(1) < self.probability:
            image = F.hflip(image)
            mask = F.hflip(mask)

        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, mask)

        return {'imidx': imidx, 'image': image, "label":mask}


class RandomPadding(object):
    def __init__(self, debug_mode, debug_path, padding=25, probability=0.5, padding_mode='constant', fill=0):
        self.padding = padding
        self.probability = probability
        self.padding_mode = padding_mode
        self.fill = fill
        self.debug_mode = debug_mode
        self.debug_path = debug_path  

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        if torch.rand(1) < self.probability:
            padding_size = random.randint(0, self.padding)  

            image = ImageOps.expand(image, border=padding_size, fill=self.fill)
            label = ImageOps.expand(label, border=padding_size, fill=self.fill)

        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}


class RandomPerspective(object):
    def __init__(self, debug_mode, debug_path, distortion_scale=0.5, probability=0.2, interpolation=InterpolationMode.BILINEAR, fill=0):
        self.distortion_scale = distortion_scale
        self.probability = probability
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale
        self.debug_mode = debug_mode
        self.debug_path = debug_path
        
        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            print("Fill should be wither a sequence or a number")
        self.fill = fill
    def get_params(self, width: int, height: int, distortion_scale: float) -> Tuple[List[List[int]], List[List[int]]]:
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints
    
    
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        
        channels, height, width = F.get_dimensions(image)
        if torch.rand(1) < self.probability:
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            image = F.perspective(image, startpoints, endpoints, InterpolationMode.BILINEAR, 255)
            label = F.perspective(label, startpoints, endpoints, InterpolationMode.BILINEAR, 0)

        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}

class RandomGrayscale(object):
    def __init__(self, debug_mode, debug_path, probability=0.2):
        self.probability = probability
        self.debug_mode = debug_mode
        self.debug_path = debug_path
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if torch.rand(1) <= self.probability:
            image = v2.Grayscale(num_output_channels=3)(image)
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}


class RandomGaussianBlur(object):
    def __init__(self,debug_mode, debug_path, kernel=(3,3), sigma=(0.1,1.0), probability=0.9):
        self.kernel = kernel
        self.sigma = sigma
        self.probability = probability
        self.debug_mode = debug_mode
        self.debug_path = debug_path
    
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if torch.rand(1) >= self.probability:
            image = v2.GaussianBlur(self.kernel, self.sigma)(image)
        
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}

class RandomHorizontalMotionBlur(object):
    def __init__(self, debug_mode, debug_path, kernel_size=21, probability=0.5):
        self.kernel_size = kernel_size
        self.probability = probability
        self.debug_mode = debug_mode
        self.debug_path = debug_path
    
    def motion_blur(self, image):
        
        blurred_image = image.filter(ImageFilter.BoxBlur(self.kernel_size // 2))
                
        return blurred_image
    
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if torch.rand(1) < self.probability:
            image = self.motion_blur(image)
        
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}


class RandomVerticalMotionBlur(object):
    def __init__(self, debug_mode, debug_path, kernel_size=21, probability=0.5):
        self.kernel_size = kernel_size
        self.probability = probability
        self.debug_mode = debug_mode
        self.debug_path = debug_path
    
    def motion_blur(self, image):
        blurred_image = image.filter(ImageFilter.BoxBlur(self.kernel_size // 2))
        return blurred_image
    
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if torch.rand(1) < self.probability:
            image = self.motion_blur(image)
        
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}


class RandomPosterize(object):
    def __init__(self, debug_mode, debug_path, bits=2, probability=0.5):
        self.bits = bits
        self.probability = probability
        self.debug_mode = debug_mode
        self.debug_path = debug_path
        
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if torch.rand(1) >= self.probability:
            image = v2.RandomPosterize(self.bits)(image)
            
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}

class RandomSharpness(object):
    def __init__(self, debug_mode, debug_path, sharpness_factor=2, probability=0.5):
        self.sharpness_factor = sharpness_factor
        self.probability = probability
        self.debug_mode = debug_mode
        self.debug_path = debug_path
        
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        
        if torch.rand(1) >= self.probability:
            image = v2.RandomAdjustSharpness(self.sharpness_factor)(image)
        
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}

class RandomEqualize(object):
    def __init__(self, debug_mode, debug_path, probability=0.5):
        self.probability = probability
        self.debug_mode = debug_mode
        self.debug_path = debug_path
        
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        
        if torch.rand(1) >= self.probability:
            image = v2.RandomEqualize()(image)
        
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}

class RandomColorJitter(object):
    def __init__(self, debug_mode, debug_path, brightness=.1, hue=.1, probability=0.5):
        self.probability = probability
        self.brightness = brightness
        self.hue = hue
        self.debug_mode = debug_mode
        self.debug_path = debug_path
        
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']


        if torch.rand(1) < self.probability:
            image = v2.ColorJitter(self.brightness, self.hue)(image)
        
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}

class RandomBackground(object):
    """Changes the bg of image"""

    def __init__(self, debug_mode, debug_path, probability=0.5):
        assert isinstance(probability, numbers.Number), 'probability should be a single number'
        self.probability = probability
        self.debug_mode = debug_mode
        self.debug_path = debug_path

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']
        
        if torch.rand(1) < self.probability:
            image = random_background(image, label)
        else:
            if torch.rand(1) < 0.95:
                bg = find_dominant_color(image)
                image = combined(image, label, bg)
                
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
     
        return {'imidx': imidx, 'image': image, 'label': label}


class RandomPixelation(object):
    def __init__(self, debug_mode, debug_path, v_pix=0.1, probability=0.5):
        self.probability = probability
        self.v_pix = v_pix
        self.debug_mode = debug_mode
        self.debug_path = debug_path

    def pixelation(self, img):

        original_width, original_height = img.size
        minRatio = 0.001
        ratio = 1 - (1 - minRatio) * (self.v_pix / 100) ** (1 / 30)
        new_width = max(2, int(original_width * ratio))
        new_height = max(2, int(original_height * ratio))

        img = img.resize((new_width, new_height), Image.BILINEAR)
        img_out = img.resize((original_width, original_height), Image.NEAREST)

        return img_out
    
    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if torch.rand(1) < self.probability:
            image = self.pixelation(image)
        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}

class RandomJpegCompression(object):
    def __init__(self, debug_mode, debug_path, max_quality=10, probability=0.5):
        self.probability = probability
        self.max_quality = max_quality
        self.debug_mode = debug_mode
        self.debug_path = debug_path

    def jpeg_compress(self, image_tensor):
        """
        Compresses an image using JPEG compression through Pillow.
        """
        import io

        buffer = io.BytesIO()
        image_tensor.save(buffer, format="JPEG", quality=self.max_quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        return compressed_image


    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if torch.rand(1) < self.probability:
            image = self.jpeg_compress(image)

        debug_params = DebuggerParams(self.__class__.__name__, '.jpg')    
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, image)
        Debugger(self.debug_mode, self.debug_path).save_image(debug_params, label)
        return {'imidx': imidx, 'image': image, 'label': label}

 