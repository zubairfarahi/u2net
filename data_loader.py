# Standard library imports
import numpy as np
from typing import Any, List

# Third-party imports
import torch
from skimage import io, transform, color

from torch.utils.data import Dataset
from PIL import ImageOps
#==========================dataset load==========================
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		from PIL import Image
  
		img = image.resize((self.output_size, self.output_size), resample=Image.BICUBIC)
		lbl = label.resize((self.output_size, self.output_size), resample=Image.NEAREST)
  
		return {'imidx':imidx, 'image':img, 'label':lbl}

import cv2
class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		# image.save('image.jpg')
		# label.save('bg.png')
		label = label.convert('L')
		from torchvision import transforms
  		
		mean = torch.tensor([0.485, 0.456, 0.406])
		std = torch.tensor([0.229, 0.224, 0.225])
		to_tensor = transforms.ToTensor()
		normalize = transforms.Normalize(mean=mean, std=std)

		if image.mode == "L":
			image = image.convert('RGB')
			mean = torch.tensor([0.485, 0.485, 0.485])
			std = torch.tensor([0.229, 0.229, 0.229])
  
		img = to_tensor(image)
		lbl = to_tensor(label)

		img = normalize(img)

		return {'imidx':torch.from_numpy(imidx), 'image': img, 'label': lbl}


class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None, mode='train'):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform
		self.mode = mode
  
	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):
		from PIL import Image
  
		imname = self.image_name_list[idx]
		imidx = np.array([idx])
  
		image = Image.open(imname).convert('RGB')
		image = ImageOps.exif_transpose(image)


		if(0==len(self.label_name_list)):
			if self.mode == 'train':
				raise(ValueError("No labels"))
			else:
				width, height = image.size
				label = Image.new("RGB", (width, height), 0)
 
		else:
			label = Image.open(self.label_name_list[idx]).convert('RGB')
		
		if label.size != image.size:
			image = image.transpose(Image.TRANSPOSE)
        
		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample



