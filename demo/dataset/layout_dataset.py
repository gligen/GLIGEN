import json, os, random, math
from collections import defaultdict
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image, ImageOps
from .base_dataset import BaseDataset, check_filenames_in_zipdata 
from io import BytesIO




def clean_annotations(annotations):
	for anno in annotations:
		anno.pop("segmentation", None)
		anno.pop("area", None)
		anno.pop("iscrowd", None)
		anno.pop("id", None)


def make_a_sentence(obj_names, clean=False):

	if clean:
		obj_names = [ name[:-6] if ("-other" in name) else name for name in obj_names]

	caption = ""
	tokens_positive = []
	for obj_name in obj_names:
		start_len = len(caption)
		caption += obj_name
		end_len = len(caption)
		caption += ", "
		tokens_positive.append(
			[[start_len, end_len]] # in real caption, positive tokens can be disjoint, thus using list of list
		)
	caption = caption[:-2] # remove last ", "

	return caption #, tokens_positive


class LayoutDataset(BaseDataset):
	"""
	Note: this dataset can somehow be achieved in cd_dataset.CDDataset
	Since if you donot set prob_real_caption=0 in CDDataset, then that 
	dataset will only use detection annotations. However, in that dataset, 
	we do not remove images but remove boxes. 

	However, in layout2img works, people will just resize raw image data into 256*256,
	thus they pre-calculate box size and apply min_box_size before min/max_boxes_per_image.
	And then they will remove images if does not follow the rule. 

	These two different methods will lead to different number of training/val images. 
	Thus this dataset here is only for layout2img.

	"""
	def __init__(self, 
                image_root,
				instances_json_path,
				stuff_json_path,
				category_embedding_path,
				fake_caption_type = 'empty', 
                image_size=256, 
                max_samples=None,
                min_box_size=0.02,
                min_boxes_per_image=3, 
                max_boxes_per_image=8,
                include_other=False, 
				random_flip=True
                ):
		super().__init__(random_crop=None, random_flip=None, image_size=None) # we only use vis_getitem func in BaseDataset, donot use the others. 

		assert fake_caption_type in ['empty', 'made']
		self.image_root = image_root
		self.instances_json_path = instances_json_path
		self.stuff_json_path = stuff_json_path
		self.category_embedding_path = category_embedding_path
		self.fake_caption_type = fake_caption_type
		self.image_size = image_size
		self.max_samples = max_samples
		self.min_box_size = min_box_size
		self.min_boxes_per_image = min_boxes_per_image
		self.max_boxes_per_image = max_boxes_per_image
		self.include_other = include_other
		self.random_flip = random_flip

	
		self.transform = transforms.Compose([transforms.Resize( (image_size, image_size) ),
											 transforms.ToTensor(),
											 transforms.Lambda(lambda t: (t * 2) - 1) ])
		
		# Load all jsons 
		with open(instances_json_path, 'r') as f:
			instances_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
		clean_annotations(instances_data["annotations"])
		self.instances_data = instances_data

		with open(stuff_json_path, 'r') as f:
			stuff_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
		clean_annotations(stuff_data["annotations"])
		self.stuff_data = stuff_data


		# Load preprocessed name embedding 
		self.category_embeddings = torch.load(category_embedding_path)
		self.embedding_len = list( self.category_embeddings.values() )[0].shape[0]
		

		# Misc  
		self.image_ids = [] # main list for selecting images
		self.image_id_to_filename = {} # file names used to read image
		self.image_id_to_size = {} # original size of this image 
		assert instances_data['images'] == stuff_data["images"] 
		for image_data in instances_data['images']:
			image_id = image_data['id']
			filename = image_data['file_name']
			width = image_data['width']
			height = image_data['height']
			self.image_ids.append(image_id)
			self.image_id_to_filename[image_id] = filename
			self.image_id_to_size[image_id] = (width, height)
		
		# All category names (including things and stuff)
		self.things_id_list = []
		self.stuff_id_list = []
		self.object_idx_to_name = {} 
		for category_data in instances_data['categories']:
			self.things_id_list.append( category_data['id'] )
			self.object_idx_to_name[category_data['id']] = category_data['name']
		for category_data in stuff_data['categories']:
			self.stuff_id_list.append( category_data['id'] )
			self.object_idx_to_name[category_data['id']] = category_data['name']
		self.all_categories = [   self.object_idx_to_name.get(k, None) for k in range(183+1) ] 


		# Add object data from instances and stuff 
		self.image_id_to_objects = defaultdict(list)
		self.select_objects( instances_data['annotations'] )
		self.select_objects( stuff_data['annotations'] )


		# Prune images that have too few or too many objects
		new_image_ids = []
		for image_id in self.image_ids:
			num_objs = len(self.image_id_to_objects[image_id])
			if self.min_boxes_per_image <= num_objs <= self.max_boxes_per_image:
				new_image_ids.append(image_id)
		self.image_ids = new_image_ids


		# Check if all filenames can be found in the zip file
		all_filenames = [self.image_id_to_filename[idx] for idx in self.image_ids]
		check_filenames_in_zipdata(all_filenames, image_root)
		


	def select_objects(self, annotations):
		for object_anno in annotations:
			image_id = object_anno['image_id']
			_, _, w, h = object_anno['bbox']
			W, H = self.image_id_to_size[image_id]
			box_area = (w * h) / (W * H)
			box_ok = box_area > self.min_box_size
			object_name = self.object_idx_to_name[object_anno['category_id']]
			other_ok = object_name != 'other' or self.include_other
			if box_ok and other_ok:
				self.image_id_to_objects[image_id].append(object_anno)


	def total_images(self):
		return len(self)


	def __getitem__(self, index):
		if self.max_boxes_per_image > 99:
			assert False, "Are you sure setting such large number of boxes?"

		out = {}

		image_id = self.image_ids[index]
		out['id'] = image_id

		flip = self.random_flip and random.random()<0.5
		
		# Image 
		filename = self.image_id_to_filename[image_id]
		zip_file = self.fetch_zipfile(self.image_root)
		image = Image.open(BytesIO(zip_file.read(filename))).convert('RGB')
		WW, HH = image.size
		if flip:
			image = ImageOps.mirror(image)
		out["image"] = self.transform(image)

		this_image_obj_annos = deepcopy(self.image_id_to_objects[image_id])

		# Make a sentence 
		obj_names = [] # used for make a sentence
		boxes = torch.zeros(self.max_boxes_per_image, 4)
		masks = torch.zeros(self.max_boxes_per_image)
		positive_embeddings = torch.zeros(self.max_boxes_per_image, self.embedding_len)
		for idx, object_anno in enumerate(this_image_obj_annos):
			obj_name = self.object_idx_to_name[ object_anno['category_id']  ]
			obj_names.append(obj_name)
			x, y, w, h = object_anno['bbox']
			x0 = x / WW
			y0 = y / HH
			x1 = (x + w) / WW
			y1 = (y + h) / HH
			if flip:
				x0, x1 = 1-x1, 1-x0
			boxes[idx] = torch.tensor([x0,y0,x1,y1])
			masks[idx] = 1 
			positive_embeddings[idx] = self.category_embeddings[obj_name] 

		if self.fake_caption_type == 'empty':
			caption = ""
		else:
			caption = make_a_sentence(obj_names, clean=True)
	
		out["caption"] = caption	
		out["boxes"] = boxes
		out["masks"] = masks
		out["positive_embeddings"] = positive_embeddings


		return out 


	def __len__(self):
		if self.max_samples is None:
			return len(self.image_ids)
		return min(len(self.image_ids), self.max_samples)	


