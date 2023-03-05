import json, os, random, math
from collections import defaultdict
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from .base_dataset import BaseDataset, check_filenames_in_zipdata, recalculate_box_and_verify_if_valid 
from io import BytesIO



def not_in_at_all(list1, list2):
	for a in list1:
		if a in list2:
			return False
	return True


def clean_annotations(annotations):
	for anno in annotations:
		anno.pop("segmentation", None)
		anno.pop("area", None)
		anno.pop("iscrowd", None)
		# anno.pop("id", None)


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


def check_all_have_same_images(instances_data, stuff_data, caption_data):
	if stuff_data is not None:
		assert instances_data["images"] == stuff_data["images"]
	if caption_data is not None:
		assert instances_data["images"] == caption_data["images"]


class CDDataset(BaseDataset):
	"CD: Caption Detection"
	def __init__(self, 
                image_root,
				category_embedding_path,
				instances_json_path = None,
				stuff_json_path = None,
				caption_json_path = None,
				prob_real_caption = 0,
				fake_caption_type = 'empty',
                image_size=256, 
                max_images=None,
                min_box_size=0.01,
                max_boxes_per_image=8,
                include_other=False, 
				random_crop = False,
				random_flip = True,
                ):
		super().__init__(random_crop, random_flip, image_size)

		self.image_root = image_root
		self.category_embedding_path = category_embedding_path
		self.instances_json_path = instances_json_path
		self.stuff_json_path = stuff_json_path
		self.caption_json_path = caption_json_path
		self.prob_real_caption = prob_real_caption
		self.fake_caption_type = fake_caption_type
		self.max_images = max_images
		self.min_box_size = min_box_size
		self.max_boxes_per_image = max_boxes_per_image
		self.include_other = include_other


		assert fake_caption_type in ["empty", "made"]
		if prob_real_caption > 0:
			assert caption_json_path is not None, "caption json must be given"
	

		# Load all jsons 
		with open(instances_json_path, 'r') as f:
			instances_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
		clean_annotations(instances_data["annotations"])
		self.instances_data = instances_data

		self.stuff_data = None
		if stuff_json_path is not None:
			with open(stuff_json_path, 'r') as f:
				stuff_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
			clean_annotations(stuff_data["annotations"])
			self.stuff_data = stuff_data

		self.captions_data = None
		if caption_json_path is not None:
			with open(caption_json_path, 'r') as f:
				captions_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
			clean_annotations(captions_data["annotations"])
			self.captions_data = captions_data


		# Load preprocessed name embedding 
		self.category_embeddings = torch.load(category_embedding_path)
		self.embedding_len = list( self.category_embeddings.values() )[0].shape[0]
		

		# Misc  
		self.image_ids = [] # main list for selecting images
		self.image_id_to_filename = {} # file names used to read image
		check_all_have_same_images(self.instances_data, self.stuff_data, self.captions_data)
		for image_data in self.instances_data['images']:
			image_id = image_data['id']
			filename = image_data['file_name']
			self.image_ids.append(image_id)
			self.image_id_to_filename[image_id] = filename

		
		# All category names (including things and stuff)
		self.object_idx_to_name = {} 
		for category_data in self.instances_data['categories']:
			self.object_idx_to_name[category_data['id']] = category_data['name']
		if self.stuff_data is not None:
			for category_data in self.stuff_data['categories']:
				self.object_idx_to_name[category_data['id']] = category_data['name']


		# Add object data from instances and stuff 
		self.image_id_to_objects = defaultdict(list)
		self.select_objects( self.instances_data['annotations'] )
		if self.stuff_data is not None:
			self.select_objects( self.stuff_data['annotations'] )

		# Add caption data 
		if self.captions_data is not None:
			self.image_id_to_captions = defaultdict(list)
			self.select_captions( self.captions_data['annotations'] )

		# Check if all filenames can be found in the zip file
		# all_filenames = [self.image_id_to_filename[idx] for idx in self.image_ids]
		# check_filenames_in_zipdata(all_filenames, image_root)
		

	def select_objects(self, annotations):
		for object_anno in annotations:
			image_id = object_anno['image_id']
			object_name = self.object_idx_to_name[object_anno['category_id']]
			other_ok = object_name != 'other' or self.include_other
			if other_ok:
				self.image_id_to_objects[image_id].append(object_anno)


	def select_captions(self, annotations):
		for caption_data in annotations:
			image_id = caption_data['image_id']
			self.image_id_to_captions[image_id].append(caption_data)


	def total_images(self):
		return len(self)


	def __getitem__(self, index):
		if self.max_boxes_per_image > 99:
			assert False, "Are you sure setting such large number of boxes?"

		out = {}

		image_id = self.image_ids[index]
		out['id'] = image_id
		
		# Image 
		filename = self.image_id_to_filename[image_id]
		image = self.fetch_image(filename)
		#WW, HH = image.size
		image_tensor, trans_info = self.transform_image(image)
		out["image"] = image_tensor
	

		# Select valid boxes after cropping (center or random)
		this_image_obj_annos = deepcopy(self.image_id_to_objects[image_id])
		areas = []
		all_obj_names = []
		all_boxes = []
		all_masks = []
		all_positive_embeddings = []
		for object_anno in this_image_obj_annos:

			x, y, w, h = object_anno['bbox']
			valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)

			if valid:
				areas.append(  (x1-x0)*(y1-y0) )
				obj_name = self.object_idx_to_name[ object_anno['category_id']  ]
				all_obj_names.append(obj_name)
				all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
				all_masks.append(1)
				all_positive_embeddings.append( self.category_embeddings[obj_name]  )

		wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
		wanted_idxs = wanted_idxs[0:self.max_boxes_per_image]
		obj_names = [] # used for making a sentence
		boxes = torch.zeros(self.max_boxes_per_image, 4)
		masks = torch.zeros(self.max_boxes_per_image)
		positive_embeddings = torch.zeros(self.max_boxes_per_image, self.embedding_len)
		for i, idx in enumerate(wanted_idxs):
			obj_names.append(  all_obj_names[idx]   )
			boxes[i] = all_boxes[idx]
			masks[i] = all_masks[idx]
			positive_embeddings[i] = all_positive_embeddings[idx]

		# Caption
		if random.uniform(0, 1) < self.prob_real_caption:
			caption_data = self.image_id_to_captions[image_id]
			idx = random.randint(0,  len(caption_data)-1 )
			caption = caption_data[idx]["caption"]
		else:
			if self.fake_caption_type == "empty":
				caption = ""
			else:
				caption = make_a_sentence(obj_names, clean=True)
		
		
		out["caption"] = caption
		out["boxes"] = boxes
		out["masks"] = masks
		out["positive_embeddings"] = positive_embeddings

		return out 


	def __len__(self):
		if self.max_images is None:
			return len(self.image_ids)
		return min(len(self.image_ids), self.max_images)	

