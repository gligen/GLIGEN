import json, os, random, math
from collections import defaultdict
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from .base_dataset_kp import BaseDataset, check_filenames_in_zipdata, recalculate_box_kps_and_verify_if_valid 
from io import BytesIO


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clean_kps(kps):
	assert len(kps) == 51
	kps = list(chunks(kps, 3))
	out = []
	for idx, kp in enumerate(kps):
		name = "kp"+str(idx).zfill(2)
		loc = [kp[0], kp[1]]
		valid = True if kp[2] == 2 else False 
		if not valid:
			loc = [0,0]
		out.append(  {"name":name, "loc":loc, "valid":valid}          )
	return out


def norm_kps(kps, image_size):
	for kp in kps:
		if kp["valid"]:
			kp_x, kp_y = kp["loc"] 
			kp["loc"] = [ kp_x/image_size, kp_y/image_size ]
	return kps 



def clean_annotations(annotations):
	for anno in annotations:
		anno.pop("segmentation", None)
		anno.pop("area", None)
		anno.pop("iscrowd", None)
		anno.pop("id", None)


def check_all_have_same_images(instances_data, caption_data):
	if caption_data is not None:
		assert instances_data["images"] == caption_data["images"]


class KeypointDataset(BaseDataset):
	def __init__(self, 
                image_root,
				keypoints_json_path = None,
				caption_json_path = None,
				prob_real_caption = 0,
                image_size=512, 
                max_images=None,
				min_box_size=0.0,
                max_persons_per_image=8,
				random_crop = False,
				random_flip = True,
                ):
		super().__init__(random_crop, random_flip, image_size)

		self.image_root = image_root
		self.keypoints_json_path = keypoints_json_path
		self.caption_json_path = caption_json_path
		self.prob_real_caption = prob_real_caption
		self.max_images = max_images
		self.min_box_size = min_box_size
		self.max_persons_per_image = max_persons_per_image
		


		if prob_real_caption > 0:
			assert caption_json_path is not None, "caption json must be given"
	

		# Load keypoint json 
		with open(keypoints_json_path, 'r') as f:
			keypoints_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
		clean_annotations(keypoints_data["annotations"])
		self.keypoints_data = keypoints_data
		
		# Load caption json
		self.captions_data = None
		if caption_json_path is not None:
			with open(caption_json_path, 'r') as f:
				captions_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
			clean_annotations(captions_data["annotations"])
			self.captions_data = captions_data
		
		# A mapping from image_id to objects (keypoint here) 
		self.image_id_to_objects = defaultdict(list)
		self.select_objects( self.keypoints_data['annotations'] )

		# A mapping from image_id to caption 
		if self.captions_data is not None:
			self.image_id_to_captions = defaultdict(list)
			self.select_captions( self.captions_data['annotations'] )

		# Misc  
		self.image_ids = [] # main list for selecting images
		self.image_id_to_filename = {} # file names used to read image
		check_all_have_same_images(self.keypoints_data, self.captions_data)
		
		for image_data in self.keypoints_data['images']:
			image_id = image_data['id']
			if image_id in self.image_id_to_objects: # not all images have person keypoint
				filename = image_data['file_name']
				self.image_ids.append(image_id)
				self.image_id_to_filename[image_id] = filename

		

	def select_objects(self, annotations):
		for object_anno in annotations:
			image_id = object_anno['image_id']
			self.image_id_to_objects[image_id].append(object_anno)

	def select_captions(self, annotations):
		for caption_data in annotations:
			image_id = caption_data['image_id']
			self.image_id_to_captions[image_id].append(caption_data)


	def total_images(self):
		return len(self)


	def __getitem__(self, index):
		if self.max_persons_per_image > 99:
			assert False, "Are you sure setting such large number of boxes?"
		
		out = {}

		image_id = self.image_ids[index]
		out['id'] = image_id
		#image_id = 18150 #180560 
		# Image 
		filename = self.image_id_to_filename[image_id]
		image = Image.open( os.path.join(self.image_root,filename) ).convert('RGB')
		image_tensor, trans_info = self.transform_image(image)
		out["image"] = image_tensor
		
		
		# Select valid boxes after cropping (center or random)
		this_image_obj_annos = deepcopy(self.image_id_to_objects[image_id])
		areas = []
		all_kps = []
		for object_anno in this_image_obj_annos:
			
			x, y, w, h = object_anno['bbox']
			kps = clean_kps( object_anno['keypoints'] )
			valid, (x0, y0, x1, y1), kps = recalculate_box_kps_and_verify_if_valid(x, y, w, h, kps, trans_info, self.image_size, self.min_box_size)

			if valid:
				areas.append(  (x1-x0)*(y1-y0) )
				all_kps.append( norm_kps(kps, self.image_size) ) 


		wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
		wanted_idxs = wanted_idxs[0:self.max_persons_per_image]
		points = torch.zeros(self.max_persons_per_image*17,2)
		masks = torch.zeros(self.max_persons_per_image*17)
		i = 0 
		for idx in wanted_idxs:
			kps = all_kps[idx]
			for kp in kps:
				points[i] = torch.tensor( kp['loc'] )
				masks[i] = 1 if kp["valid"] else 0 
				i += 1 
	
		# Caption
		if random.uniform(0, 1) < self.prob_real_caption:
			caption_data = self.image_id_to_captions[image_id]
			idx = random.randint(0,  len(caption_data)-1 )
			caption = caption_data[idx]["caption"]
		else:
			caption = ""

		out["caption"] = caption
		out["points"] = points
		out["masks"] = masks

		return out 


	def __len__(self):
		if self.max_images is None:
			return len(self.image_ids)
		return min(len(self.image_ids), self.max_images)	

