from tkinter.messagebox import NO
import torch 
import json 
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as transforms
from io import BytesIO
import random
import torchvision.transforms.functional as TF

from .tsv import TSVFile

from io import BytesIO
import base64
import numpy as np


def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

def decode_tensor_from_string(arr_str, use_tensor=True):
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
    if use_tensor:
        arr = torch.from_numpy(arr)
    return arr


def decode_item(item):
    "This is for decoding TSV for box data"
    item = json.loads(item)
    item['image'] = decode_base64_to_pillow(item['image'])

    for anno in item['annos']:
        anno['image_embedding_before'] = decode_tensor_from_string(anno['image_embedding_before'])
        anno['text_embedding_before'] = decode_tensor_from_string(anno['text_embedding_before'])
        anno['image_embedding_after'] = decode_tensor_from_string(anno['image_embedding_after'])
        anno['text_embedding_after'] = decode_tensor_from_string(anno['text_embedding_after'])
    return item


def decode_item_depth(item):
    "This is for decoding TSV for depth data"
    item = json.loads(item)
    item['depth'] = decode_base64_to_pillow(item['depth'])
    return item



class DepthDataset():
    def __init__(self, tsv_path, depth_tsv_path, prob_use_caption=1, image_size=512, random_flip=False):

        self.tsv_path = tsv_path
        self.depth_tsv_path = depth_tsv_path
        self.prob_use_caption = prob_use_caption
        self.image_size = image_size
        self.random_flip = random_flip

        # Load tsv data
        self.tsv_file = TSVFile(self.tsv_path)
        self.depth_tsv_file = TSVFile(self.depth_tsv_path)      

        self.pil_to_tensor = transforms.PILToTensor()


    def total_images(self):
        return len(self)


    def get_item_from_tsv(self, index):
        _, item = self.tsv_file[index]
        item = decode_item(item)
        return item


    def get_item_from_depth_tsv(self, index):
        _, item = self.depth_tsv_file[index]
        item = decode_item_depth(item)
        return item



    def __getitem__(self, index):

        raw_item = self.get_item_from_tsv(index)
        raw_item_depth = self.get_item_from_depth_tsv(index)

        assert raw_item['data_id'] == raw_item_depth['data_id']
        
        out = {}

        out['id'] = raw_item['data_id']
        image = raw_item['image']
        depth = raw_item_depth['depth']

        # - - - - - center_crop, resize and random_flip - - - - - - #  
        assert  image.size == depth.size   

        crop_size = min(image.size)
        image = TF.center_crop(image, crop_size)
        image = image.resize( (self.image_size, self.image_size) )

        depth = TF.center_crop(depth, crop_size)
        depth = depth.resize( (self.image_size, self.image_size) )


        if self.random_flip and random.random()<0.5:
            image = ImageOps.mirror(image)
            depth = ImageOps.mirror(depth)
        
        out['image'] = ( self.pil_to_tensor(image).float()/255 - 0.5 ) / 0.5
        out['depth'] = ( self.pil_to_tensor(depth).float()/255 - 0.5 ) / 0.5
        out['mask'] = torch.tensor(1.0) 

        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            out["caption"] = raw_item["caption"]
        else:
            out["caption"] = ""

        return out


    def __len__(self):
        return len(self.tsv_file)


