from tkinter.messagebox import NO
import torch 
import json 
from collections import defaultdict
from PIL import Image, ImageDraw
from copy import deepcopy
import os 
import torchvision.transforms as transforms
import torchvision
from .base_dataset import BaseDataset, check_filenames_in_zipdata, recalculate_box_and_verify_if_valid  
from io import BytesIO
import random

from .tsv import TSVFile

from io import BytesIO
import base64
from PIL import Image
import numpy as np


def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

def decode_tensor_from_string(arr_str, use_tensor=True):
    arr = np.frombuffer(base64.b64decode(arr_str), dtype='float32')
    if use_tensor:
        arr = torch.from_numpy(arr)
    return arr

def decode_item(item):
    item = json.loads(item)
    item['image'] = decode_base64_to_pillow(item['image'])

    for anno in item['annos']:
        anno['image_embedding_before'] = decode_tensor_from_string(anno['image_embedding_before'])
        anno['text_embedding_before'] = decode_tensor_from_string(anno['text_embedding_before'])
        anno['image_embedding_after'] = decode_tensor_from_string(anno['image_embedding_after'])
        anno['text_embedding_after'] = decode_tensor_from_string(anno['text_embedding_after'])
    return item

def check_unique(images, fields):
    for field in fields:
        temp_list = []
        for img_info in images:
            temp_list.append(img_info[field])
        assert len(set(temp_list)) == len(temp_list), field

def clean_data(data):
    for data_info in data:
        data_info.pop("original_img_id", None)
        data_info.pop("original_id", None)
        data_info.pop("sentence_id", None)  # sentence id for each image (multiple sentences for one image)
        data_info.pop("dataset_name", None)  
        data_info.pop("data_source", None) 
        data_info["data_id"] = data_info.pop("id")


def clean_annotations(annotations):
    for anno_info in annotations:
        anno_info.pop("iscrowd", None) # I have checked that all 0 for flickr, vg, coco
        anno_info.pop("category_id", None)  # I have checked that all 1 for flickr vg. This is not always 1 for coco, but I do not think we need this annotation
        anno_info.pop("area", None)
        # anno_info.pop("id", None)
        anno_info["data_id"] = anno_info.pop("image_id")


def draw_box(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline ="red", width=2) # x0 y0 x1 y1 
    return img 


def xyhw2xyxy(box):
    x0, y0, w, h = box
    return [ x0, y0, x0+w, y0+h ]


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


def mask_for_random_drop_text_or_image_feature(masks, random_drop_embedding):
    """
    input masks tell how many valid grounding tokens for this image
    e.g., 1,1,1,1,0,0,0,0,0,0...

    If random_drop_embedding=both.  we will random drop either image or
    text feature for each token, 
    but we always make sure there is at least one feature used. 
    In other words, the following masks are not valid 
    (because for the second obj, no feature at all):
    image: 1,0,1,1,0,0,0,0,0
    text:  1,0,0,0,0,0,0,0,0

    if random_drop_embedding=image. we will random drop image feature 
    and always keep the text one.  

    """
    N = masks.shape[0]

    if random_drop_embedding=='both':
        temp_mask = torch.ones(2,N)
        for i in range(N):
            if random.uniform(0, 1) < 0.5: # else keep both features 
                idx = random.sample([0,1], 1)[0] # randomly choose to drop image or text feature 
                temp_mask[idx,i] = 0 
        image_masks = temp_mask[0]*masks
        text_masks = temp_mask[1]*masks
    
    if random_drop_embedding=='image':
        image_masks = masks*(torch.rand(N)>0.5)*1
        text_masks = masks

    return image_masks, text_masks





def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def inv_project(y, projection_matrix):
    """
    y (Batch*768) should be the CLIP feature (after projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim).  
    this function will return the CLIP penultimate feature. 
    
    Note: to make sure getting the correct penultimate feature, the input y should not be normalized. 
    If it is normalized, then the result will be scaled by CLIP feature norm, which is unknown.   
    """
    return y@torch.transpose(torch.linalg.inv(projection_matrix), 0, 1)




class TSVDataset(BaseDataset):
    def __init__(self, 
                tsv_path,
                which_embedder='clip',
                which_layer=['after','after'], # text and image  
                prob_use_caption=1,
                random_drop_embedding='none',
                image_size=256, 
                min_box_size=0.01,
                max_boxes_per_data=8,
                max_images=None, # set as 30K used to eval
                random_crop = False,
                random_flip = True,
                ):
        image_root = "a placeholder path as we are using tsv here"
        super().__init__(image_root, random_crop, random_flip, image_size)
        self.tsv_path = tsv_path
        self.which_embedder = which_embedder
        self.prob_use_caption = prob_use_caption
        self.random_drop_embedding = random_drop_embedding
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images

        assert which_layer in [ ['after','after'],  ['before','after_renorm'], ['before','after_reproject'] ]
        assert random_drop_embedding in ['none', 'both', 'image']
        self.which_layer_text  = which_layer[0]
        self.which_layer_image = which_layer[1]

        #self.projection_matrix = torch.load(os.path.join(os.path.dirname(__file__), 'projection_matrix')  )
        self.projection_matrix = torch.load('projection_matrix')

        # Load tsv data
        self.tsv_file = TSVFile(self.tsv_path)

        
        # Load preprocessed name embedding 
        if which_embedder == 'bert':
            self.embedding_len = 1280
        elif which_embedder == 'clip':
            self.embedding_len = 768
        else:
            assert False

    def total_images(self):
        return len(self)

    def get_item_from_tsv(self, index):
        _, item = self.tsv_file[index]
        item = decode_item(item)
        return item


    def mapping(self, image_embedding):
        if self.which_layer_image == 'after':
            # both use CLIP aligned feature 
            return image_embedding
        elif self.which_layer_image == 'after_renorm':
            # text use before, but image use after projection but normalize to 28.7 
            return image_embedding*28.7
        elif self.which_layer_image == 'after_reproject':
            image_embedding = project( image_embedding.unsqueeze(0), self.projection_matrix.T )
            image_embedding = image_embedding.squeeze(0)
            image_embedding = image_embedding / image_embedding.norm() 
            image_embedding = image_embedding * 28.7 
            return image_embedding



    def __getitem__(self, index):
        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes?"

        raw_item = self.get_item_from_tsv(index)
        is_det = raw_item.get('is_det', False) # if it is from detection (such as o365), then we will make a caption

        out = {}

        # -------------------- id and image ------------------- # 
        out['id'] = raw_item['data_id']
        image = raw_item['image']
        image_tensor, trans_info = self.transform_image(image)
        out["image"] = image_tensor



        # -------------------- grounding token ------------------- # 
        annos = raw_item['annos']
        
        areas = []
        all_boxes = []
        all_masks = []
        all_text_embeddings = []
        all_image_embeddings = []
        if is_det:
            all_category_names = []

        text_embedding_name = 'text_embedding_before' if self.which_layer_text == 'before' else 'text_embedding_after'
        image_embedding_name = 'image_embedding_after'
        
        for anno in annos:
            x, y, w, h = anno['bbox']
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)

            if valid:
                areas.append(  (x1-x0)*(y1-y0)  )
                all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
                all_masks.append(1)
                all_text_embeddings.append(anno[text_embedding_name])
                all_image_embeddings.append(  self.mapping(anno[image_embedding_name])  )
                if is_det:
                    all_category_names.append(anno["category_name"])

                
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]

        boxes = torch.zeros(self.max_boxes_per_data, 4)
        masks = torch.zeros(self.max_boxes_per_data)
        text_embeddings =  torch.zeros(self.max_boxes_per_data, self.embedding_len)
        image_embeddings = torch.zeros(self.max_boxes_per_data, self.embedding_len)
        if is_det:
            category_names = []
        for i, idx in enumerate(wanted_idxs):
            boxes[i] = all_boxes[idx]
            masks[i] = all_masks[idx]
            text_embeddings[i] =  all_text_embeddings[idx]
            image_embeddings[i] = all_image_embeddings[idx]
            if is_det:
                category_names.append(all_category_names[idx])

        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(masks, self.random_drop_embedding)
        else:
            image_masks = masks
            text_masks = masks


        out["boxes"] = boxes
        out["masks"] = masks
        out["image_masks"] = image_masks
        out["text_masks"] = text_masks
        out["text_embeddings"] =  text_embeddings  
        out["image_embeddings"] = image_embeddings      
        


        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            if is_det:
                out["caption"] = make_a_sentence(category_names)
            else:
                out["caption"] = raw_item["caption"]
        else:
            out["caption"] = ""

        return out



    def __len__(self):
        return len(self.tsv_file)


