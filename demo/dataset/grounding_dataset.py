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



class GroundingDataset(BaseDataset):
    def __init__(self, 
                image_root, 
                json_path,
                annotation_embedding_path,
                prob_real_caption=1,
                image_size=256, 
                min_box_size=0.01,
                max_boxes_per_data=8,
                max_images=None, # set as 30K used to eval
                random_crop = False,
                random_flip = True,
                ):
        super().__init__(image_root, random_crop, random_flip, image_size)
        self.image_root = image_root
        self.json_path = json_path
        self.annotation_embedding_path = annotation_embedding_path
        self.prob_real_caption = prob_real_caption
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images


        # Load raw data 
        with open(json_path, 'r') as f:
            json_raw = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
        self.data = json_raw["images"] # donot name it images, which is misleading
        self.annotations = json_raw["annotations"]

        
        # Load preprocessed name embedding 
        if 'bert' in annotation_embedding_path:
            self.embedding_len = 1280
        elif 'clip' in annotation_embedding_path:
            self.embedding_len = 768
        else:
            assert False


        # clean data and annotation
        check_unique( self.data, ['id'] )
        check_unique( self.annotations, ['id'] )
        clean_data(self.data)
        clean_annotations(self.annotations)
        self.data_id_list = [  datum['data_id'] for datum in self.data   ]
        self.data = { datum['data_id']:datum  for datum in self.data } # map self.data from a list into a dict 


        # data point to its annotation mapping 
        self.data_id_to_annos = defaultdict(list)
        for anno in self.annotations:
            self.data_id_to_annos[ anno["data_id"] ].append(anno)

        

        # These are not used that offen, but are useful in some cases
        self.file_names = [] # all training images 
        self.file_name_to_data_ids = defaultdict(list) # for each image, there are multiple data points (captions)
        for data_id in self.data_id_list:
            fine_name = self.data[data_id]["file_name"]
            self.file_names.append(fine_name)
            self.file_name_to_data_ids[fine_name].append(data_id)
        self.file_names = list(set(self.file_names))


        if self.max_images is not None:
            "This is only used as COCO2017P evulation, when we set max_images as 30k"
            assert False, 'I have commented out the following code to save cpu memory'
            # new_data_id_list = []
            # new_file_name_to_data_ids = defaultdict(list)
            # self.file_names = self.file_names[0:self.max_images]
            # for file_name in self.file_names:
            #     data_id = self.file_name_to_data_ids[file_name][0]
            #     new_data_id_list.append(data_id)
            #     new_file_name_to_data_ids[file_name].append(data_id)
            # self.data_id_list = new_data_id_list
            # self.file_name_to_data_ids = new_file_name_to_data_ids


		# Check if all filenames can be found in the zip file
        # all_filenames = [self.data[idx]['file_name']  for idx in self.data_id_list ]
        # check_filenames_in_zipdata(all_filenames, image_root)  
         

    def total_images(self):
        return len(self.file_names)


    def __getitem__(self, index):
        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes?"

        out = {}

        data_id = self.data_id_list[index]
        out['id'] = data_id
        

        # Image and caption 
        file_name = self.data[data_id]['file_name']
        image = self.fetch_image(file_name)
        image_tensor, trans_info = self.transform_image(image)
        out["image"] = image_tensor

        if random.uniform(0, 1) < self.prob_real_caption:
            out["caption"] = self.data[data_id]["caption"]
        else:
            out["caption"] = ""

        

        annos = deepcopy(self.data_id_to_annos[data_id])
        areas = []
        all_boxes = []
        all_masks = []
        all_positive_embeddings = []


        for anno in annos:

            x, y, w, h = anno['bbox']
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)

            if valid:
                areas.append(  (x1-x0)*(y1-y0)  )
                all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
                all_masks.append(1)
                all_positive_embeddings.append( torch.load(os.path.join(self.annotation_embedding_path,str(anno["id"])), map_location='cpu'  )  )
                
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]

        boxes = torch.zeros(self.max_boxes_per_data, 4)
        masks = torch.zeros(self.max_boxes_per_data)
        positive_embeddings = torch.zeros(self.max_boxes_per_data, self.embedding_len)
        for i, idx in enumerate(wanted_idxs):
            boxes[i] = all_boxes[idx]
            masks[i] = all_masks[idx]
            positive_embeddings[i] = all_positive_embeddings[idx]


        out["boxes"] = boxes
        out["masks"] = masks
        out["positive_embeddings"] = positive_embeddings      
        
        return out



    def __len__(self):
        return len(self.data_id_list)


