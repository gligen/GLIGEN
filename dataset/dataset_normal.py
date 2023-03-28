import os 
import torch 
import json 
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import numpy as np


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def exist_in(short_str, list_of_string):
    for string in list_of_string:
        if short_str in string:
            return True 
    return False 


def clean_files(image_files, normal_files):
    """
    Not sure why some images do not have normal map annotations, thus delete these images from list. 

    The implementation here is inefficient .....  
    """
    new_image_files = []

    for image_file in image_files:
        image_file_basename = os.path.basename(image_file).split('.')[0]
        if exist_in(image_file_basename,normal_files):
            new_image_files.append(image_file)
    image_files = new_image_files


    # a sanity check 
    for image_file, normal_file in zip(image_files, normal_files):
        image_file_basename = os.path.basename(image_file).split('.')[0]
        normal_file_basename = os.path.basename(normal_file).split('.')[0]
        assert image_file_basename == normal_file_basename[:-7] 
    
    return image_files, normal_files




class NormalDataset():
    def __init__(self, image_rootdir, normal_rootdir, caption_path, prob_use_caption=1, image_size=512, random_flip=False):
        self.image_rootdir = image_rootdir
        self.normal_rootdir = normal_rootdir
        self.caption_path = caption_path
        self.prob_use_caption = prob_use_caption 
        self.image_size = image_size
        self.random_flip = random_flip


        # Image and normal files 
        image_files = recursively_read(rootdir=image_rootdir, must_contain="", exts=['png'])
        image_files.sort()
        normal_files = recursively_read(rootdir=normal_rootdir, must_contain="normal", exts=['npy'])
        normal_files.sort()

        image_files, normal_files = clean_files(image_files, normal_files)
        self.image_files = image_files
        self.normal_files = normal_files

        # Open caption json 
        with open(caption_path, 'r') as f:
            self.image_filename_to_caption_mapping = json.load(f)

  
        self.pil_to_tensor = transforms.PILToTensor()


    def total_images(self):
        return len(self)


    def __getitem__(self, index):

        image_path = self.image_files[index]
        
        out = {}

        out['id'] = index
        image = Image.open(image_path).convert("RGB")

        normal = np.load( self.normal_files[index] ) # -1 to 1 numpy array 
        normal = ((normal*0.5+0.5)*255).astype("uint8")
        normal = Image.fromarray(normal) # first convet normal map from array to image. So we can do crop etc easily 
        assert image.size == normal.size

        
        # - - - - - center_crop, resize and random_flip - - - - - - #  

        crop_size = min(image.size)
        image = TF.center_crop(image, crop_size)
        image = image.resize( (self.image_size, self.image_size) )

        normal = TF.center_crop(normal, crop_size)
        normal = normal.resize( (self.image_size, self.image_size) )


        if self.random_flip and random.random()<0.5:
            image = ImageOps.mirror(image)
            normal = ImageOps.mirror(normal)
        
        out['image'] = ( self.pil_to_tensor(image).float()/255 - 0.5 ) / 0.5
        out['normal'] = ( self.pil_to_tensor(normal).float()/255 - 0.5 ) / 0.5 # -1,1 is the actual range from numpy array annotation
        out['mask'] = torch.tensor(1.0) 
        
        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            out["caption"] = self.image_filename_to_caption_mapping[ os.path.basename(image_path) ]
        else:
            out["caption"] = ""

        return out


    def __len__(self):
        return len(self.image_files)


