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




class SemanticDataset():
    def __init__(self, image_rootdir, sem_rootdir, caption_path, prob_use_caption=1, image_size=512, random_flip=False):
        self.image_rootdir = image_rootdir
        self.sem_rootdir = sem_rootdir
        self.caption_path = caption_path
        self.prob_use_caption = prob_use_caption 
        self.image_size = image_size
        self.random_flip = random_flip

        
        # Image and normal files 
        image_files = recursively_read(rootdir=image_rootdir, must_contain="", exts=['jpg'])
        image_files.sort()
        sem_files = recursively_read(rootdir=sem_rootdir, must_contain="", exts=['png'])
        sem_files.sort()
        

        self.image_files = image_files
        self.sem_files = sem_files

        # Open caption json 
        with open(caption_path, 'r') as f:
            self.image_filename_to_caption_mapping = json.load(f)

  
        assert len(self.image_files) == len(self.sem_files) == len(self.image_filename_to_caption_mapping)
        self.pil_to_tensor = transforms.PILToTensor()


    def total_images(self):
        return len(self)


    def __getitem__(self, index):

        image_path = self.image_files[index]
        
        out = {}

        out['id'] = index
        image = Image.open(image_path).convert("RGB")
        sem = Image.open( self.sem_files[index]  ).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation 

        assert image.size == sem.size

        
        # - - - - - center_crop, resize and random_flip - - - - - - #  

        crop_size = min(image.size)
        image = TF.center_crop(image, crop_size)
        image = image.resize( (self.image_size, self.image_size) )

        sem = TF.center_crop(sem, crop_size)
        sem = sem.resize( (self.image_size, self.image_size), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly

        if self.random_flip and random.random()<0.5:
            image = ImageOps.mirror(image)
            sem = ImageOps.mirror(sem)       

        sem = self.pil_to_tensor(sem)[0,:,:]

        input_label = torch.zeros(152, self.image_size, self.image_size)
        sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

        out['image'] = ( self.pil_to_tensor(image).float()/255 - 0.5 ) / 0.5
        out['sem'] = sem
        out['mask'] = torch.tensor(1.0) 


        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            out["caption"] = self.image_filename_to_caption_mapping[ os.path.basename(image_path) ]
        else:
            out["caption"] = ""

        return out


    def __len__(self):
        return len(self.image_files)


