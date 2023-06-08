import torch 
# from ldm.modules.encoders.modules import FrozenCLIPEmbedder
# from ldm.modules.encoders.modules import BERTEmbedder
from transformers import CLIPProcessor, CLIPModel
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import os 
import math
import clip 
from PIL import Image
from torchvision import transforms
import multiprocessing
from zipfile import ZipFile 
from io import BytesIO


def split_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clean_annotations(annotations):
    for anno in annotations:
        anno.pop("segmentation", None)
        anno.pop("area", None)
        anno.pop("iscrowd", None)


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



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


class Base():
    def __init__(self, image_root):
        self.image_root = image_root
        self.use_zip = True if image_root[-4:] == ".zip" else False 
        if self.use_zip:
            self.zip_dict = {}

        # This is CLIP mean and std
        # Since our image is cropped from bounding box, thus we directly resize to 224*224 without center_crop to keep obj whole information. 
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711] )
        ])

    def fetch_zipfile(self, ziproot):
        pid = multiprocessing.current_process().pid # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(ziproot)
        zip_file = self.zip_dict[pid]
        return zip_file

    def fetch_image(self, file_name):
        if self.use_zip:
            zip_file = self.fetch_zipfile(self.image_root)
            image = Image.open( BytesIO(zip_file.read(file_name)) ).convert('RGB')
        else:
            image = Image.open(  os.path.join(self.image_root,file_name)   ).convert('RGB')
        return image




class GroundedTextImageDataset_Detection(Base):
    def __init__(self, instances_json_path, image_root, chunk_idx, total_chunk):
        super().__init__(image_root)

        self.image_root = image_root
        self.instances_json_path = instances_json_path


        # Load all jsons 
        with open(instances_json_path, 'r') as f:
            instances_data = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
        clean_annotations(instances_data["annotations"])
        self.instances_data = instances_data
        self.annotations = instances_data["annotations"] # 25407598 total for O365

        # Misc  
        self.image_ids = [] # main list for selecting images
        self.image_id_to_filename = {} # file names used to read image
        for image_data in self.instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
        
        # All category names 
        self.object_idx_to_name = {} 
        for category_data in self.instances_data['categories']:
            self.object_idx_to_name[category_data['id']] = category_data['name']

        if total_chunk is not None:
            chunk_size = math.ceil(len(self.annotations) / total_chunk)
            self.annotations = list( split_chunks(self.annotations,chunk_size) )[chunk_idx]

    def __getitem__(self, index):

        anno = self.annotations[index]
        anno_id = anno["id"]
        X,Y,W,H = anno['bbox']

        filename = self.image_id_to_filename[anno['image_id']]
        image = self.fetch_image(filename)
        image_crop = self.preprocess(  image.crop( (X,Y,X+W,Y+H) ).resize( (224,224), Image.BICUBIC )  )
        positive = self.object_idx_to_name[anno['category_id']]

        return {'positive':positive,  'anno_id':anno_id, 'image_crop':image_crop}

    def __len__(self):
        return len(self.annotations)
	









class GroundedTextImageDataset_Grounding(Base):
    def __init__(self, json_path, image_root, chunk_idx, total_chunk):
        super().__init__(image_root)

        self.image_root = image_root
        
        with open(json_path, 'r') as f:
            json_raw = json.load(f) # keys: 'info', 'images', 'licenses', 'categories', 'annotations'
        self.data = json_raw["images"] # donot name it images, which is misleading
        self.annotations = json_raw["annotations"]

        if total_chunk is not None:
            chunk_size = math.ceil(len(self.annotations) / total_chunk)
            self.annotations = list( split_chunks(self.annotations,chunk_size) )[chunk_idx]

        self.data = {  datum["id"]:datum for datum in self.data }
    
    def __getitem__(self, index):

        anno = self.annotations[index]
        anno_id = anno["id"]
        X,Y,W,H = anno['bbox']

        caption = self.data[ anno["image_id"]  ]["caption"]
        file_name = self.data[ anno["image_id"]  ]["file_name"]
        image = self.fetch_image(file_name)
        image_crop = self.preprocess(  image.crop( (X,Y,X+W,Y+H) ).resize( (224,224), Image.BICUBIC )  )

        positive = ''
        for (start, end) in anno['tokens_positive']:
            positive += caption[start:end]
            positive += ' '       
        positive = positive[:-1]

        return {'positive':positive,  'anno_id':anno_id, 'image_crop':image_crop}
       
    def __len__(self):
        return len(self.annotations)



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =#




@torch.no_grad()
def fire_clip_before_after(loader, folder):
    """
    This will save CLIP feature before/after projection. 

    before projection text feature is the one used by stable-diffsuion. 
    For before_projection, its feature is unmormalized. 
    For after_projection, which is CLIP aligned space, its feature is normalized.   

    You may want to use project / inv_project to project image feature into CLIP text space. (Haotian's idea)
    """
    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)
    #projection_matrix = torch.load('projection_matrix').cuda()

    os.makedirs(  os.path.join(folder, 'text_features_before'),  exist_ok=True   )
    os.makedirs(  os.path.join(folder, 'text_features_after'),  exist_ok=True   )
    os.makedirs(  os.path.join(folder, 'image_features_before'), exist_ok=True   )
    os.makedirs(  os.path.join(folder, 'image_features_after'), exist_ok=True   )

    
    for batch in tqdm(loader):

        inputs = processor(text=batch['positive'],  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = batch['image_crop'].cuda() # we use our own preprocessing without center_crop 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)

        text_before_features = outputs.text_model_output.pooler_output # before projection feature
        text_after_features = outputs.text_embeds # normalized after projection feature (CLIP aligned space)

        image_before_features = outputs.vision_model_output.pooler_output # before projection feature
        image_after_features = outputs.image_embeds # normalized after projection feature (CLIP aligned space)

        for idx, text_before, text_after, image_before, image_after  in zip(batch["anno_id"], text_before_features, text_after_features, image_before_features, image_after_features):
            
            save_name = os.path.join(folder, 'text_features_before', str(int(idx)) )
            torch.save(text_before.clone().cpu(), save_name)

            save_name = os.path.join(folder, 'text_features_after', str(int(idx)) )
            torch.save(text_after.clone().cpu(), save_name)

            save_name = os.path.join(folder, 'image_features_before', str(int(idx)) )
            torch.save(image_before.clone().cpu(), save_name)

            save_name = os.path.join(folder, 'image_features_after', str(int(idx)) )
            torch.save(image_after.clone().cpu(), save_name)





@torch.no_grad()
def fire_clip_after(loader, folder):

    model, preprocess = clip.load("ViT-L/14", device='cuda')
    model.eval()


    os.makedirs(  os.path.join(folder, 'image_features'), exist_ok=True   )
    os.makedirs(  os.path.join(folder, 'text_features'),  exist_ok=True   )
    
    for batch in tqdm(loader):

        image_features = model.encode_image(batch["image_crop"].cuda())
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)

        text = clip.tokenize(batch["positive"]).to('cuda')
        text_features = model.encode_text(text)
        text_features = text_features / torch.norm(text_features, dim=1, keepdim=True)

        # (image_features*text_features).sum(dim=1).mean()  0.198 
        for idx, image_feature, text_feature in zip(batch["anno_id"], image_features, text_features):
            
            save_name = os.path.join(folder, 'image_features', str(int(idx)) )
            torch.save(image_feature.clone(), save_name)

            save_name = os.path.join(folder, 'text_features', str(int(idx)) )
            torch.save(text_feature.clone(), save_name)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_o365", type=bool, default=False)
    parser.add_argument("--only_after", type=bool, default=False, help='if false, then both before and after projection CLIP feature will be saved')
    parser.add_argument("--chunk_idx", type=int, default=None)
    parser.add_argument("--total_chunk", type=int, default=None)
    parser.add_argument("--json_path", type=str,  default="../../DATA/LAION-HAOTIAN-DOWNLOAD-TEST/laion_1142.json", help="")
    parser.add_argument("--image_root", type=str,  default="../../DATA/LAION-HAOTIAN-DOWNLOAD-TEST/", help="")
    parser.add_argument("--folder", type=str,  default="out", help="")
    args = parser.parse_args()



    if args.total_chunk is not None:
        assert args.chunk_idx in list(range(args.total_chunk))

    if args.is_o365:
        dataset = GroundedTextImageDataset_Detection(args.json_path, args.image_root, args.chunk_idx, args.total_chunk)
    else:
        dataset = GroundedTextImageDataset_Grounding(args.json_path, args.image_root, args.chunk_idx, args.total_chunk)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=False)
    os.makedirs(args.folder, exist_ok=True)

    if args.only_after:
        fire_clip_after(loader, args.folder)
    else:
        fire_clip_before_after(loader, args.folder)


