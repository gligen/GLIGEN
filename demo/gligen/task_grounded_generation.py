import argparse
from PIL import Image, ImageDraw
from evaluator import Evaluator
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import default_device, instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from evaluator import set_alpha_scale, save_images, draw_masks_from_boxes
import numpy as np
import clip 
from functools import partial
import torchvision.transforms.functional as F
import random


device = default_device()


def alpha_generator(length, type=[1,0,0]):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    
    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas


def draw_box(img, locations):
    colors = ["red", "green", "blue", "olive", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    WW,HH = img.size
    for bid, box in enumerate(locations):
        draw.rectangle([box[0]*WW, box[1]*HH, box[2]*WW, box[3]*HH], outline =colors[bid % len(colors)], width=5)
    return img 

def load_ckpt(config, state_dict):
    model = instantiate_from_config(config.model).to(device).eval()
    autoencoder = instantiate_from_config(config.autoencoder).to(device).eval()
    text_encoder = instantiate_from_config(config.text_encoder).to(device).eval()
    diffusion = instantiate_from_config(config.diffusion).to(device)

    autoencoder.load_state_dict( state_dict["autoencoder"]  )
    text_encoder.load_state_dict( state_dict["text_encoder"]  )
    diffusion.load_state_dict( state_dict["diffusion"]  )

    model.load_state_dict(state_dict['model'])
    set_alpha_scale(model, config.alpha_scale)
    print("ckpt is loaded")

    return model, autoencoder, text_encoder, diffusion




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    feature_type = ['before','after_reproject'] # text feature, image feature 

    if is_image:
        image = input #Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].to(device) # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).to(device)  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if feature_type[1] == 'after_renorm':
            feature = feature*28.7
        if feature_type[1] == 'after_reproject':
            feature = project( feature, torch.load('gligen/projection_matrix').to(device).T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['pixel_values'] = torch.ones(1,3,224,224).to(device) # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        outputs = model(**inputs)
        feature = outputs.text_embeds if feature_type[0] == 'after' else outputs.text_model_output.pooler_output
    return feature



def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
def fire_clip(text_encoder, meta, batch=1, max_objs=30, clip_model=None):
    phrases = meta["phrases"]
    images = meta["images"]

    if clip_model is None:
        version = "openai/clip-vit-large-patch14"
        model = CLIPModel.from_pretrained(version).to(device)
        processor = CLIPProcessor.from_pretrained(version)
    else:
        version = "openai/clip-vit-large-patch14"
        assert clip_model['version'] == version
        model = clip_model['model']
        processor = clip_model['processor']

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)

    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )
    
    if len(text_features) > 0:
        text_features = torch.cat(text_features, dim=0)
        image_features = torch.cat(image_features, dim=0)

        for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
            boxes[idx] = torch.tensor(box)
            masks[idx] = 1
            text_embeddings[idx] = text_feature
            image_embeddings[idx] = image_feature
    
    
    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta["has_text_mask"], max_objs ),
        "image_masks" : masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta["has_image_mask"], max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }
    return batch_to_device(out, device) 





@torch.no_grad()
def grounded_generation_box(loaded_model_list, instruction, *args, **kwargs):
    
    # -------------- prepare model and misc --------------- # 
    model, autoencoder, text_encoder, diffusion = loaded_model_list
    batch_size = instruction["batch_size"]
    is_inpaint = True if "input_image" in instruction else False
    save_folder = os.path.join("create_samples", instruction["save_folder_name"])


    # -------------- set seed if required --------------- # 
    if instruction.get('fix_seed', False):
        random_seed = instruction['rand_seed']
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # ------------- prepare input for the model ------------- #
    batch = fire_clip(text_encoder, instruction, batch_size, clip_model=kwargs.get('clip_model', None))
    context = text_encoder.encode(  [instruction["prompt"]]*batch_size  )
    uc = text_encoder.encode( batch_size*[""] )
    # print(batch['boxes'])
    input = dict(x = None, 
                timesteps = None, 
                context = context, 
                boxes = batch['boxes'], 
                masks = batch['masks'], 
                text_masks = batch['text_masks'],
                image_masks = batch['image_masks'], 
                text_embeddings = batch["text_embeddings"], 
                image_embeddings = batch["image_embeddings"] )

    inpainting_mask = x0 = None # used for inpainting
    if is_inpaint:       
        input_image = F.pil_to_tensor(  instruction["input_image"] ) 
        input_image = ( input_image.float().unsqueeze(0).to(device) / 255 - 0.5 ) / 0.5
        x0 = autoencoder.encode( input_image )
        if instruction["actual_mask"] is not None:
            inpainting_mask = instruction["actual_mask"][None, None].expand(batch['boxes'].shape[0], -1, -1, -1).to(device)
        else:
            # inpainting_mask = draw_masks_from_boxes( batch['boxes'], (x0.shape[-2], x0.shape[-1])  ).to(device)
            actual_boxes = [instruction['inpainting_boxes_nodrop'] for _ in range(batch['boxes'].shape[0])]
            inpainting_mask = draw_masks_from_boxes(actual_boxes, (x0.shape[-2], x0.shape[-1])  ).to(device)
        # extra input for the model 
        masked_x0 = x0*inpainting_mask
        inpainting_extra_input = torch.cat([masked_x0,inpainting_mask], dim=1)
        input["inpainting_extra_input"] = inpainting_extra_input


    # ------------- prepare sampler ------------- #
    alpha_generator_func = partial(alpha_generator, type=instruction["alpha_type"])
    if False:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 

    # ------------- run sampler ... ------------- #
    shape = (batch_size, model.in_channels, model.image_size, model.image_size)
    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=instruction['guidance_scale'], mask=inpainting_mask, x0=x0)
    samples_fake = autoencoder.decode(samples_fake)


    # ------------- other logistics ------------- #
    os.makedirs( os.path.join(save_folder, 'images'), exist_ok=True)
    os.makedirs( os.path.join(save_folder, 'layout'), exist_ok=True)
    os.makedirs( os.path.join(save_folder, 'overlay'), exist_ok=True)

    start = len(  os.listdir(os.path.join(save_folder, 'images')) )
    image_ids = list(range(start,start+batch_size))
    print(image_ids)

    sample_list = []
    overlay_list = []
    for image_id, sample in zip(image_ids, samples_fake):
        # save in local
        img_name = str(int(image_id))+'.png'

        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))

        overlay = draw_box(sample.copy(), instruction['locations'])

        sample.save(os.path.join(save_folder, "images", img_name))
        overlay.save(os.path.join(save_folder, "overlay", img_name))

        # demo output 
        sample_list.append(sample)
        overlay_list.append(overlay)

    return sample_list, overlay_list
        


# if __name__ == "__main__":
    

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--folder", type=str,  default="create_samples", help="path to OUTPUT")
#     parser.add_argument("--official_ckpt", type=str,  default='../../../data/sd-v1-4.ckpt', help="")

#     parser.add_argument("--batch_size", type=int, default=10, help="This will overwrite the one in yaml.")
#     parser.add_argument("--no_plms", action='store_true')
#     parser.add_argument("--guidance_scale", type=float,  default=5, help="")
#     parser.add_argument("--alpha_scale", type=float,  default=1, help="scale tanh(alpha). If 0, the behaviour is same as original model")
#     args = parser.parse_args()

#     assert "sd-v1-4.ckpt" in args.official_ckpt, "only support for stable-diffusion model"


#     grounded_generation(args)

    



