import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

device = "cuda"


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
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
    if type == None:
        type = [1,0,0]

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



def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image



@torch.no_grad()
def prepare_batch_kp(meta, batch=1, max_persons_per_image=8):
    
    points = torch.zeros(max_persons_per_image*17,2)
    idx = 0 
    for this_person_kp in meta["locations"]:
        for kp in this_person_kp:
            points[idx,0] = kp[0]
            points[idx,1] = kp[1]
            idx += 1
    
    # derive masks from points
    masks = (points.mean(dim=1)!=0) * 1 
    masks = masks.float()

    out = {
        "points" : points.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
    }

    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_hed(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    hed_edge = Image.open(meta['hed_image']).convert("RGB")
    hed_edge = crop_and_resize(hed_edge)
    hed_edge = ( pil_to_tensor(hed_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "hed_edge" : hed_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_canny(meta, batch=1):
    """ 
    The canny edge is very sensitive since I set a fixed canny hyperparamters; 
    Try to use the same setting to get edge 

    img = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img,100,200)
    edges = PIL.Image.fromarray(edges)

    """
    
    pil_to_tensor = transforms.PILToTensor()

    canny_edge = Image.open(meta['canny_image']).convert("RGB")
    canny_edge = crop_and_resize(canny_edge)

    canny_edge = ( pil_to_tensor(canny_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "canny_edge" : canny_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_depth(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    depth = Image.open(meta['depth']).convert("RGB")
    depth = crop_and_resize(depth)
    depth = ( pil_to_tensor(depth).float()/255 - 0.5 ) / 0.5

    out = {
        "depth" : depth.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 



@torch.no_grad()
def prepare_batch_normal(meta, batch=1):
    """
    We only train normal model on the DIODE dataset which only has a few scene.

    """
    
    pil_to_tensor = transforms.PILToTensor()

    normal = Image.open(meta['normal']).convert("RGB")
    normal = crop_and_resize(normal)
    normal = ( pil_to_tensor(normal).float()/255 - 0.5 ) / 0.5

    out = {
        "normal" : normal.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 





def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    return labelmap_rgb

@torch.no_grad()
def prepare_batch_sem(meta, batch=1):

    pil_to_tensor = transforms.PILToTensor()

    sem = Image.open( meta['sem']  ).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation 
    sem = TF.center_crop(sem, min(sem.size))
    sem = sem.resize( (512, 512), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly
    try:
        sem_color = colorEncode(np.array(sem), loadmat('color150.mat')['colors'])
        Image.fromarray(sem_color).save("sem_vis.png")
    except:
        pass 
    sem = pil_to_tensor(sem)[0,:,:]
    input_label = torch.zeros(152, 512, 512)
    sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

    out = {
        "sem" : sem.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 



@torch.no_grad()
def run(meta, config, starting_noise=None):

    # - - - - - prepare models - - - - - # 
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])



    # - - - - - update config from args - - - - - # 
    config.update( vars(args) )
    config = OmegaConf.create(config)


    # - - - - - prepare batch - - - - - #
    if "keypoint" in meta["ckpt"]:
        batch = prepare_batch_kp(meta, config.batch_size)
    elif "hed" in meta["ckpt"]:
        batch = prepare_batch_hed(meta, config.batch_size)
    elif "canny" in meta["ckpt"]:
        batch = prepare_batch_canny(meta, config.batch_size)
    elif "depth" in meta["ckpt"]:
        batch = prepare_batch_depth(meta, config.batch_size)
    elif "normal" in meta["ckpt"]:
        batch = prepare_batch_normal(meta, config.batch_size)
    elif "sem" in meta["ckpt"]:
        batch = prepare_batch_sem(meta, config.batch_size)
    else:
        batch = prepare_batch(meta, config.batch_size)
    context = text_encoder.encode(  [meta["prompt"]]*config.batch_size  )
    uc = text_encoder.encode( config.batch_size*[""] )
    if args.negative_prompt is not None:
        uc = text_encoder.encode( config.batch_size*[args.negative_prompt] )


    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 


    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None # used as model input 
    if "input_image" in meta:
        # inpaint mode 
        assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        
        inpainting_mask = draw_masks_from_boxes( batch['boxes'], model.image_size  ).cuda()
        
        input_image = F.pil_to_tensor( Image.open(meta["input_image"]).convert("RGB").resize((512,512)) ) 
        input_image = ( input_image.float().unsqueeze(0).cuda() / 255 - 0.5 ) / 0.5
        z0 = autoencoder.encode( input_image )
        
        masked_z = z0*inpainting_mask
        inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
    

    # - - - - - input for gligen - - - - - #
    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    input = dict(
                x = starting_noise, 
                timesteps = None, 
                context = context, 
                grounding_input = grounding_input,
                inpainting_extra_input = inpainting_extra_input,
                grounding_extra_input = grounding_extra_input,

            )


    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)


    # - - - - - save - - - - - #
    output_folder = os.path.join( args.folder,  meta["save_folder_name"])
    os.makedirs( output_folder, exist_ok=True)

    start = len( os.listdir(output_folder) )
    image_ids = list(range(start,start+config.batch_size))
    print(image_ids)
    for image_id, sample in zip(image_ids, samples_fake):
        img_name = str(int(image_id))+'.png'
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))
        sample.save(  os.path.join(output_folder, img_name)   )




if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="generation_samples", help="root folder for output")


    parser.add_argument("--batch_size", type=int, default=5, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    #parser.add_argument("--negative_prompt", type=str,  default=None, help="")
    args = parser.parse_args()
    


    meta_list = [ 

        # - - - - - - - - GLIGEN on text grounding for generation - - - - - - - - # 
        dict(
            ckpt = "../gligen_checkpoints/checkpoint_generation_text.pth",
            prompt = "a teddy bear sitting next to a bird",
            phrases = ['a teddy bear', 'a bird'],
            locations = [ [0.0,0.09,0.33,0.76], [0.55,0.11,1.0,0.8] ],
            alpha_type = [0.3, 0.0, 0.7],
            save_folder_name="generation_box_text"
        ), 


        # - - - - - - - - GLIGEN on text grounding for inpainting - - - - - - - - # 
        dict(
            ckpt = "../gligen_checkpoints/checkpoint_inpainting_text.pth",
            input_image = "inference_images/dalle2_museum.jpg",
            prompt = "a corgi and a cake",
            phrases =   ['corgi', 'cake'],
            locations = [ [0.25, 0.28, 0.42, 0.52], [0.14, 0.58, 0.58, 0.92], ], # mask will be derived from box 
            save_folder_name="inpainting_box_text"
        ),


        # - - - - - - - - GLIGEN on image grounding for generation - - - - - - - - # 
        dict(
            ckpt = "../gligen_checkpoints/checkpoint_generation_text_image.pth",
            prompt = "an alarm clock sitting on the beach",
            images = ['inference_images/clock.png'],
            phrases = ['alarm clock'],
            locations = [ [0.0,0.09,0.53,0.76] ],
            alpha_type = [1.0, 0.0, 0.0],
            save_folder_name="generation_box_image"
        ),



        # - - - - - - - - GLIGEN on text and style grounding for generation - - - - - - - - # 
        dict(
            ckpt = "../gligen_checkpoints/checkpoint_generation_text_image.pth",
            prompt = "a brick house in the woods, anime, oil painting",
            phrases =   ['a brick house',            'placehoder'],
            images =    ['inference_images/placeholder.png', 'inference_images/style_golden.jpg'],
            locations = [ [0.4,0.2,1.0,0.8],         [0.0, 1.0, 0.0, 1.0] ],
            alpha_type = [1, 0, 0],  
            text_mask = [1,0],  # the second text feature will be masked 
            image_mask =[0,1],  # the first image feature will be masked
            save_folder_name="generation_box_text_style"
        ), 


        # - - - - - - - - GLIGEN on image grounding for inpainting - - - - - - - - # 
        dict(
            ckpt = "../gligen_checkpoints/checkpoint_inpainting_text_image.pth",
            input_image = "inference_images/beach.jpg",
            prompt = "a bigben on the beach",
            images = [ 'inference_images/bigben.jpg'],
            locations = [ [0.18, 0.08, 0.62, 0.75] ], # mask will be derived from box 
            save_folder_name="inpainting_box_image"
        ),



        # - - - - - - - - GLIGEN on hed grounding for generation - - - - - - - - # 
        dict(
            ckpt ="../gligen_checkpoints/checkpoint_generation_hed.pth",
            prompt = "a man is eating breakfast",  
            hed_image = 'inference_images/hed_man_eat.png',
            save_folder_name="hed",
            alpha_type = [0.9, 0, 0.1], 
        ),




        # - - - - - - - - GLIGEN on canny grounding for generation - - - - - - - - # 
        dict(
            ckpt ="../gligen_checkpoints/checkpoint_generation_canny.pth",
            prompt = "A Humanoid Robot Designed for Companionship", 
            canny_image = 'inference_images/canny_robot.png',
            alpha_type = [0.9, 0, 0.1], 
            save_folder_name="canny"
        ),




        # - - - - - - - - GLIGEN on normal grounding for generation - - - - - - - - # 
        dict(
            ckpt ="../gligen_checkpoints/checkpoint_generation_normal.pth",
            prompt = "a large tree with no leaves in front of a building", # 
            normal = 'inference_images/normal_tree_building.jpg', # a normal map 
            alpha_type = [0.7, 0, 0.3], 
            save_folder_name="normal",
        ),


        # - - - - - - - - GLIGEN on depth grounding for generation - - - - - - - - # 
        dict(
            ckpt ="../gligen_checkpoints/checkpoint_generation_depth.pth",
            prompt = "a Vibrant colorful Bird Sitting on Tree Branch", # 
            depth = 'inference_images/depth_bird.png', 
            alpha_type = [0.7, 0, 0.3], 
            save_folder_name="depth"
        ),


        # - - - - - - - - GLIGEN on sem grounding for generation - - - - - - - - # 
        dict(
            ckpt ="../gligen_checkpoints/checkpoint_generation_sem.pth",
            prompt = "a living room filled with lots of furniture and plants", # 
            sem = 'inference_images/sem_ade_living_room.png', # ADE raw annotation  
            alpha_type = [0.7, 0, 0.3], 
            save_folder_name="sem"
        ),



        # - - - - - - - - GLIGEN on keypoint grounding for generation - - - - - - - - # 
        dict(
            ckpt = "../gligen_checkpoints/checkpoint_generation_keypoint.pth",
            prompt = "A young man and a small boy are talking",
            locations = [  
                            [
                                [0.7598, 0.2542],
                                [0.7431, 0.2104],
                                [0.8118, 0.2021],
                                [0.0000, 0.0000],
                                [0.9514, 0.1813],
                                [0.7806, 0.2917],
                                [0.0000, 0.0000],
                                [0.6785, 0.5125],
                                [0.0000, 0.0000],
                                [0.5389, 0.6479],
                                [0.6785, 0.6750],
                                [0.7973, 0.7042],
                                [0.0000, 0.0000],
                                [0.6181, 0.7375],
                                [0.9764, 0.8458],
                                [0.0000, 0.0000],
                                [0.0000, 0.0000]
                            ], 

                            [
                                [0.2681, 0.4313],
                                [0.2514, 0.3979],
                                [0.0000, 0.0000],
                                [0.0785, 0.3854],
                                [0.0000, 0.0000],
                                [0.0910, 0.5583],
                                [0.0000, 0.0000],
                                [0.1243, 0.8479],
                                [0.0000, 0.0000],
                                [0.0000, 0.0000],
                                [0.0000, 0.0000],
                                [0.0000, 0.0000],
                                [0.0000, 0.0000],
                                [0.2410, 0.8146],
                                [0.1202, 0.6146],
                                [0.0000, 0.0000],
                                [0.2743, 0.7188]
                            ], 

             ],  # from id=18150 val set in coco2017k
            alpha_type = [0.3, 0.0, 0.7],
            save_folder_name="keypoint"
        ),



    ]


    starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)
    starting_noise = None
    for meta in meta_list:
        run(meta, args, starting_noise)

    



