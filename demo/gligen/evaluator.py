import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import default_device, instantiate_from_config
import numpy as np
import random
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import  DistributedSampler
import os 
from tqdm import tqdm
from distributed import get_rank, synchronize, get_world_size
from trainer import read_official_ckpt, batch_to_device, ImageCaptionSaver, wrap_loader #, get_padded_boxes
from PIL import Image
import math
import json

device = default_device()


def draw_masks_from_boxes(boxes,size):

    image_masks = [] 
    for box in boxes:
        image_mask = torch.ones(size[0],size[1])
        for bx in box:
            x0, x1 = bx[0]*size[0], bx[2]*size[0]
            y0, y1 = bx[1]*size[1], bx[3]*size[1]
            image_mask[int(y0):int(y1), int(x0):int(x1)] = 0 
        image_masks.append(image_mask)
    return torch.stack(image_masks).unsqueeze(1)
        


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale
            # print("scale:   ", alpha_scale)
            # print("attn:  ", module.alpha_attn)
            # print("dense:   ", module.alpha_dense)
            # print('  ')
            # print('  ')


def save_images(samples, image_ids, folder, to256):
    for sample, image_id in zip(samples, image_ids):
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        img_name = str(int(image_id))+'.png'
        img = Image.fromarray(sample.astype(np.uint8))
        if to256:
            img = img.resize( (256,256), Image.BICUBIC)
        img.save(os.path.join(folder,img_name))


def ckpt_to_folder_name(basename):
    name=""
    for s in basename:
        if s.isdigit():
            name+=s
    seen = round( int(name)/1000, 1 )
    return str(seen).ljust(4,'0')+'k'


class Evaluator:
    def __init__(self, config):

        self.config = config
        self.device = torch.device(device)
     

        # = = = = = create model and diffusion = = = = = #
        if self.config.ckpt != "real":

            self.model = instantiate_from_config(config.model).to(self.device)
            self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
            self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
            self.diffusion = instantiate_from_config(config.diffusion).to(self.device)

            # donot need to load official_ckpt for self.model here, since we will load from our ckpt
            state_dict = read_official_ckpt( os.path.join(config.DATA_ROOT, config.official_ckpt_name)  )
            self.autoencoder.load_state_dict( state_dict["autoencoder"]  )
            self.text_encoder.load_state_dict( state_dict["text_encoder"]  )
            self.diffusion.load_state_dict( state_dict["diffusion"]  )


        # = = = = = load from our ckpt = = = = = #
        if self.config.ckpt == "real":
            print("Saving all real images...")
            self.just_save_real = True 
        else:
            checkpoint = torch.load(self.config.ckpt, map_location="cpu")
            which_state = 'ema' if 'ema' in checkpoint else "model"
            which_state = which_state if config.which_state is None else config.which_state
            self.model.load_state_dict(checkpoint[which_state])
            print("ckpt is loaded")
            self.just_save_real = False
            set_alpha_scale(self.model, self.config.alpha_scale)

            self.autoencoder.eval()
            self.model.eval()
            self.text_encoder.eval()
        

        # = = = = = create data = = = = = #
        self.dataset_eval = ConCatDataset(config.val_dataset_names, config.DATA_ROOT, config.which_embedder, train=False)
        print("total eval images: ", len(self.dataset_eval))
        sampler = DistributedSampler(self.dataset_eval,shuffle=False) if config.distributed else None
        loader_eval = DataLoader( self.dataset_eval,batch_size=config.batch_size, 
                                                    num_workers=config.workers, 
                                                    pin_memory=True, 
                                                    sampler=sampler,
                                                    drop_last=False) # shuffle default is False
        self.loader_eval = loader_eval
        

        # = = = = = create output folder = = = = = #
        folder_name = ckpt_to_folder_name(os.path.basename(config.ckpt))
        self.outdir = os.path.join(config.OUTPUT_ROOT, folder_name)
        self.outdir_real = os.path.join(self.outdir,'real')
        self.outdir_fake = os.path.join(self.outdir,'fake')
        if config.to256:
            self.outdir_real256 = os.path.join(self.outdir,'real256')
            self.outdir_fake256 = os.path.join(self.outdir,'fake256')
        synchronize() # if rank0 is faster, it may mkdir before the other rank call os.listdir()
        if get_rank() == 0:
            os.makedirs(self.outdir, exist_ok=True)
            os.makedirs(self.outdir_real, exist_ok=True)
            os.makedirs(self.outdir_fake, exist_ok=True)
            if config.to256:
                os.makedirs(self.outdir_real256, exist_ok=True)
                os.makedirs(self.outdir_fake256, exist_ok=True)
        print(self.outdir) # double check 

        self.evaluation_finished = False
        if os.path.exists(  os.path.join(self.outdir,'score.txt')  ):
            self.evaluation_finished = True 
            

    def alread_saved_this_batch(self, batch):
        existing_real_files = os.listdir( self.outdir_real  )
        existing_fake_files = os.listdir( self.outdir_fake  )
        status = []
        for image_id in batch["id"]:
            img_name = str(int(image_id))+'.png'
            status.append(img_name in existing_real_files)
            status.append(img_name in existing_fake_files)
        return all(status)


    @torch.no_grad()
    def start_evaluating(self):

        iterator = tqdm( self.loader_eval, desc='Evaluating progress')   
        for batch in iterator:

            #if not self.alread_saved_this_batch(batch):
            if True:

                batch_to_device(batch, self.device)
                batch_size = batch["image"].shape[0]
                samples_real = batch["image"]

                if self.just_save_real:                    
                    samples_fake = None
                else:
                    uc = self.text_encoder.encode( batch_size*[""] )
                    context = self.text_encoder.encode(  batch["caption"]  )
                
                    image_mask = x0 = None 
                    if self.config.inpaint:
                        image_mask = draw_masks_from_boxes( batch['boxes'], self.model.image_size  ).to(device)
                        x0 = self.autoencoder.encode( batch["image"] )

                    shape = (batch_size, self.model.in_channels, self.model.image_size, self.model.image_size)
                    if self.config.no_plms:
                        sampler = DDIMSampler(self.diffusion, self.model)
                        steps = 250 
                    else:
                        sampler = PLMSSampler(self.diffusion, self.model)
                        steps = 50 

                    input = dict( x=None, timesteps=None, context=context, boxes=batch['boxes'], masks=batch['masks'], positive_embeddings=batch["positive_embeddings"] )
                    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=self.config.guidance_scale, mask=image_mask, x0=x0)
                    samples_fake = self.autoencoder.decode(samples_fake)


                save_images(samples_real, batch['id'], self.outdir_real, to256=False )
                if self.config.to256:
                    save_images(samples_real, batch['id'], self.outdir_real256, to256=True )

                if samples_fake is not None:
                    save_images(samples_fake, batch['id'], self.outdir_fake, to256=False )
                    if self.config.to256:
                        save_images(samples_fake, batch['id'], self.outdir_fake256, to256=True )


    def fire_fid(self):
        paths = [self.outdir_real, self.outdir_fake]
        if self.config.to256:
            paths = [self.outdir_real256, self.outdir_fake256]
        


        
        




    

    
            











