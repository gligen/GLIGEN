import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
import random
import time 
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
import shutil
import torchvision
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from distributed import get_rank, synchronize, get_world_size
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy
try:
    from apex import amp 
except:
    pass 
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #

class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, images, real, captions, seen):
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'.png')
        torchvision.utils.save_image( images, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range )
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_real.png')
        torchvision.utils.save_image( real, save_path, nrow=self.nrow)

        assert images.shape[0] == len(captions)

        save_path = os.path.join(self.base_path, 'captions.txt')
        with open(save_path, "a") as f:
            f.write( str(seen).zfill(8) + ':\n' )    
            for cap in captions:
                f.write( cap + '\n' )  
            f.write( '\n' ) 



def read_official_ckpt(ckpt_path):      
    "Read offical pretrained ckpt and convert into my style" 
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     
    return out 


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

           
def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    #curr_folder_name = os.getcwd().split("/")[-1]
    name = os.path.join( OUTPUT_ROOT, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('ckpt found '+ potential_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 






class Trainer:
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            torch.save(  vars(config),  os.path.join(self.name, "config_dict.pth")     )

        # = = = = = = = = = = create model and diffusion = = = = = = = = = = #
        self.model = instantiate_from_config(config.model).to(self.device)
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)


        state_dict = read_official_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
        missing_keys, unexpected_keys = self.model.load_state_dict( state_dict["model"], strict=False  )
        assert unexpected_keys == []
        original_params_names = list( state_dict["model"].keys()  )
        self.autoencoder.load_state_dict( state_dict["autoencoder"]  )
        self.text_encoder.load_state_dict( state_dict["text_encoder"]  )
        self.diffusion.load_state_dict( state_dict["diffusion"]  )
 
        self.autoencoder.eval()
        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)



        # = = load from ckpt: (usually second stage whole model finetune) = = #
        if self.config.ckpt is not None:
            first_stage_ckpt = torch.load(self.config.ckpt, map_location="cpu")
            self.model.load_state_dict(first_stage_ckpt["model"])




        # = = = = = = = = = = create opt = = = = = = = = = = #
        print("  ")
        print("IMPORTANT: following code decides which params trainable!")
        print("  ")

        if self.config.whole:
            print("Entire model is trainable")
            params = list(self.model.parameters())
        else:
            print("Only new added components will be updated")
            params = []
            trainable_names = []
            for name, p in self.model.named_parameters():
                if ("transformer_blocks" in name) and ("fuser" in name):
                    params.append(p) 
                    trainable_names.append(name)
                elif  "position_net" in name:
                    params.append(p) 
                    trainable_names.append(name)
                else:
                    # all new added trainable params have to be haddled above
                    # otherwise it will trigger the following error  
                    assert name in original_params_names, name  
            
            all_params_name = list( self.model.state_dict().keys()  )
            assert set(all_params_name) == set(trainable_names + original_params_names) 

        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        count_params(params)
        
        self.master_params = list(self.model.parameters()) # note: you cannot assign above params as master_params since that is only trainable one
        
        if config.enable_ema:
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()

        # = = = = = = = = = = create scheduler = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False 



        # = = = = = = = = = = create data = = = = = = = = = = #  
        train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None
        dataset_train = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, config.which_embedder, train=True, repeats=train_dataset_repeats)
        sampler = DistributedSampler(dataset_train) if config.distributed else None 
        loader_train = DataLoader( dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=(sampler is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler)
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)

        if get_rank() == 0:
            total_image = dataset_train.total_images()
            print("Total training images: ", total_image)     
        

        # = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = #
        self.starting_iter = 0  
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            if config.enable_ema:
                self.ema.load_state_dict(checkpoint["ema"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.starting_iter = checkpoint["iters"]
            if self.starting_iter >= config.total_iters:
                synchronize()
                print("Training finished. Start exiting")
                exit()


        # = = = = = misc = = = = = #    
        if get_rank() == 0:
            print("Actual total need see images is: ", config.total_iters*config.total_batch_size)
            print("Equivalent training epoch is: ", (config.total_iters*config.total_batch_size) / len(dataset_train) )         
            self.image_caption_saver = ImageCaptionSaver(self.name)
            # self.counter = Counter(config.total_batch_size, config.save_every_images)

        if config.use_o2:
            self.model, self.opt = amp.initialize(self.model, self.opt, opt_level="O2")
            self.model.use_o2 = True


        # = = = = = wrap into ddp = = = = = #
        if config.distributed:
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False )





    @torch.no_grad()
    def get_input(self, batch):

        z = self.autoencoder.encode( batch["image"] )

        context = self.text_encoder.encode( batch["caption"]  )

        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, self.config.resample_step_gamma) * 1000).long()
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
        
        return z, t, context 


    def run_one_step(self, batch):
        x_start, t, context = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        input = dict(x = x_noisy, 
                     timesteps = t, 
                     context = context, 
                     boxes = batch['boxes'],
                     masks = batch['masks'], 
                     text_masks = batch['text_masks'],
                     image_masks = batch['image_masks'], 
                     text_embeddings = batch["text_embeddings"], 
                     image_embeddings = batch["image_embeddings"]  )
        model_output = self.model(input)
        
        loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight

        self.loss_dict = {"loss": loss.item()}

        return loss 
        


    def start_training(self):

        if not self.config.use_o2:
            # use pytorch mixed training which is similar to o1 but faster
            scaler = torch.cuda.amp.GradScaler()


        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',  disable=get_rank() != 0 )
        self.model.train()
        for iter_idx in iterator: # note: iter_idx is not from 0 if resume training
            self.iter_idx = iter_idx

            self.opt.zero_grad()
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            if self.config.use_o2:
                loss = self.run_one_step(batch)
                with amp.scale_loss(loss, self.opt) as scaled_loss:
                    scaled_loss.backward()
                self.opt.step()
            else:
                enabled = True if self.config.use_mixed else False
                with torch.cuda.amp.autocast(enabled=enabled):  # with torch.autocast(enabled=True):
                    loss = self.run_one_step(batch)
                scaler.scale(loss).backward() 
                scaler.step(self.opt)
                scaler.update()


            self.scheduler.step()

            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)


            if (get_rank() == 0):
                if (iter_idx % 10 == 0):
                    self.log_loss() 
                if (iter_idx == 0)  or  ( iter_idx % self.config.save_every_iters == 0 )  or  (iter_idx == self.config.total_iters-1):
                    self.save_ckpt_and_result()
            synchronize()

        
        synchronize()
        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    

    @torch.no_grad()
    def save_ckpt_and_result(self):

        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if not self.config.disable_inference_in_training:
            # Do a quick inference on one training batch 
            batch_here = self.config.batch_size
            batch = sub_batch( next(self.loader_train), batch_here)
            batch_to_device(batch, self.device)

            
            real_images_with_box_drawing = [] # we save this durining trianing for better visualization
            for i in range(batch_here):
                temp_data = {"image": batch["image"][i], "boxes":batch["boxes"][i]}
                im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                real_images_with_box_drawing.append(im)
            real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)

            
            uc = self.text_encoder.encode( batch_here*[""] )
            context = self.text_encoder.encode(  batch["caption"]  )
            
            ddim_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)      
            shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)
            input = dict( x = None, 
                          timesteps = None, 
                          context = context, 
                          boxes = batch['boxes'], 
                          masks = batch['masks'],
                          text_masks = batch['text_masks'], 
                          image_masks = batch['image_masks'], 
                          text_embeddings = batch["text_embeddings"], 
                          image_embeddings = batch["image_embeddings"] )
            samples = ddim_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)
            
            # old 
            # autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
            # autoencoder_wo_wrapper = autoencoder_wo_wrapper.cpu() # To save GPU 
            # samples = autoencoder_wo_wrapper.decode(samples.cpu())
            # autoencoder_wo_wrapper = autoencoder_wo_wrapper.to(self.device)

            # new 
            autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
            samples = autoencoder_wo_wrapper.decode(samples).cpu()

            self.image_caption_saver(samples, real_images_with_box_drawing,  batch["caption"], iter_name)

        ckpt = dict(model = model_wo_wrapper.state_dict(),
                    opt = self.opt.state_dict(),
                    scheduler= self.scheduler.state_dict(),
                    iters = self.iter_idx+1 )
        if self.config.enable_ema:
            ckpt["ema"] = self.ema.state_dict()
        torch.save( ckpt, os.path.join(self.name, "checkpoint_"+str(iter_name).zfill(8)+".pth") )
        torch.save( ckpt, os.path.join(self.name, "checkpoint_latest.pth") )








