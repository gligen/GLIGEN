import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F



class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        # -------------------------------------------------------------- #
        self.linears_text = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linears_image = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        # -------------------------------------------------------------- #
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_image_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, boxes, masks, text_masks, image_masks, text_embeddings, image_embeddings):
        B, N, _ = boxes.shape 
        masks = masks.unsqueeze(-1) # B*N*1 
        text_masks = text_masks.unsqueeze(-1) # B*N*1 
        image_masks = image_masks.unsqueeze(-1) # B*N*1
        
        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        text_null  = self.null_text_feature.view(1,1,-1) # 1*1*C
        image_null = self.null_image_feature.view(1,1,-1) # 1*1*C
        xyxy_null  = self.null_position_feature.view(1,1,-1) # 1*1*C

        # replace padding with learnable null embedding 
        text_embeddings  = text_embeddings*text_masks  + (1-text_masks)*text_null
        image_embeddings = image_embeddings*image_masks + (1-image_masks)*image_null
        xyxy_embedding = xyxy_embedding*masks + (1-masks)*xyxy_null

        objs_text  = self.linears_text(  torch.cat([text_embeddings, xyxy_embedding], dim=-1)  )
        objs_image = self.linears_image( torch.cat([image_embeddings,xyxy_embedding], dim=-1)  )
        objs = torch.cat( [objs_text,objs_image], dim=1 )

        assert objs.shape == torch.Size([B,N*2,self.out_dim])        
        return objs



