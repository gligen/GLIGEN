import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F



class PositionNet(nn.Module):
    def __init__(self,  max_persons_per_image, out_dim, fourier_freqs=8):
        super().__init__()
        self.max_persons_per_image = max_persons_per_image
        self.out_dim = out_dim

        self.person_embeddings   = torch.nn.Parameter(torch.zeros([max_persons_per_image,out_dim]))
        self.keypoint_embeddings = torch.nn.Parameter(torch.zeros([17,out_dim]))
         

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*2 # 2 is sin&cos, 2 is xy 

        self.linears = nn.Sequential(
            nn.Linear( self.out_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_person_feature = torch.nn.Parameter(torch.zeros([self.out_dim]))
        self.null_xy_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, points, masks):
        
        masks = masks.unsqueeze(-1)
        N = points.shape[0]

        person_embeddings = self.person_embeddings.unsqueeze(1).repeat(1,17,1).reshape(self.max_persons_per_image*17, self.out_dim)
        keypoint_embeddings = torch.cat([self.keypoint_embeddings]*self.max_persons_per_image, dim=0)
        person_embeddings = person_embeddings + keypoint_embeddings # (num_person*17) * C 
        person_embeddings = person_embeddings.unsqueeze(0).repeat(N,1,1)

        # embedding position (it may includes padding as placeholder)
        xy_embedding = self.fourier_embedder(points) # B*N*2 --> B*N*C

        
        # learnable null embedding 
        person_null = self.null_person_feature.view(1,1,-1)
        xy_null =  self.null_xy_feature.view(1,1,-1)

        # replace padding with learnable null embedding 
        person_embeddings = person_embeddings*masks + (1-masks)*person_null
        xy_embedding = xy_embedding*masks + (1-masks)*xy_null

        objs = self.linears(  torch.cat([person_embeddings, xy_embedding], dim=-1)  )
      
        return objs



