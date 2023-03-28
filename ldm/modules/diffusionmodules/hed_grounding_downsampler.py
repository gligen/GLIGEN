import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F



class GroundingDownsampler(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        # No learnable params for hed edge map, just downsample it with bicubic

    def forward(self, grounding_extra_input):
        # this is actually gary scale, but converted to rgb in dataset, information redudant 
        grounding_extra_input = grounding_extra_input[:,0].unsqueeze(1)

        out = torch.nn.functional.interpolate(grounding_extra_input, (64,64), mode='bicubic')
        assert out.shape[1] == self.out_dim 
        return out


