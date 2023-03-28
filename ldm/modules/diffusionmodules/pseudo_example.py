"""
This is a high-level pseudo code for grounding net. 

This class needs to tokenize grounding input into gronding tokens which 
will be used in GatedAttenion layers. 


class PositionNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        kwargs should be defined by model.grounding_tokenizer in config yaml file.  

    def forward(self, **kwargs):

        kwargs should be the output of grounding_tokenizer_input network
        
        return grounding_tokens # with shape: Batch * Num_Of_Token* Token_Channel_Dimension



"""


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 


"""
This is a high-level pseudo code for downsampler. 

This class needs to process input and output a spatial feature such that it will be 
fed into the first conv layer. 


class GroundingDownsampler(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        kwargs should be defined by model.grounding_downsampler in config yaml file.  

        you MUST define self.out_dim such that Unet knows add how many extra layers


    def forward(self, **kwargs):

        kwargs should be the output of grounding_downsampler_input network
        
        return spatial_feature # with shape: Batch * self.out_dim * H *W (64*64 for SD)



"""