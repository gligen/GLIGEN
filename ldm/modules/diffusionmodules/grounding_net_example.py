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