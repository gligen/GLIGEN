"""
Please write your own class for processing batch from dataloader. On the high-level,
this class needs to handle data needed by Grounding Tokenizer. 

This class needs to have 2 functions: prepare() and get_null_input() and one propery 'set'

prepare() takes in batch from dataloader and output input args for Grounding Tokenizer
get_null_input() will output null input args for Grounding Tokenizer. 


get_null_input() usually requires additional information from prepare() such as certain feature dimension,
thus typecially get_null_input() should be called at least prepare() was called once before, 
thus 'set' is used to tell if prepare() has been called or not 




class GroundingNetInput:
    
    def __init__(self):
        self.set = False 

    def prepare(self, batch):

        self.set = True 

        your code here .... 

        return {}


    def get_null_input(self, batch=None, device=None, dtype=None):

        assert self.set

        your code here ...

        return {}


"""