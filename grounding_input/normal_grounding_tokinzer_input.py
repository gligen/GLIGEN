import os 
import torch as th 



class GroundingNetInput:
    def __init__(self):
        self.set = False 

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        input only for the ground tokenizer. 
        """

        self.set = True

        normal=batch['normal'] 
        mask=batch['mask']

        self.batch, self.C, self.H, self.W = normal.shape
        self.device = normal.device
        self.dtype = normal.dtype

        return {"normal":normal, "mask":mask}


    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch =  self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        normal = th.zeros(self.batch, self.C, self.H, self.W).type(dtype).to(device) 
        mask = th.zeros(batch).type(dtype).to(device) 

        return {"normal":normal, "mask":mask}







