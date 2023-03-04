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
        
        points = batch['points'] 
        masks = batch['masks']

        self.batch, self.max_persons_per_image, _ = points.shape
        self.max_persons_per_image = int(self.max_persons_per_image / 17) 
        self.device = points.device
        self.dtype = points.dtype

        return {"points":points, "masks":masks}


    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch =  self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        points = th.zeros(batch, self.max_persons_per_image*17, 2).to(device) 
        masks = th.zeros(batch, self.max_persons_per_image*17).to(device) 

        return {"points":points, "masks":masks}







