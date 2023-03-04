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

        boxes=batch['boxes'] 
        masks=batch['masks']
        positive_embeddings=batch["text_embeddings"] 

        self.batch, self.max_box, self.in_dim = positive_embeddings.shape
        self.device = positive_embeddings.device
        self.dtype = positive_embeddings.dtype

        return {"boxes":boxes, "masks":masks, "positive_embeddings":positive_embeddings}


    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch =  self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        boxes = th.zeros(batch, self.max_box, 4,).type(dtype).to(device) 
        masks = th.zeros(batch, self.max_box).type(dtype).to(device) 
        positive_embeddings = th.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device) 

        return {"boxes":boxes, "masks":masks, "positive_embeddings":positive_embeddings}







