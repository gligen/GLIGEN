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

        boxes = batch['boxes']
        masks = batch['masks'] 
        text_masks = batch['text_masks']
        image_masks = batch['image_masks'] 
        text_embeddings = batch["text_embeddings"] 
        image_embeddings = batch["image_embeddings"]

        self.batch, self.max_box, self.in_dim = text_embeddings.shape
        self.device = text_embeddings.device
        self.dtype = text_embeddings.dtype

        return {"boxes":boxes, 
                "masks":masks, 
                "text_masks":text_masks,
                "image_masks":image_masks,
                "text_embeddings":text_embeddings,
                "image_embeddings":image_embeddings,
                }


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
        text_masks = th.zeros(batch, self.max_box).type(dtype).to(device) 
        image_masks = th.zeros(batch, self.max_box).type(dtype).to(device) 
        text_embeddings =  th.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device) 
        image_embeddings = th.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device) 

        return {"boxes":boxes, 
                "masks":masks, 
                "text_masks":text_masks,
                "image_masks":image_masks,
                "text_embeddings":text_embeddings,
                "image_embeddings":image_embeddings,
                }






