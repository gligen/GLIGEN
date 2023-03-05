import os 

class DatasetCatalog:
    def __init__(self, ROOT, which_embedder):
        assert which_embedder in ['clip', 'bert']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.VGGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params": dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/gqa/tsv/train-00.tsv'),
            )
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.FlickrGrounding = {
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/flickr30k/tsv/train-00.tsv'),
            )
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.SBUGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/SBU/tsv/train-00.tsv'),
            )
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.CC3MGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv/train-00.tsv'),
            )
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.CC12MGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/CC12M/tsv/train-00.tsv'),
            )
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        # temp = 'category_embedding_clip.pth' if which_embedder == 'clip' else 'category_embedding_bert.pth' 
        # obj365_category_embedding_path = os.path.join(ROOT, 'OBJECTS365', temp)

        self.Obj365Detection = {   
        "target": "dataset.tsv_dataset.TSVDataset",
        "train_params":dict(
            tsv_path=os.path.join(ROOT,'OBJECTS365/tsv/train-00.tsv'),
            ),
        }


