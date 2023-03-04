import os 

class DatasetCatalog:
    def __init__(self, ROOT):


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.VGGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params": dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/gqa/tsv/train-00.tsv'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.FlickrGrounding = {
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/flickr30k/tsv/train-00.tsv'),
            ),
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.SBUGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/SBU/tsv/train-00.tsv'),
            ),
         }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.CC3MGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv/train-00.tsv'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 


        self.CC12MGrounding = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'GROUNDING/CC12M/tsv/train-00.tsv'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.Obj365Detection = {   
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'OBJECTS365/tsv/train-00.tsv'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.COCO2017Keypoint = {   
            "target": "dataset.dataset_kp.KeypointDataset",
            "train_params":dict(
                image_root = os.path.join(ROOT,'COCO/images'),
                keypoints_json_path = os.path.join(ROOT,'COCO/annotations2017/person_keypoints_train2017.json'),
                caption_json_path = os.path.join(ROOT,'COCO/annotations2017/captions_train2017.json'),
            ),
        }


