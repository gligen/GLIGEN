from .catalog import DatasetCatalog
from ldm.util import instantiate_from_config
import torch 




class ConCatDataset():
    def __init__(self, dataset_name_list, ROOT, which_embedder, train=True, repeats=None):
        self.datasets = [] 
        cul_previous_dataset_length = 0 
        offset_map = []
        which_dataset = []

        if repeats is None:
            repeats = [1] * len(dataset_name_list)
        else:
            assert len(repeats) == len(dataset_name_list)
            

        Catalog = DatasetCatalog(ROOT, which_embedder)
        for dataset_idx, (dataset_name, yaml_params) in enumerate(dataset_name_list.items()):
            repeat = repeats[dataset_idx]

            dataset_dict = getattr(Catalog, dataset_name)
            
            target = dataset_dict['target']
            params = dataset_dict['train_params'] if train else dataset_dict['val_params']
            if yaml_params is not None:
                params.update(yaml_params)
            dataset = instantiate_from_config( dict(target=target, params=params) )
            
            self.datasets.append(dataset)
            for _ in range(repeat):
                offset_map.append(  torch.ones(len(dataset))*cul_previous_dataset_length  )
                which_dataset.append(  torch.ones(len(dataset))*dataset_idx  )
                cul_previous_dataset_length += len(dataset)
        offset_map = torch.cat(offset_map, dim=0).long()
        self.total_length = cul_previous_dataset_length

        self.mapping = torch.arange(self.total_length) - offset_map
        self.which_dataset = torch.cat(which_dataset, dim=0).long()


    def total_images(self):
        count = 0
        for dataset in self.datasets:
            print(dataset.total_images())
            count += dataset.total_images()
        return count



    def __getitem__(self, idx):
        dataset = self.datasets[ self.which_dataset[idx] ]   
        return dataset[ self.mapping[idx] ]     


    def __len__(self):
        return self.total_length
            




