from torch.utils.data import Dataset, ConcatDataset

from g_iqa.g_datasets.folder.JSONDataset import JSONFolder

class MultiDatasetFolder(Dataset):
    def __init__(self, root, data_jsons, transforms, dataset_names):
        self.datasets = []
        for name in dataset_names:
            data_json = data_jsons[name]
            dataset = JSONFolder(
                root=root, data_json=data_json, transforms=transforms)
            self.datasets.append(dataset)
            
        self.datasets = ConcatDataset(self.datasets)
        self.scene = []
        for d in self.datasets.datasets:
            self.scene += d.get_scene_list()
    
    def __getitem__(self, index):
        # try:
        return self.datasets.__getitem__(index)
        # except Exception as e:
        #     print(f'Error: {e}')
        #     return self.dataset.__getitem__(0)
    
    def __len__(self):
        return len(self.datasets)
    
    def get_scene_list(self):
        return self.scene