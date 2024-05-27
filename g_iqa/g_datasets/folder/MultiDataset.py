from torch.utils.data import Dataset, ConcatDataset

from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder

from g_iqa.g_datasets.folder.SPAQ import SPAQFolder
from g_iqa.g_datasets.folder.LIVEChallenge import LIVEChallengeFolder
from g_iqa.g_datasets.folder.Koniq_10k import Koniq_10kFolder
from g_iqa.g_datasets.folder.BID import BIDFolder

from g_iqa.g_datasets.folder.Kadid_10k import Kadid_10kFolder
from g_iqa.g_datasets.folder.LIVE import LIVEFolder
from g_iqa.g_datasets.folder.CSIQ import CSIQFolder

from g_iqa.g_datasets.folder.PARA import PARAFolder
from g_iqa.g_datasets.folder.EVA import EVAFolder

class MultiDatasetFolder(Dataset):
    def __init__(self, root, data_jsons, transforms, dataset_names):
        self.datasets = []
        for name in dataset_names:
            data_json = data_jsons[name]
            if name == 'live':
                dataset = LIVEFolder(
                    root=root, data_json=data_json, transform=transforms)
            elif name == 'livec':
                dataset = LIVEChallengeFolder(
                    root=root, data_json=data_json, transform=transforms)
            elif name == 'csiq':
                dataset = CSIQFolder(
                    root=root, data_json=data_json, transform=transforms)
            elif name == 'kadid10k':
                dataset = Kadid_10kFolder(
                    root=root, data_json=data_json, transform=transforms)
            elif name == 'koniq10k':
                dataset = Koniq_10kFolder(
                    root=root, data_json=data_json, transform=transforms)
            elif name == 'piq23':
                dataset = PIQ23Folder(
                    root=root, data_json=data_json, transform=transforms)
            elif name == 'spaq':
                dataset = SPAQFolder(
                    root=root, data_json=data_json, transform=transforms)
            elif name == 'para':
                dataset = PARAFolder(
                    root=root, data_json=data_json, transform=transforms)
            elif name == 'eva':
                dataset = EVAFolder(
                    root=root, data_json=data_json, transform=transforms)
            else:
                raise NotImplementedError(f"Not support dataset: {name}")
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