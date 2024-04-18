from torch.utils.data import Dataset, ConcatDataset

from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder
from g_iqa.g_datasets.folder.SPAQ import SPAQFolder
from g_iqa.g_datasets.folder.LIVEChallenge import LIVEChallengeFolder
from g_iqa.g_datasets.folder.Koniq_10k import Koniq_10kFolder
from g_iqa.g_datasets.folder.BID import BIDFolder


class MultiDatasetFolder(Dataset):
    def __init__(self, root, index, transform):
        self.dataset = []
        for r in root:
            if r.endswith('PIQ23/'):
                self.dataset.append(PIQ23Folder(r, index, transform))
                # self.dataset.extend([PIQ23Folder(r, index, transform) for _ in range(len(r))])
            elif r.endswith('SPAQ/'):
                self.dataset.append(SPAQFolder(r, None, transform))
            elif r.endswith('LIVEW/'):
                self.dataset.append(LIVEChallengeFolder(r, None, transform))
            elif r.endswith('koniq10k/'):
                self.dataset.append(Koniq_10kFolder(r, None, transform))
            elif r.endswith('BID/'):
                self.dataset.append(BIDFolder(r, None, transform))
            else:
                raise ValueError(f"Unknown dataset: {r}")
            
        self.dataset = ConcatDataset(self.dataset)
        self.scene = []
        for d in self.dataset.datasets:
            self.scene += d.get_scene_list()
    
    def __getitem__(self, index):
        # try:
        return self.dataset.__getitem__(index)
        # except Exception as e:
        #     print(f'Error: {e}')
        #     return self.dataset.__getitem__(0)
    
    def __len__(self):
        return len(self.dataset)
    
    def get_scene_list(self):
        return self.scene