import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as F

from prefetch_generator import BackgroundGenerator


from g_iqa.g_datasets import folder
from g_iqa.g_datasets.g_transforms import RandomCropMiniPatch, NineCrop, NineCropMiniPatch, SceneSampler

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def stack_crops(crops):
    return torch.stack([crop for crop in crops])

class DataGenerator(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, data_json, input_size, batch_size=1, istrain=True, scene_sampling=0):

        self.batch_size = batch_size
        self.istrain = istrain
        self.scene_sampling = scene_sampling

        if istrain:
            transforms = [
                torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize(size=(input_size+20, input_size+20), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.RandomCrop(size=input_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))
                ]), # whole image for global coarser feature
                torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    # torchvision.transforms.RandomCrop(size=input_size),
                    torchvision.transforms.Resize(size=input_size*3, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.ToTensor(),
                    RandomCropMiniPatch(size=(input_size, input_size), center=False),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))
                ]) # multi-mini-patch for local finer feature
            ]
        else:
            transforms = [
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(input_size+20, input_size+20), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.FiveCrop(size=input_size),
                    NineCrop(size=(input_size, input_size)),
                    torchvision.transforms.Lambda(stack_crops),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                ]),
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=input_size*3, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.ToTensor(),
                    # FiveCropMiniPatch(size=(input_size, input_size)),
                    NineCropMiniPatch(size=(input_size, input_size)),
                    torchvision.transforms.Lambda(stack_crops),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                ])
            ]

        if isinstance(dataset, list) and len(dataset) == 1:
            dataset = dataset[0]
        if isinstance(dataset, str):
            data_json = data_json[dataset]
            
        if isinstance(dataset, list) and len(dataset) > 1:
            self.data = folder.MultiDatasetFolder(
                root=path, data_jsons=data_json, transforms=transforms, dataset_names=dataset)
            # raise NotImplementedError("MultiDatasetFolder is not implemented yet")
        elif dataset == 'live':
            self.data = folder.LIVEFolder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'livec':
            self.data = folder.LIVEChallengeFolder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'csiq':
            self.data = folder.CSIQFolder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'kadid10k':
            self.data = folder.Kadid_10kFolder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'agiqa3k':
            self.data = folder.AGIQA3KFolder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'koniq10k':
            self.data = folder.Koniq_10kFolder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'bid':
            self.data = folder.BIDFolder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'cid2013':
            self.data = folder.CID2013Folder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'tid2013':
            self.data = folder.TID2013Folder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'piq23':
            self.data = folder.PIQ23Folder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'spaq':
            self.data = folder.SPAQFolder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'para':
            self.data = folder.PARAFolder(
                root=path, data_json=data_json, transform=transforms)
        elif dataset == 'eva':
            self.data = folder.EVAFolder(
                root=path, data_json=data_json, transform=transforms)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def get_data(self):
        if self.scene_sampling > 0:
            scene_sampler = SceneSampler(self.data, self.batch_size, self.scene_sampling)
            shuffle = False
        else:
            scene_sampler = None
            shuffle = self.istrain

        if self.istrain:
            dataloader = DataLoaderX(
                self.data, batch_size=self.batch_size, shuffle=shuffle, num_workers=16, pin_memory=True, sampler=scene_sampler, drop_last=True)
        else:
            dataloader = DataLoaderX(
                self.data, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return dataloader