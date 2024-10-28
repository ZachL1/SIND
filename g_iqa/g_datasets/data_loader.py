import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.functional as F

from prefetch_generator import BackgroundGenerator


from g_iqa.g_datasets.folder import JSONFolder, MultiDatasetFolder
from g_iqa.g_datasets.g_transforms import RandomCropMiniPatch, NineCrop, NineCropMiniPatch, SceneSampler, ConditionalResize

# imagenet mean and std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# # OpenAI CLIP mean and std
# MEAN = [0.48145466, 0.4578275, 0.40821073]
# STD = [0.26862954, 0.26130258, 0.27577711]

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def stack_crops(crops):
    return torch.stack([crop for crop in crops])

class DataGenerator(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, data_json, input_size, batch_size=1, istrain=True, scene_sampling=0, testing_aug=False):

        self.batch_size = batch_size
        self.istrain = istrain
        self.scene_sampling = scene_sampling

        global_resize = torchvision.transforms.Resize(
            size=(input_size+32, input_size+32), # 256 x 256
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True
        )
        local_resize = torchvision.transforms.Resize(
            size=input_size*3, # 672 x *
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True
        )
        # To some extent in raw resolution. Actually, it produces similar results to using fixed resolution (672x*).
        # local_resize = ConditionalResize(
        #     size=input_size+32, # resize to 256 if shorter side is less than 256
        #     max_short_size=input_size*3, # resize to 1120 if shorter side is greater than 1120
        #     interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        #     antialias=True
        # )

        if istrain:
            transforms = [
                torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    global_resize,
                    torchvision.transforms.RandomCrop(size=input_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=MEAN,
                                                        std=STD)
                ]), # whole image for global coarser feature
                torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    local_resize,
                    torchvision.transforms.ToTensor(),
                    RandomCropMiniPatch(size=(input_size, input_size), center=False),
                    torchvision.transforms.Normalize(mean=MEAN,
                                                        std=STD)
                ]) # multi-mini-patch for local finer feature
            ]
        elif testing_aug:
            transforms = [
                torchvision.transforms.Compose([
                    global_resize,
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.FiveCrop(size=input_size),
                    NineCrop(size=(input_size, input_size)),
                    torchvision.transforms.Lambda(stack_crops),
                    torchvision.transforms.Normalize(mean=MEAN,
                                                    std=STD)
                ]),
                torchvision.transforms.Compose([
                    local_resize,
                    torchvision.transforms.ToTensor(),
                    # FiveCropMiniPatch(size=(input_size, input_size)),
                    NineCropMiniPatch(size=(input_size, input_size)),
                    torchvision.transforms.Lambda(stack_crops),
                    torchvision.transforms.Normalize(mean=MEAN,
                                                    std=STD)
                ])
            ]
        else:
            transforms = [
                torchvision.transforms.Compose([
                    global_resize,
                    torchvision.transforms.CenterCrop(size=input_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=MEAN,
                                                    std=STD)
                ]),
                torchvision.transforms.Compose([
                    local_resize,
                    torchvision.transforms.ToTensor(),
                    RandomCropMiniPatch(size=(input_size, input_size), center=True),
                    torchvision.transforms.Normalize(mean=MEAN,
                                                    std=STD)
                ])
            ]

        if isinstance(dataset, list) and len(dataset) == 1:
            dataset = dataset[0]
        if isinstance(dataset, str):
            data_json = data_json[dataset] if isinstance(data_json, dict) else data_json
            
        if isinstance(dataset, list) and len(dataset) > 1:
            self.data = MultiDatasetFolder(
                root=path, data_jsons=data_json, transforms=transforms, dataset_names=dataset)
        elif dataset in ['live', 'livec', 'csiq', 'kadid10k', 'agiqa3k', 'koniq10k', 'bid', 'cid2013', 'tid2013', 'piq23', 'spaq', 'para', 'eva', 'ava', 'common_json']:
            self.data = JSONFolder(
                root=path, data_json=data_json, transforms=transforms)
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
                self.data, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        return dataloader