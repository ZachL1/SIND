import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))


def main():
    # config
    patch_size = 224
    test_patch_num = 1
    clip_model = 'openai/ViT-B-16'
    device = 'cuda:0'
    model_paths = [ # we use ensemble to get the final result
        './weights/finetune35_2_from_70_34_833.pth',
        './weights/finetune35_30_from_mask50_34_82.pth',
    ]

    # test data and model
    test_data = DataGenerator(data_dir='.', patch_size=patch_size, patch_num=test_patch_num, batch_size=1, istrain=False).get_data_loader()
    model = LocalGlobalClipIQA(clip_model=clip_model, clip_freeze=False, precision='fp32')
    model.to(device)

    preds = []
    for model_path in model_paths:
        pred = inference(model, test_data, model_path, device)
        preds.append(pred)

    # average ensemble result
    pred_scores = np.array(preds).mean(axis=0).tolist()

    # Load the images.csv file
    images_df = pd.read_csv(r'./images.csv')

    # Add a 'SCORE' column with a default value of 0.0
    images_df['SCORE'] = pred_scores

    # Save the modified DataFrame to result_23_PIQ.csv in the results folder
    images_df.to_csv('./results/result_23_PIQ.csv', index=False, sep=',')

    print('Done!')

def inference(model, test_data, model_path, device):
    # Load the model
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state)

    model.eval()

    # Get the predictions
    pred_scores = []
    for batch in tqdm(test_data):
        img = batch['img'].to(device)
        img_pt = batch['img_pt'].to(device)

        # we use T = 9 for testing argumentation
        if len(img.shape) == 5:
            B, T, C, H, W = img.shape
            img = img.view(B*T, C, H, W)
            img_pt = img_pt.view(B*T, C, H, W)

        with torch.no_grad():
            pred = model(img, img_pt)
        
        # reduce the T dimension
        if pred.size(0) != B:
            pred = pred.view(B, T)
            pred = torch.mean(pred, dim=1, keepdim=True)
            
        pred_scores.append(pred.item())

    return pred_scores


#########################################################################
################################# Model #################################
import open_clip

def load_clip_model(clip_model="openai/ViT-B-16", clip_freeze=True, precision='fp16'):
    pretrained, model_tag = clip_model.split('/')
    clip_model = open_clip.create_model(model_tag, precision=precision, pretrained=None, force_quick_gelu=True)
    if clip_freeze:
        for param in clip_model.parameters():
            param.requires_grad = False

    if model_tag == 'ViT-B-16':
        feature_size = dict(global_feature=512, local_feature=[196, 768])
    elif model_tag == 'ViT-L-14-quickgelu' or model_tag == 'ViT-L-14':
        feature_size = dict(global_feature=768, local_feature=[256, 1024])
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}")

    return clip_model, feature_size


class QualityFusionHead(nn.Module):
    def __init__(self, global_feature_size, local_feature_size, output_size=1):
        super(QualityFusionHead, self).__init__()
        self.global_feature_size = global_feature_size
        self.local_feature_size = local_feature_size
        self.output_size = output_size

        crop_patch = 7 * 7

        self.local_proj = nn.Sequential(
            nn.Conv1d(local_feature_size[0], crop_patch, 1),
            nn.ReLU(),
            nn.Conv1d(crop_patch, 1, 1),
            nn.ReLU(),
            nn.Linear(local_feature_size[1], global_feature_size),
            nn.ReLU(),
        )

        self.quality_predictor = nn.Sequential(
            # nn.TransformerEncoderLayer(d_model=1024, nhead=8),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, output_size),
            # nn.Linear(1024, output_size),
            nn.Linear(global_feature_size*2, output_size),
        )
        # print('[QualityFusionHead] TransformerEncoderLayer')

    def forward(self, global_features, local_features):
        local_features = self.local_proj(local_features).squeeze(1)

        features = torch.cat([global_features, local_features], dim=1)
        quality = self.quality_predictor(features)

        return quality


class LocalGlobalClipIQA(nn.Module):
    def __init__(self, clip_model="openai/ViT-B-16", clip_freeze=True, precision='fp16'):
        super(LocalGlobalClipIQA, self).__init__()
        self.clip_freeze = clip_freeze

        self.clip_model, feature_size = load_clip_model(clip_model, clip_freeze, precision)
        self.head = QualityFusionHead(feature_size['global_feature'], feature_size['local_feature'])

    def forward(self, x_global, x_local):
        global_features, _ = self.clip_model.encode_image(x_global) # B, 512
        _, local_features = self.clip_model.encode_image(x_local) # B, 196, 768

        quality = self.head(global_features, local_features)

        return quality


#########################################################################
################################## Data #################################
from torch.utils.data import Sampler, DataLoader, Dataset
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

class DataGenerator(object):
    def __init__(self, data_dir, patch_size, patch_num, batch_size=1, istrain=True, scene_sampling=0):

        self.batch_size = batch_size
        self.istrain = istrain
        self.scene_sampling = scene_sampling

        if istrain:
            transforms = [
                torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize(size=(patch_size+20, patch_size+20), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))
                ]), # whole image for global coarser feature
                torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    # torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.Resize(size=patch_size*3, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.ToTensor(),
                    RandomCropMiniPatch(size=(patch_size, patch_size), center=False),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))
                ]) # multi-mini-patch for local finer feature
            ]
        else:
            transforms = [
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(patch_size+20, patch_size+20), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.FiveCrop(size=patch_size),
                    NineCrop(size=(patch_size, patch_size)),
                    torchvision.transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                ]),
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=patch_size*3, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.ToTensor(),
                    # FiveCropMiniPatch(size=(patch_size, patch_size)),
                    NineCropMiniPatch(size=(patch_size, patch_size)),
                    torchvision.transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                    std=(0.229, 0.224, 0.225))
                ])
            ]

        self.data = PIQ23Folder(
                root=data_dir, transform=transforms, patch_num=patch_num)

    def get_data_loader(self):
        if self.scene_sampling > 0:
            # scene_sampler = SceneSampler(self.data, self.batch_size, self.scene_sampling)
            shuffle = False
        else:
            scene_sampler = None
            shuffle = self.istrain

        if self.istrain:
            dataloader = DataLoader(
                self.data, batch_size=self.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, sampler=scene_sampler, drop_last=True)
        else:
            dataloader = DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return dataloader

class PIQ23Folder(Dataset):

    def __init__(self, root, transform, patch_num, scene_base=0):
        data_dir = root
        all_set = os.path.join(data_dir, 'images.csv')
        all_df = pd.read_csv(all_set)
        imgs_name = all_df['IMAGE'].tolist()
        imgpath = [os.path.join(data_dir, img_name.replace('\\', '/')) for img_name in imgs_name]
        labels = all_df['JOD'].tolist()
        scene = [scene_base+int(s) for s in all_df['CLASS'].tolist()]

        sample = []

        for item in range(len(imgpath)):
            for i in range(patch_num):
                sample.append(dict(
                    path = imgpath[item],
                    target = labels[item],
                    scene = scene[item],
                    img_name = imgs_name[item],
                ))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        sample = self.samples[index]
        path = sample['path']
        target = sample['target']
        scene = sample['scene']
        img_name = sample['img_name']
        
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img, img_pt = (tf(img) for tf in self.transform)

        return dict(
            img=img,
            img_pt=img_pt,
            label=target,
            scene=scene,
            img_name=img_name
        )

    def __len__(self):
        length = len(self.samples)
        return length
    
    def get_scene_list(self):
        return [sample['scene'] for sample in self.samples]

class RandomCropMiniPatch(object):
    """Crop the given tensor multi-mini-patch and cat them together."""
    def __init__(self, size, patch_num=7, center=False):
        self.size = size # output size = (size, size)
        self.patch_num = patch_num # number of patches = patch_num * patch_num
        self.center = center

    def __call__(self, img:torch.Tensor):
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"img should be a Tensor. Got {type(img)}")
        
        c, h, w = img.size()
        th, tw = self.size
        if w == tw and h == th:
            return img

        assert th % self.patch_num == 0 and tw % self.patch_num == 0, "output size should be divided by patch_num"

        patch_szw = tw // self.patch_num
        patch_szh = th // self.patch_num
        scale_w = w // self.patch_num
        scale_h = h // self.patch_num

        assert scale_h > patch_szh and scale_w > patch_szh, "img can not crop to mini patch"

        if self.center:
            rd_ps_h = [(scale_h - patch_szh) // 2] * self.patch_num
            rd_ps_w = [(scale_w - patch_szw) // 2] * self.patch_num
        else:
            rd_ps_h = torch.randint(scale_h-patch_szh, (self.patch_num,))
            rd_ps_w = torch.randint(scale_w-patch_szh, (self.patch_num,))
        
        mask = torch.zeros((h, w)).bool()
        for i in range(self.patch_num):
            for j in range(self.patch_num):
                mask[i*scale_h+rd_ps_h[i]:i*scale_h+rd_ps_h[i]+patch_szh, j*scale_w+rd_ps_w[j]:j*scale_w+rd_ps_w[j]+patch_szw] = True

        recat_patchs = img[:, mask].view(c, th, tw)
                
        return recat_patchs
        
class FiveCropMiniPatch(object):
    """Crop the given tensor multi-mini-patch and cat them together."""
    def __init__(self, size, patch_num=7):
        self.size = size # output size = (size, size)
        self.patch_num = patch_num # number of patches = patch_num * patch_num

    def __call__(self, img:torch.Tensor):
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"img should be a Tensor. Got {type(img)}")
        
        c, h, w = img.size()
        th, tw = self.size
        if w == tw and h == th:
            return img

        assert th % self.patch_num == 0 and tw % self.patch_num == 0, "output size should be divided by patch_num"

        patch_szw = tw // self.patch_num
        patch_szh = th // self.patch_num
        scale_w = w // self.patch_num
        scale_h = h // self.patch_num

        assert scale_h > patch_szh and scale_w > patch_szh, "img can not crop to mini patch"

        positions = [(0, 0), (0, scale_w - patch_szw), (scale_h - patch_szh, 0), (scale_h - patch_szh, scale_w - patch_szw), ((scale_h - patch_szh) // 2, (scale_w - patch_szw) // 2)]
        
        recat_patchs = []
        for pos_h, pos_w in positions:
            mask = torch.zeros((h, w)).bool()
            for i in range(self.patch_num):
                for j in range(self.patch_num):
                    mask[i*scale_h+pos_h:i*scale_h+pos_h+patch_szh, j*scale_w+pos_w:j*scale_w+pos_w+patch_szw] = True
            recat_patchs.append(img[:, mask].view(c, th, tw))
        
        return recat_patchs

class NineCropMiniPatch(object):
    """Crop the given tensor multi-mini-patch and cat them together."""
    def __init__(self, size, patch_num=7):
        self.size = size # output size = (size, size)
        self.patch_num = patch_num # number of patches = patch_num * patch_num

    def __call__(self, img:torch.Tensor):
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"img should be a Tensor. Got {type(img)}")
        
        c, h, w = img.size()
        th, tw = self.size
        if w == tw and h == th:
            return img

        assert th % self.patch_num == 0 and tw % self.patch_num == 0, "output size should be divided by patch_num"

        patch_szw = tw // self.patch_num
        patch_szh = th // self.patch_num
        scale_w = w // self.patch_num
        scale_h = h // self.patch_num

        assert scale_h > patch_szh and scale_w > patch_szh, "img can not crop to mini patch"

        positions = [(0, 0), (0, scale_w - patch_szw), (scale_h - patch_szh, 0), (scale_h - patch_szh, scale_w - patch_szw), ((scale_h - patch_szh) // 2, (scale_w - patch_szw) // 2), ((scale_h - patch_szh) // 2, 0), ((scale_h - patch_szh) // 2, scale_w - patch_szw), (0, (scale_w - patch_szw) // 2), (scale_h - patch_szh, (scale_w - patch_szw) // 2)]
        
        recat_patchs = []
        for pos_h, pos_w in positions:
            mask = torch.zeros((h, w)).bool()
            for i in range(self.patch_num):
                for j in range(self.patch_num):
                    mask[i*scale_h+pos_h:i*scale_h+pos_h+patch_szh, j*scale_w+pos_w:j*scale_w+pos_w+patch_szw] = True
            recat_patchs.append(img[:, mask].view(c, th, tw))
        
        return recat_patchs

class NineCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            c, height, width = img.size()
        elif isinstance(img, Image.Image):
            width, height = img.size
        w, h = self.size
        
        if width < w or height < h:
            raise ValueError("Image is smaller than crop size")

        left = 0
        top = 0
        center_x = (width - w) // 2
        center_y = (height - h) // 2
        right = width - w
        bottom = height - h

        crops = []
        crops.append(F.crop(img, top, left, h, w))
        crops.append(F.crop(img, top, center_x, h, w))
        crops.append(F.crop(img, top, right, h, w))
        crops.append(F.crop(img, center_y, left, h, w))
        crops.append(F.crop(img, center_y, center_x, h, w))
        crops.append(F.crop(img, center_y, right, h, w))
        crops.append(F.crop(img, bottom, left, h, w))
        crops.append(F.crop(img, bottom, center_x, h, w))
        crops.append(F.crop(img, bottom, right, h, w))

        return crops





if __name__ == "__main__":
    main()