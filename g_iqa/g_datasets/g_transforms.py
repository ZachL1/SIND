import torch
import torchvision.transforms.functional as F
from torch.utils.data import Sampler

import matplotlib.pyplot as plt
import PIL

import random
import math




class SceneSampler(Sampler[int]):
    '''
    random sample, sample scene within a batch are all the same
    '''
    def __init__(self, data_source, batch_size, scene_sampling=1):
        self.data_source = data_source
        self.batch_size = batch_size
        self.scene_sampling = scene_sampling
        self.scene_bs = self.batch_size // self.scene_sampling
        self.data_len = len(data_source)

        self.update()

    def __iter__(self):
        if self.data_len != len(self.data_source):
            self.update()
        # init seed
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        random.seed(seed)

        
        # pad each len(scene_index) to an integer multiple of batch_size
        scene_index_pad = {}
        for s in self.scene:
            pad = (self.scene_bs - len(self.scene_index[s]) % self.scene_bs) % self.scene_bs
            # random pad
            scene_index_pad[s] = self.scene_index[s] + [random.choice(self.scene_index[s]) for _ in range(pad)]
            # shuffle
            random.shuffle(scene_index_pad[s])

        # [[scene_batch], [scene_batch], ...]
        index_scene_batch = []
        for s in self.scene:
            scene_batch = [scene_index_pad[s][i:i+self.scene_bs] for i in range(0, len(scene_index_pad[s]), self.scene_bs)]
            # index_scene_batch.append(scene_batch)
            index_scene_batch.extend(scene_batch)

        # shuffle scene_batch
        random.shuffle(index_scene_batch)

        # yield index
        yield from [index for scene_batch in index_scene_batch for index in scene_batch]
             

    def __len__(self):
        if self.data_len != len(self.data_source):
            self.update()
        return self.num_samples
    
    def update(self):
        scene_list = self.data_source.get_scene_list()
        self.scene = set(scene_list)
        self.scene_index = {s: [i for i, scene in enumerate(scene_list) if scene == s] for s in self.scene}
        self.num_samples = sum([math.ceil(len(self.scene_index[s]) / self.scene_bs) * self.scene_bs for s in self.scene])


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

        # # log recat_patchs
        # # save torch.Tensor to .png
        # torchvision.utils.save_image(img, "img.png")
        # torchvision.utils.save_image(recat_patchs, "recat_patchs.png")

        # # visualize patchs in img
        # img = img.permute(1,2,0).numpy()
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(img)
        # for i in range(self.patch_num):
        #     for j in range(self.patch_num):
        #         # rect = plt.Rectangle((j*patch_szw, i*patch_szh), patch_szw, patch_szh, linewidth=1, edgecolor='r', facecolor='none')
        #         rect = plt.Rectangle((j*scale_w+rd_ps_w[j], i*scale_h+rd_ps_h[i]), patch_szw, patch_szh, linewidth=1, edgecolor='r', facecolor='none')
        #         # ij = i*self.patch_num+j
        #         # rect = plt.Rectangle((j*scale_w+rd_ps_w[ij], i*scale_h+rd_ps_h[ij]), patch_szw, patch_szh, linewidth=1, edgecolor='r', facecolor='none')
        #         ax.add_patch(rect)
        # plt.savefig("img_patchs.png")
                
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

            # # log recat_patchs
            # # save torch.Tensor to .png
            # torchvision.utils.save_image(img, "img.png")
            # torchvision.utils.save_image(recat_patchs[-1], "recat_patchs.png")

            # # visualize patchs in img
            # img_hwc = img.permute(1,2,0).numpy()
            # fig, ax = plt.subplots(1, 1)
            # ax.imshow(img_hwc)
            # for i in range(self.patch_num):
            #     for j in range(self.patch_num):
            #         rect = plt.Rectangle((j*scale_w+pos_w, i*scale_h+pos_h), patch_szw, patch_szh, linewidth=1, edgecolor='r', facecolor='none')
            #         ax.add_patch(rect)
            # plt.savefig("img_patchs.png")
        
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
        elif isinstance(img, PIL.Image.Image):
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
