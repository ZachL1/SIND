from PIL import Image
import json
import os
import torch
from torch.utils import data as data

from pyiqa.utils.registry import DATASET_REGISTRY
from .base_iqa_dataset import BaseIQADataset

@DATASET_REGISTRY.register()
class GeneralJSONDataset(BaseIQADataset):
    """General No Reference dataset with meta info file.
    """
    def init_path_mos(self, opt):
        target_img_folder = opt['dataroot_target']
        with open(opt['meta_info_file'], 'r') as f:
            self.paths_mos = json.load(f)['files']
        for path_mos in self.paths_mos:
            path_mos['image'] = os.path.join(target_img_folder, path_mos['image'])

    def __getitem__(self, index):

        img_path = self.paths_mos[index]['image']
        mos_label = float(self.paths_mos[index]['score'])
        img_pil = Image.open(img_path).convert('RGB')

        # # keep ratio and resize shorter edge to 1024
        # w, h = img_pil.size
        # if min(w, h) > 1024:
        #     if w > h:
        #         ow = 1024
        #         oh = int(1024 * h / w)
        #     else:
        #         oh = 1024
        #         ow = int(1024 * w / h)
        #     img_pil = img_pil.resize((ow, oh), Image.BICUBIC)

        img_tensor = self.trans(img_pil) * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])
                
        return {'img': img_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path}
