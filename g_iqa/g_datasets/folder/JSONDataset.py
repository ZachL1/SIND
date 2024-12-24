import torch.utils.data as data
from PIL import Image
import os
    

class JSONFolder(data.Dataset):
    def __init__(self, root, data_json, transforms):
        self.transform = transforms

        self.samples = []
        for item in data_json:
            self.samples.append(dict(
                path = os.path.join(root, item['image']),
                target = item['score'],
                scene = item['domain_id'],
                img_name = item['image'],
            ))

    def __getitem__(self, index):
        sample = self.samples[index]
        path = sample['path']
        target = sample['target']
        scene = sample['scene']
        img_name = sample['img_name']
        
        img = pil_loader(path)
        img_pt = img
        if self.transform is not None:
            img = self.transform[0](img)
            img_pt = self.transform[1](img) if len(self.transform) > 1 else img

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


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')