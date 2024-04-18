import os
import random
import pandas as pd

from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder

def extract_positive_indices(row):
    return [idx for idx, value in enumerate(row) if value > 0]

class SPAQFolder(PIQ23Folder):
    def __init__(self, root, index=None, transform=None, scene_base=100):
        # data_dir = os.path.join(root, 'SPAQ')
        data_dir = root
        all_set = os.path.join(data_dir, 'annotations/MOS and Image attribute scores.xlsx')
        all_df = pd.read_excel(all_set)
        imgs_name = all_df['Image name'].tolist()
        imgpath = [os.path.join(data_dir, f'TestImage/{img_name}') for img_name in imgs_name]
        labels = all_df['MOS'].tolist()
        
        # the category probability distribution of images, index is the image name
        scene_df = pd.read_excel(os.path.join(data_dir, 'annotations/Scene category labels.xlsx'))
        assert all_df['Image name'].tolist() == scene_df['Image name'].tolist()
        # 由于SPAQ中一个图片可能对应多个场景，因此这里提取所有场景类别作为candidate，每轮epoch从对应的candidate中随机挑选一个作为其场景标签
        # scene 标签只用于训练时区分不同场景（场景采样），对于测试而言没有任何意义
        scene_candidate = scene_df.drop(columns='Image name').apply(lambda row: extract_positive_indices(row), axis=1)
        scene = [scene_base + random.choice(scene_list) for scene_list in self.scene_candidate.values()]

        self.scene_candidate = {img_name: [scene_base + scene for scene in scene_list] for img_name, scene_list in zip(imgs_name, scene_candidate)}

        if index is None:
            index = list(range(len(imgs_name)))
            
        # check domain category
        print('domain category:', set(s for s in scene_candidate[i] for i in index))

        self.samples = []
        for i, item in enumerate(index):
            self.samples.append(dict(
                path = imgpath[item],
                target = labels[item],
                scene = scene[item],
                img_name = imgs_name[item],
            ))
        self.transform = transform
    
    def update_scene(self, scene_base):
        for sample in self.samples:
            sample['scene'] = scene_base + random.choice(self.scene_candidate[sample['img_name']])

    