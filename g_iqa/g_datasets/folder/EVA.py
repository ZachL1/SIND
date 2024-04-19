import os
import json
import pandas as pd

from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder


class EVAFolder(PIQ23Folder):
    def __init__(self, root, data_json, transform):
        super().__init__(root, data_json, transform)
    '''
    # old version
    def __init__(self, root, index=None, transform=None, scene_base=1100):
        data_dir = root
        all_set = os.path.join(data_dir, 'annotations/EVA_all.json')
        with open(all_set, 'r') as f:
            all_js = json.load(f)['files']
        imgs_name = [item['image'] for item in all_js]
        imgpath = [os.path.join(data_dir, img_name) for img_name in imgs_name]
        labels = [item['score'] for item in all_js]
        
        # domain label to unique int
        scene = [int(item['category']) + scene_base for item in all_js]

        if index is None:
            index = list(range(len(imgs_name)))
        
        # check domain category
        print('domain category:', set(scene[i] for i in index))

        self.samples = []
        for i, item in enumerate(index):
            self.samples.append(dict(
                path = imgpath[item],
                target = labels[item],
                scene = scene[item],
                img_name = imgs_name[item],
            ))
        self.transform = transform
    '''