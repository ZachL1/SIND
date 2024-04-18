import os
import json
import pandas as pd

from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder

'''
import os
import numpy as np
import pandas as pd
import json

base_dir = '/home/dzc/workspace/G-IQA/data/EVA'

all_files = []
with open(f'{base_dir}/eva-dataset-master/data/image_content_category.csv', 'r') as f:
    lines = f.readlines()

info = pd.read_csv(f'{base_dir}/eva-dataset-master/data/votes_filtered.csv', sep='=', dtype={'image_id': str})
for line in lines[1:]:
    id, cat = line.strip().split(',')
    id, cat = id.strip('"'), cat.strip('"')
    score = info['score'][info['image_id']==id].to_numpy()
    if len(score) == 0:
        continue
    all_files.append({
        'image': f'eva-dataset-master/images/EVA_category/EVA_category/{cat}/{id}.jpg',
        'category': cat,
        'score': np.mean(score)
    })
    assert os.path.exists(os.path.join(base_dir, all_files[-1]['image']))

with open(f'{base_dir}/annotations/EVA_all.json', 'w') as f:
    json.dump({'files': all_files}, f)
'''


class EVAFolder(PIQ23Folder):
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