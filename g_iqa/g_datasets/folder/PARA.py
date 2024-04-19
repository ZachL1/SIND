import os
import pandas as pd

from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder

class PARAFolder(PIQ23Folder):
    def __init__(self, root, data_json, transform):
        super().__init__(root, data_json, transform)
    '''
    # old version
    def __init__(self, root, index=None, transform=None, scene_base=1000, iaa=True):
        data_dir = root
        all_set = os.path.join(data_dir, 'annotation/PARA-GiaaAll.csv')
        all_df = pd.read_csv(all_set)
        imgs_name = all_df['imageName'].tolist()
        sessions = all_df['sessionId'].tolist()
        imgpath = [os.path.join(data_dir, f'imgs/{session}/{img_name}') for session, img_name in zip(sessions, imgs_name)]
        labels = all_df['aestheticScore_mean'].tolist() if iaa else all_df['qualityScore_mean'].tolist()
        
        # domain label to unique int
        all_scene = sorted(all_df['semantic'].unique())
        scene_dict = {scene: i for i, scene in enumerate(all_scene)}
        scene = [scene_dict[s] + scene_base for s in all_df['semantic'].tolist()]

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