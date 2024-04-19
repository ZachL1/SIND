import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import pandas as pd
    

class PIQ23Folder(data.Dataset):
    def __init__(self, root, data_json, transform):
        self.transform = transform

        self.samples = []
        for item in data_json:
            self.samples.append(dict(
                path = os.path.join(root, item['image']),
                target = item['score'],
                scene = item['domain_id'],
                img_name = item['image'],
            ))

    '''
    # old version - verified
    def __init__(self, root, index:list, transform, scene_base=0):
        data_dir = root
        all_set = os.path.join(data_dir, 'Scores_Overall.csv')
        all_df = pd.read_csv(all_set)
        imgs_name = all_df['IMAGE PATH'].tolist()
        imgpath = [os.path.join(data_dir, img_name.replace('\\', '/')) for img_name in imgs_name]
        labels = all_df['JOD'].tolist()
        
        # domain label to unique int
        scene = [scene_base+int(s.split("_")[-1]) for s in all_df['SCENE'].tolist()]
        
        # check domain category
        print('domain category:', set(scene[i] for i in index))

        sample = []

        for i, item in enumerate(index):
            sample.append(dict(
                path = imgpath[item],
                target = labels[item],
                scene = scene[item],
                img_name = imgs_name[item],
            ))
        self.samples = sample
        self.transform = transform

        # # fit all scene target to base_scene
        # base_scene = self.samples[0]['scene']
        # base_scene_target = [sample['target'] for sample in self.samples if sample['scene'] == base_scene]
        # # compute a, b for each scene
        # scene_ab_dict = {}
        # for s in set(scene):
        #     scene_target = [sample['target'] for sample in self.samples if sample['scene'] == s]
        #     a, b = np.polyfit(scene_target, base_scene_target, 1)
        #     scene_ab_dict[s] = (a, b)
        # # ajust target
        # for sample in self.samples:
        #     a, b = scene_ab_dict[sample['scene']]
        #     sample['target'] = a * sample['target'] + b

        # # normalize target for each scene
        # if is_training:
        #     scene_std_mean_dict = {}
        #     for s in set(scene):
        #         scene_target = [sample['target'] for sample in self.samples if sample['scene'] == s]
        #         scene_std_mean_dict[s] = (np.std(scene_target), np.mean(scene_target))
        #     for sample in self.samples:
        #         std, mean = scene_std_mean_dict[sample['scene']]
        #         sample['target'] = (sample['target'] - mean) / std
    '''


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.samples[index]
        path = sample['path']
        target = sample['target']
        scene = sample['scene']
        img_name = sample['img_name']
        
        img = pil_loader(path)
        img_pt = img
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

    


class TID2013Folder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath,'.bmp.BMP')
        txtpath = os.path.join(root, 'mos_with_names.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')