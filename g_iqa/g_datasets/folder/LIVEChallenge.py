import os
import numpy as np
import scipy

from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder


class LIVEChallengeFolder(PIQ23Folder):

    def __init__(self, root, index=None, transform=None, scene_base=200):

        # data_dir = os.path.join(root, 'LIVEW')
        data_dir = root
        imgpath = scipy.io.loadmat(os.path.join(data_dir, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        mos = scipy.io.loadmat(os.path.join(data_dir, 'Data', 'AllMOS_release.mat'))
        labels = mos['AllMOS_release'].astype(np.float32)
        labels = labels[0][7:1169]
        # scene = ['livew_0'] * len(imgpath)
        scene = [scene_base] * len(imgpath)

        if index is None:
            index = list(range(len(imgpath)))
        self.samples = []
        for i, item in enumerate(index):
            self.samples.append(dict(
                path = os.path.join(data_dir, 'Images', imgpath[item][0][0]),
                target = labels[item],
                scene = scene[item],
                img_name = imgpath[item][0][0],
            ))

        self.transform = transform
