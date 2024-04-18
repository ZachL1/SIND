import os
import csv
import pandas as pd

from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder


class Koniq_10kFolder(PIQ23Folder):

    def __init__(self, root, index=None, transform=None, scene_base=300):
        # data_dir = os.path.join(root, 'koniq10k')
        data_dir = root
        imgname = []
        mos_all = []
        csv_file = os.path.join(data_dir, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = float(row['MOS_zscore'])
                mos_all.append(mos)
        scene = [scene_base] * len(imgname)
        if index is None:
            index = list(range(len(imgname)))

        sample = []
        for i, item in enumerate(index):
            sample.append(dict(
                path = os.path.join(data_dir, '1024x768', imgname[item]),
                target = mos_all[item],
                scene = scene[item],
                img_name = imgname[item],
            ))

        self.samples = sample
        self.transform = transform

