import os
import numpy as np
import scipy

from g_iqa.g_datasets.folder.PIQ23_base import PIQ23Folder

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


class CSIQFolder(PIQ23Folder):
    def __init__(self, root, data_json, transform):
        super().__init__(root, data_json, transform)
    '''
    # old version - Not verified
    def __init__(self, root, index=None, transform=None, scene_base=600):

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath,'.png')
        txtpath = os.path.join(root, 'csiq_label.txt')
        fh = open(txtpath, 'r')
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                sample.append((os.path.join(root, 'dst_imgs_all', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform
    '''


if __name__ == "__main__":
    import json
    with open('/home/dzc/workspace/G-IQA/data/data_json/for_cross_set/test/csiq.json', 'r') as f:
        data_json = json.load(f)['files']
    data = CSIQFolder(root='/home/dzc/workspace/G-IQA/data', data_json=data_json, transform=None)
    print(data[0])