from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import json
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cv2

from train_utils import loss_by_scene

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass


class JSONRatingsDataset(Dataset):
    """General No Reference dataset with meta info file.
    """
    def __init__(self, json_data, root_dir, transform=None):
        self.paths_mos = json_data
        for path_mos in self.paths_mos:
            path_mos['image'] = os.path.join(root_dir, path_mos['image'])
        
        self.transform = transform

    def __len__(self):
        return len(self.paths_mos)

    def __getitem__(self, index):

        img_path = self.paths_mos[index]['image']
        mos_label = float(self.paths_mos[index]['score'])

        im = Image.open(img_path).convert('RGB')
        if im.mode == 'P':
            im = im.convert('RGB')
        image = np.asarray(im)

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

        sample = {'image': image, 'rating': mos_label}

        if self.transform:
            sample = self.transform(sample)
        sample['scene'] = self.paths_mos[index]['domain_id']
        return sample
    
    def get_scene_list(self):
        return [sample['domain_id'] for sample in self.paths_mos]



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # image = transform.resize(image, (new_h, new_w))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}

class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out

class Net(nn.Module):
    def __init__(self , resnet, net):
        super(Net, self).__init__()
        self.resnet_layer = resnet
        self.net = net


    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.net(x)

        return x


def computeSpearman(dataloader_valid, model):
    ratings = []
    predictions = []
    scenes = []
    with torch.no_grad():
        cum_loss = 0
        for batch_idx, data in enumerate(tqdm(dataloader_valid)):
            inputs = data['image']
            batch_size = inputs.size()[0]
            labels = data['rating'].view(batch_size, -1)
            # labels = labels / 100.0
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                except:
                    print(inputs, labels)
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs_a = model(inputs)
            ratings.append(labels.float().cpu())
            predictions.append(outputs_a.float().cpu())
            scenes.append(data['scene'].view(batch_size, -1).cpu())

    ratings_i = np.vstack(ratings)
    predictions_i = np.vstack(predictions)
    scenes_i = np.vstack(scenes)
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)[0]
    pl = pearsonr(a,b)[0]

    # compute the spearman correlation for each scene (pipal, tid2013)
    if False:
        sp_list = []
        pl_list = []
        for scene in set(scenes_i[:,0]):
            scene_ratings = ratings_i[scenes_i[:,0] == scene, 0]
            scene_predictions = predictions_i[scenes_i[:,0] == scene, 0]
            sp = spearmanr(scene_ratings, scene_predictions)[0]
            pl = pearsonr(scene_ratings, scene_predictions)[0]
            sp_list.append(abs(sp))
            pl_list.append(abs(pl))
        return np.mean(sp_list), np.mean(pl_list), a, b

    return abs(sp), abs(pl), a, b

def finetune_model(train_json, test_json_dict, epochs=100):
    # epochs = 100
    srocc_l = []

    # data_dir = os.path.join('LIVE_WILD')
    # images = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_score.csv'), sep=',')

    # images_fold = "LIVE_WILD/"
    # if not os.path.exists(images_fold):
    #     os.makedirs(images_fold)
    for i in range(1):
        # images_train, images_test = train_test_split(images, train_size = 0.8)

        # train_path = images_fold + "train_image" + ".csv"
        # test_path = images_fold + "test_image" + ".csv"
        # images_train.to_csv(train_path, sep=',', index=False)
        # images_test.to_csv(test_path, sep=',', index=False)

        model = torch.load('model_IQA/TID2013_KADID10K_IQA_Meta_resnet18.pt')
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4,  weight_decay=0)
        model.cuda()

        best_srcc = {dname: 0 for dname in test_json_dict.keys()}
        best_plcc = {dname: 0 for dname in test_json_dict.keys()}
        spearman = 0
        for epoch in range(epochs):
            if epoch % 2 == 0:
                print('############# test phase epoch %2d ###############' % epoch)
                for dname, test_json in test_json_dict.items():
                    dataloader_valid = load_data('test', test_json=test_json)
                    model.eval()
                    model.cuda()
                    srcc, plcc, gt, pred = computeSpearman(dataloader_valid, model)
                    if srcc > best_srcc[dname]:
                        best_srcc[dname] = srcc
                        best_plcc[dname] = plcc
                        pred_gt = np.stack((pred, gt), axis=1)
                        np.savetxt(f'{save_dir}/pred_gt_{dname}.txt', pred_gt, fmt='%.4f')

                    print(f'Test on {dname} dataset')
                    print('current srocc {:4f}, best srocc {:4f}'.format(srcc, best_srcc[dname]))
                    print('current plcc {:4f}, best plcc {:4f}'.format(plcc, best_plcc[dname]))


            # print('############# train phase epoch %2d ###############' % epoch)
            # optimizer = exp_lr_scheduler(optimizer, epoch)

            # if epoch == 0:
            #     dataloader_valid = load_data('train')
            #     model.eval()

            #     sp = computeSpearman(dataloader_valid, model)[0]
            #     if sp > spearman:
            #         spearman = sp
            #     print('no train srocc {:4f}'.format(sp))

            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train = load_data('train', train_json=train_json)
            model.train()  # Set model to training mode
            for batch_idx, data in enumerate(tqdm(dataloader_train)):
                inputs = data['image']
                batch_size = inputs.size()[0]
                labels = data['rating'].view(batch_size, -1)
                scene = data['scene']
                # labels = labels / 100.0
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    except:
                        print(inputs, labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                if args.alignment:
                    loss = loss_by_scene(outputs, labels, scene, 'l1')
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # #print('############# test phase epoch %2d ###############' % epoch)
            # dataloader_valid = load_data('test')
            # model.eval()

            # sp, pl = computeSpearman(dataloader_valid, model)
            # if sp > spearman:
            #     spearman = sp
            # print('Validation Results - Epoch: {:2d}, PLCC: {:4f}, SROCC: {:4f}, '
            #       'best SROCC: {:4f}'.format(epoch, pl, sp, spearman))

        srocc_l.append(spearman)

    # ind = 'Results/LIVEWILD'
    # file = pd.DataFrame(columns=[ind], data=srocc_l)
    # file.to_csv(ind+'.csv')
    # print('average srocc {:4f}'.format(np.mean(srocc_l)))

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.8**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)




class SceneSampler(Sampler[int]):
    '''
    random sample, sample scene within a batch are all the same
    '''
    def __init__(self, data_source, batch_size, scene_sampling=1):
        self.data_source = data_source
        self.batch_size = batch_size
        self.scene_sampling = scene_sampling
        self.scene_bs = self.batch_size // self.scene_sampling
        self.data_len = len(data_source)

        self.update()

    def __iter__(self):
        if self.data_len != len(self.data_source):
            self.update()
        # init seed
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        random.seed(seed)

        
        # pad each len(scene_index) to an integer multiple of batch_size
        scene_index_pad = {}
        for s in self.scene:
            pad = (self.scene_bs - len(self.scene_index[s]) % self.scene_bs) % self.scene_bs
            # random pad
            scene_index_pad[s] = self.scene_index[s] + [random.choice(self.scene_index[s]) for _ in range(pad)]
            # shuffle
            random.shuffle(scene_index_pad[s])

        # [[scene_batch], [scene_batch], ...]
        index_scene_batch = []
        for s in self.scene:
            scene_batch = [scene_index_pad[s][i:i+self.scene_bs] for i in range(0, len(scene_index_pad[s]), self.scene_bs)]
            # index_scene_batch.append(scene_batch)
            index_scene_batch.extend(scene_batch)

        # shuffle scene_batch
        random.shuffle(index_scene_batch)

        # yield index
        yield from [index for scene_batch in index_scene_batch for index in scene_batch]
             

    def __len__(self):
        if self.data_len != len(self.data_source):
            self.update()
        return self.num_samples
    
    def update(self):
        scene_list = self.data_source.get_scene_list()
        self.scene = set(scene_list)
        self.scene_index = {s: [i for i, scene in enumerate(scene_list) if scene == s] for s in self.scene}
        self.num_samples = sum([math.ceil(len(self.scene_index[s]) / self.scene_bs) * self.scene_bs for s in self.scene])


def load_data(mod = 'train', train_json=None, test_json=None):
    if mod == 'train':
        with open(train_json, 'r') as f:
            train_data = json.load(f)['files']
        # normalize the score to [0, 1]
        # score_mean = sum(item['score'] for item in train_data) / len(train_data)
        # score_std = math.sqrt(sum((item['score'] - score_mean)**2 for item in train_data) / len(train_data))
        # for item in train_data:
        #     item['score'] = (item['score'] - score_mean) / score_std
        max_score = max(item['score'] for item in train_data)
        min_score = min(item['score'] for item in train_data)
        for item in train_data:
            item['score'] = (item['score'] - min_score) / (max_score - min_score)

        output_size = (224, 224)
        transformed_dataset_train = JSONRatingsDataset(train_data, 
                                                       root_dir='data',
                                                       transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(
                                                                                      output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
        if args.alignment:
            scene_sampler = SceneSampler(transformed_dataset_train, batch_size=128, scene_sampling=4)
            dataloader = DataLoader(transformed_dataset_train, batch_size=128,
                                    shuffle=False, num_workers=4, collate_fn=my_collate, sampler=scene_sampler, pin_memory=True, drop_last=True)
        else:
            dataloader = DataLoader(transformed_dataset_train, batch_size=128,
                                    shuffle=False, num_workers=4, collate_fn=my_collate)
    else:
        with open(test_json, 'r') as f:
            data = json.load(f)['files']
        output_size = (224, 224)
        transformed_dataset_valid = JSONRatingsDataset(data, 
                                                       root_dir='data', 
                                                       transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                        Normalize(),
                                                                                        ToTensor(),
                                                                                        ]))
        dataloader = DataLoader(transformed_dataset_valid, batch_size= 50,
                                    shuffle=False, num_workers=4, collate_fn=my_collate)



    # meta_num = 50
    # data_dir = os.path.join('LIVE_WILD/')
    # train_path = os.path.join(data_dir,  'train_image.csv')
    # test_path = os.path.join(data_dir,  'test_image.csv')

    # output_size = (224, 224)
    # transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
    #                                                 root_dir='/home/hancheng/IQA/iqa-db/LIVE_WILD/images/',
    #                                                 transform=transforms.Compose([Rescale(output_size=(256, 256)),
    #                                                                               RandomHorizontalFlip(0.5),
    #                                                                               RandomCrop(
    #                                                                                   output_size=output_size),
    #                                                                               Normalize(),
    #                                                                               ToTensor(),
    #                                                                               ]))
    # transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
    #                                                 root_dir='/home/hancheng/IQA/iqa-db/LIVE_WILD/images/',
    #                                                 transform=transforms.Compose([Rescale(output_size=(224, 224)),
    #                                                                               Normalize(),
    #                                                                               ToTensor(),
    #                                                                               ]))
    # bsize = meta_num

    # if mod == 'train':
    #     dataloader = DataLoader(transformed_dataset_train, batch_size=bsize,
    #                               shuffle=False, num_workers=0, collate_fn=my_collate)
    # else:
    #     dataloader = DataLoader(transformed_dataset_valid, batch_size= 50,
    #                                 shuffle=False, num_workers=0, collate_fn=my_collate)

    return dataloader

# finetune_model()

if __name__ == '__main__':
    # add argument parser
    import sys
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--leave_one_out', action='store_true')
    parser.add_argument('--train_data', type=str, default='spaq')
    parser.add_argument('--alignment', action='store_true')
    args = parser.parse_args()

    if args.leave_one_out:
        # for leave one out exp
        for domain_dir in os.listdir(f'data_json/for_leave_one_out/{args.train_data}'):
            save_dir = f'exp_results/leave_one_out/{args.train_data}_{args.alignment}/{domain_dir}'
            os.makedirs(save_dir, exist_ok=True)

            train_json = f'data_json/for_leave_one_out/{args.train_data}/{domain_dir}/train.json'
            test_json = {
                args.train_data: f'data_json/for_leave_one_out/{args.train_data}/{domain_dir}/test.json'
            }

            with open(f'{save_dir}/train.log', 'w') as f:
                sys.stdout = f
                finetune_model(train_json=train_json, test_json_dict=test_json, epochs=50)

    else:
        # for cross dataset
        save_dir = f'exp_results/cross_set/{args.train_data}'
        os.makedirs(save_dir, exist_ok=True)

        train_json = f'data_json/for_cross_set/train/{args.train_data}_train.json'
        test_json_dict = {
            # 'koniq10k': 'data_json/for_cross_set/test/koniq10k_test.json',
            # 'spaq': 'data_json/for_cross_set/test/spaq_test.json',
            # 'livec': 'data_json/for_cross_set/test/livec.json',
            # 'live': 'data_json/for_cross_set/test/live.json',
            # 'csiq': 'data_json/for_cross_set/test/csiq.json',
            # 'bid': 'data_json/for_cross_set/test/bid.json',
            # 'cid2013': 'data_json/for_cross_set/test/cid2013.json',
            # 'nnid': 'data_json/for_cross_set/test/nnid.json',
            'pipal': 'data_json/for_cross_set/test/pipal_test.json',
            'tid2013': 'data_json/for_cross_set/test/tid2013_test.json',
        }

        with open(f'{save_dir}/train.log', 'w') as f:
            sys.stdout = f
            finetune_model(train_json=train_json, test_json_dict=test_json_dict)
