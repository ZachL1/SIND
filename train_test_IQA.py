import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch

from g_iqa.IQASolver import IQASolver

# import multiprocessing
# try:
#     multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_piq23_train_test(data_dir):
    all_set = os.path.join(data_dir, 'Scores_Overall.csv')
    test_set = f'{data_dir}/../piq23_starter_kit/ntire24_overall_scene_test.csv'

    all_set = pd.read_csv(all_set)['IMAGE PATH'].tolist()
    test_set = pd.read_csv(test_set)['IMAGE PATH'].tolist()
    train_set = list(set(all_set) - set(test_set))

    # generate train and test set index in all_set
    train_idx = [all_set.index(img) for img in train_set]
    test_idx = [all_set.index(img) for img in test_set]

    return train_idx, test_idx


def split_dataset_by_domain(data_dir, domain_idx):
    '''
    leave one domain out
    domain can be scene-based or distortion-based 
    '''
    if data_dir.endswith('PIQ23/'):
        all_df = pd.read_csv(os.path.join(data_dir, 'Scores_Overall.csv'))
        all_scene = all_df['CONDITION'].unique()
        domain = all_scene[domain_idx]
        all_scene = sorted(all_scene)
        train_idx = all_df[all_df['CONDITION'] != domain].index.tolist()
        test_idx = all_df[all_df['CONDITION'] == domain].index.tolist()

    elif data_dir.endswith('SPAQ/'):
        annos_df = pd.read_excel(os.path.join(data_dir, 'annotations/MOS and Image attribute scores.xlsx'))
        # the category probability distribution of images, index is the image name
        scene_df = pd.read_excel(os.path.join(data_dir, 'annotations/Scene category labels.xlsx'))
        assert annos_df['Image name'].tolist() == scene_df['Image name'].tolist()
        all_scene = scene_df.columns[1:]
        all_scene = sorted(all_scene)
        domain = all_scene[domain_idx]
        train_idx = scene_df[scene_df[domain] == 0].index.tolist()
        test_idx = scene_df[scene_df[domain] > 0].index.tolist()

    elif data_dir.endswith('PARA/'):
        all_df = pd.read_csv(os.path.join(data_dir, 'annotation/PARA-GiaaAll.csv'))
        all_scene = all_df['semantic'].unique()
        all_scene = sorted(all_scene)
        domain = all_scene[domain_idx]
        train_idx = all_df[all_df['semantic'] != domain].index.tolist()
        test_idx = all_df[all_df['semantic'] == domain].index.tolist()

    elif data_dir.endswith('EVA/'):
        all_set = os.path.join(data_dir, 'annotations/EVA_all.json')
        with open(all_set, 'r') as f:
            all_js = json.load(f)['files']
        all_scene = list(set(item['category'] for item in all_js))
        all_scene = sorted(all_scene)
        domain = all_scene[domain_idx]
        train_idx = [i for i, item in enumerate(all_js) if item['category'] != domain]
        test_idx = [i for i, item in enumerate(all_js) if item['category'] == domain]



    else:
        raise NotImplementedError

    print(all_scene)
    return train_idx, test_idx

def main(config):

    folder_path = {
        'live': '/home/ssl/Database/databaserelease2/',
        'csiq': '/home/ssl/Database/CSIQ/',
        'tid2013': '/home/ssl/Database/TID2013/',
        'livec': '/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/',
        # 'koniq-10k': '/home/ssl/Database/koniq-10k/',
        # 'bid': '/home/ssl/Database/BID/',
        'piq23': './data/PIQ23/',
        'spaq': './data/SPAQ/',
        'koniq10k': './data/koniq10k/',
        'livew': './data/LIVEW/',
        'bid': './data/BID/',
        'para': './data/PARA/',
        'eva': './data/EVA/',
    }

    domain_lists = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        # 'livec': list(range(0, 1162)),
        # 'koniq-10k': list(range(0, 10073)),
        # 'bid': list(range(0, 586)),
        'piq23': list(range(0, 4)), # Outdoor, Indoor, Lowlight, Night
        'spaq': list(range(0, 9)), # Animal, Cityscape, Human, Indoor scene, Landscape, Night scene, Plant, Still-life, Others
        'para': list(range(0, 10)), # 10 scenes
        'eva': list(range(0, 6)), # 6 scenes

    }
    domain_list = domain_lists[config.train_dataset[0]] if len(config.train_dataset) == 1 else [0]

    srcc_all = np.zeros(len(domain_list), dtype=np.float32)
    plcc_all = np.zeros(len(domain_list), dtype=np.float32)

    proj_dir = config.project_dir
    for i in domain_list:
        print('Training and testing on %s dataset for domain %d ...' % (config.train_dataset, i))
        config.project_dir = os.path.join(proj_dir, f'test_domain_{i}')

        if len(config.train_dataset) == 1:
            assert config.train_dataset == config.test_dataset
            train_index, test_index = split_dataset_by_domain(folder_path[config.train_dataset[0]], i)
        else:
            train_index, test_index = None, None

        solver = IQASolver(config, folder_path, train_index, test_index)
        srcc_all[i], plcc_all[i] = solver.train()

        del solver
        torch.cuda.empty_cache()

    print(srcc_all)
    print(plcc_all)
    srcc_mean, plcc_mean = srcc_all.mean(), plcc_all.mean()
    srcc_med, plcc_med = np.median(srcc_all), np.median(plcc_all)

    print('Testing mean SRCC %4.4f,\tmean PLCC %4.4f' % (srcc_mean, plcc_mean))
    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))


def leave_one_out_exp(config):
    assert len(config.train_dataset) == 1, 'Only support one dataset for leave-one-out experiment'
    train_data_name = config.train_dataset[0]
    json_dir = os.path.join(config.json_dir, 'for_leave_one_out', train_data_name)

    srcc_all = []
    plcc_all = []

    proj_dir = config.project_dir
    for domain_dir in os.listdir(json_dir):
        # if int(domain_dir.split('_')[2]) not in [102, 104, ]:
        #     continue

        test_domain, d_name = domain_dir.split('_')[-2:]
        print('Training and testing on %s dataset for domain %s ...' % (train_data_name, d_name))

        with open(os.path.join(json_dir, domain_dir, 'train.json'), 'r') as f:
            train_datajson = json.load(f)['files']
        with open(os.path.join(json_dir, domain_dir, 'test.json'), 'r') as f:
            test_datajson = json.load(f)['files']

        config.project_dir = os.path.join(proj_dir, f'test_domain_{test_domain}')

        solver = IQASolver(config, config.data_root, {train_data_name: train_datajson}, {train_data_name: test_datajson})
        srcc, plcc = solver.train()
        srcc_all.append(srcc)
        plcc_all.append(plcc)

        torch.distributed.barrier()
        del solver
        torch.cuda.empty_cache()
    
    print(srcc_all)
    print(plcc_all)
    srcc_mean, plcc_mean = np.mean(srcc_all), np.mean(plcc_all)
    srcc_med, plcc_med = np.median(srcc_all), np.median(plcc_all)

    print('Testing mean SRCC %4.4f,\tmean PLCC %4.4f' % (srcc_mean, plcc_mean))
    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
    



def load_datajson_for_cross_set(datasets:list, json_dir:str, istrain:bool):
    '''
    datasets: list of dataset name
    json_dir: directory of json files
    istrain: True for training, False for testing
    '''
    train_json = {
        'spaq': 'for_cross_set/train/spaq_train.json',
        'koniq10k': 'for_cross_set/train/koniq10k_train.json',
        'kadid10k': 'for_cross_set/train/kadid10k_train.json',
    }
    test_json = {
        'spaq': 'for_cross_set/test/spaq_test.json',
        'livec': 'for_cross_set/test/livec.json',
        'koniq10k': 'for_cross_set/test/koniq10k_test.json',
        'bid': 'for_cross_set/test/bid.json',
        'cid2013': 'for_cross_set/test/cid2013.json',

        'agiqa3k': 'for_cross_set/test/agiqa3k.json',

        'kadid10k': 'for_cross_set/test/kadid10k_test.json',
        'live': 'for_cross_set/test/live.json',
        'csiq': 'for_cross_set/test/csiq.json',
    }
    datajson = {}
    for dataname in datasets:
        json_file = train_json[dataname] if istrain else test_json[dataname]
        with open(os.path.join(json_dir, json_file), 'r') as f:
            datajson[dataname] = json.load(f)['files']
    return datajson

def cross_dataset_exp(config):

    train_datajson = load_datajson_for_cross_set(config.train_dataset, config.json_dir, istrain=True)
    test_datajson = load_datajson_for_cross_set(config.test_dataset, config.json_dir, istrain=False)

    solver = IQASolver(config, config.data_root, train_datajson, test_datajson)
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ################## Model Config ##################
    parser.add_argument('--clip_model', default='openai/ViT-B-16', help="pretrained clip weights, see https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv")
    parser.add_argument('--clip_freeze', action="store_true", help="freeze clip weights")
    parser.add_argument('--ema_decay', dest='ema_decay', type=float, default=0.99, help='ema decay')
    ################## Resume/Finetune Config ##################
    parser.add_argument('--resume_from', type=str, default=None, help="resume from checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0, help="resume start epoch")
    parser.add_argument('--load_from', type=str, default=None, help="load from checkpoint")

    ################## Data Config ##################
    parser.add_argument('--train_dataset', dest='train_dataset', nargs='+', type=str, default='piq23', help='datasets for training, piq23|spaq|koniq10k|livew|bid')
    parser.add_argument('--test_dataset', dest='test_dataset', nargs='+', type=str, default='piq23', help='datasets for testing')
    parser.add_argument('--input_size', dest='input_size', type=int, default=672, help='Crop size for training & testing image')
    parser.add_argument('--data_root', type=str, default='./data', help='data root')
    parser.add_argument('--json_dir', type=str, default='./data_json', help='data json')

    ################## Training Config ##################
    # parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='Epochs for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--project_dir', type=str, default='exp_log/debug', help='project dir for tensorboard')
    parser.add_argument('--mixed_precision', default='no', choices=['no', 'fp16'], help='Mixed precision training, no(fp32)|fp16')
    ################## optimize Config ##################
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=1, help='Learning rate ratio for backbone i.e. lr/lr_ratio')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay of Adam')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='epoch of warmup')
    ################## important Config ##################
    parser.add_argument('--loss_type', type=str, default='l1', help='loss type for training')
    parser.add_argument('--scene_sampling', type=int, default=0, help='Number of domain categories, must be greater than 0, otherwise random sampling')

    ################## Abalation Config ##################
    parser.add_argument('--all_global', action="store_true", help='Use all global features')
    parser.add_argument('--exp_type', type=str, choices=['leave-one-out', 'cross-set'], help='Experiment type')

    config = parser.parse_args()

    # os.environ["http_proxy"] = 'http://192.168.195.225:7890'
    # os.environ["https_proxy"] = 'http://192.168.195.225:7890'

    if config.exp_type == 'cross-set':
        cross_dataset_exp(config)
    elif config.exp_type == 'leave-one-out':
        leave_one_out_exp(config)
    else:
        raise NotImplementedError

