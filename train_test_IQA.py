import os
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import time

from g_iqa.IQASolver import IQASolver

# import multiprocessing
# try:
#     multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass


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
        srcc, plcc, _, _ = solver.train()
        srcc_all.append(srcc)
        plcc_all.append(plcc)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        del solver
        torch.cuda.empty_cache()
    
    print(srcc_all)
    print(plcc_all)
    srcc_mean, plcc_mean = np.mean(srcc_all), np.mean(plcc_all)
    srcc_med, plcc_med = np.median(srcc_all), np.median(plcc_all)

    print('Testing mean SRCC %4.4f,\tmean PLCC %4.4f' % (srcc_mean, plcc_mean))
    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
    



def load_datajson_for_cross_set(datasets:list, json_dir:str, istrain:bool, dataset_domain=False):
    '''
    datasets: list of dataset name
    json_dir: directory of json files
    istrain: True for training, False for testing
    '''
    train_json = {
        'spaq': 'for_cross_set/train/spaq_train.json',
        'koniq10k': 'for_cross_set/train/koniq10k_train.json',

        'para': 'for_cross_set/train/para_train.json',
    }
    test_json = {
        'spaq': 'for_cross_set/test/spaq_test.json',
        'koniq10k': 'for_cross_set/test/koniq10k_test.json',
        'livec': 'for_cross_set/test/livec.json',

        'bid': 'for_cross_set/test/bid.json',
        'cid2013': 'for_cross_set/test/cid2013.json',

        'live': 'for_cross_set/test/live.json',
        'csiq': 'for_cross_set/test/csiq.json',

        'para': 'for_cross_set/test/para_test.json',
    }
    datajson = {}
    for dataname in datasets:
        json_file = train_json[dataname] if istrain else test_json[dataname]
        with open(os.path.join(json_dir, json_file), 'r') as f:
            datajson[dataname] = json.load(f)['files']
        if dataset_domain:
            for item in datajson[dataname]:
                item['domain_id'] = int(item['domain_id'] / 100)
    return datajson

def load_datajson_from_random(datasets:list, json_dir:str, istrain:bool, cnt=1, dataset_domain=False):
    '''
    datasets: list of dataset name
    json_dir: directory of json files
    istrain: True for training, False for testing
    cnt: number of train-test times
    '''
    datajson = {}
    for dataname in datasets:
        train_json = f'random_split/{dataname}/{cnt}/train.json'
        test_json = f'random_split/{dataname}/{cnt}/test.json'
        json_file = train_json if istrain else test_json
        with open(os.path.join(json_dir, json_file), 'r') as f:
            datajson[dataname] = json.load(f)['files']
        if dataset_domain:
            for item in datajson[dataname]:
                item['domain_id'] = int(item['domain_id'] / 100)
        
        # 4x LIVE and CSIQ
        if istrain and dataname in ['live', 'csiq']:
            datajson[dataname] = datajson[dataname] * 4
    return datajson

def random_split_exp(config):
    '''
    Random split dataset for 10 times, following LIQE split
    '''
    srcc_all = {dname: [] for dname in config.test_dataset}
    plcc_all = {dname: [] for dname in config.test_dataset}
    srcc_by_epoch = {dname: [] for dname in config.test_dataset}
    plcc_by_epoch = {dname: [] for dname in config.test_dataset}
    for i in range(1, 11):
        print('Train-test %d ...' % i)
        train_datajson = load_datajson_from_random(config.train_dataset, config.json_dir, istrain=True, cnt=i, dataset_domain=config.dataset_domain)
        test_datajson = load_datajson_from_random(config.test_dataset, config.json_dir, istrain=False, cnt=i)

        solver = IQASolver(config, config.data_root, train_datajson, test_datajson)
        srcc, plcc, srcc_e, plcc_e = solver.train()
        for dname in config.test_dataset:
            srcc_all[dname].append(srcc[dname])
            plcc_all[dname].append(plcc[dname])
            srcc_by_epoch[dname].append(srcc_e[dname])
            plcc_by_epoch[dname].append(plcc_e[dname])
        
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        del solver
        torch.cuda.empty_cache()
        time.sleep(60)
        
    if torch.distributed.get_rank() != 0:
        for dname in config.test_dataset:
            srcc_mean, plcc_mean = np.mean(srcc_all[dname]), np.mean(plcc_all[dname])
            srcc_med, plcc_med = np.median(srcc_all[dname]), np.median(plcc_all[dname])
            srcc_std, plcc_std = np.std(srcc_all[dname]), np.std(plcc_all[dname])
            print('Testing %s dataset:' % dname)
            print('mean SRCC %4.4f,\tmean PLCC %4.4f' % (srcc_mean, plcc_mean))
            print('median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
            print('std SRCC %4.4f,\tstd PLCC %4.4f' % (srcc_std, plcc_std))

            val_epoch_len = len(srcc_by_epoch[dname][0])
            print(srcc_by_epoch[dname])
        
        for e in range(val_epoch_len):
            print('\n\nEpoch %d:' % e)
            for dname in config.test_dataset:
                srcc_e = [srcc_by_epoch[dname][i][e] for i in range(10)]
                plcc_e = [plcc_by_epoch[dname][i][e] for i in range(10)]
                srcc_mean, plcc_mean = np.mean(srcc_e), np.mean(plcc_e)
                srcc_med, plcc_med = np.median(srcc_e), np.median(plcc_e)
                srcc_std, plcc_std = np.std(srcc_e), np.std(plcc_e)
                print('Testing %s dataset:' % dname)
                print('mean SRCC %4.4f,\tmean PLCC %4.4f' % (srcc_mean, plcc_mean))
                print('median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
                print('std SRCC %4.4f,\tstd PLCC %4.4f' % (srcc_std, plcc_std))


def cross_dataset_exp(config):

    train_datajson = load_datajson_for_cross_set(config.train_dataset, config.json_dir, istrain=True, dataset_domain=config.dataset_domain)
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
    parser.add_argument('--input_size', dest='input_size', type=int, default=224, help='Crop size for training & testing image')
    parser.add_argument('--data_root', type=str, default='./data', help='data root')
    parser.add_argument('--json_dir', type=str, default='./data_json', help='data json')
    parser.add_argument('--dataset_domain', action="store_true", help="True if domain is dataset, False if domain is scene")

    ################## Training Config ##################
    # parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    parser.add_argument('--epochs', dest='epochs', type=int, default=35, help='Epochs for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--project_dir', type=str, default='exp_log/debug', help='project dir for tensorboard')
    parser.add_argument('--mixed_precision', default='no', choices=['no', 'fp16'], help='Mixed precision training, no(fp32)|fp16')
    ################## optimize Config ##################
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=1, help='Learning rate ratio for backbone i.e. lr/lr_ratio')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5, help='Weight decay of Adam')
    parser.add_argument('--warmup_epoch', type=int, default=5, help='epoch of warmup')
    ################## important Config ##################
    parser.add_argument('--loss_type', type=str, default='l1', help='loss type for training')
    parser.add_argument('--scene_sampling', type=int, default=0, help='Number of domain categories, must be greater than 0, otherwise random sampling. Note: it will actually *2 if use DDP')

    ################## Abalation Config ##################
    parser.add_argument('--all_global', action="store_true", help='Use all global features')
    parser.add_argument('--local_global', action="store_true", help='Use our local global complementary token combination')
    parser.add_argument('--exp_type', type=str, choices=['leave-one-out', 'cross-set', 'random-split'], help='Experiment type')

    config = parser.parse_args()

    # os.environ["http_proxy"] = 'http://192.168.195.225:7890'
    # os.environ["https_proxy"] = 'http://192.168.195.225:7890'

    if config.exp_type == 'cross-set':
        cross_dataset_exp(config)
    elif config.exp_type == 'leave-one-out':
        leave_one_out_exp(config)
    elif config.exp_type == 'random-split':
        random_split_exp(config)
    else:
        raise NotImplementedError

