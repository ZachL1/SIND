import os
import argparse
import random
import numpy as np
import pandas as pd
from IQASolver import IQASolver

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
    }

    # img_num = {
    #     'live': list(range(0, 29)),
    #     'csiq': list(range(0, 30)),
    #     'tid2013': list(range(0, 25)),
    #     'livec': list(range(0, 1162)),
    #     'koniq-10k': list(range(0, 10073)),
    #     'bid': list(range(0, 586)),
    # }
    # sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float32)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float32)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        # print('Round %d' % (i+1))
        # # Randomly select 80% images for training and the rest for testing
        # random.shuffle(sel_num)
        # train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        # test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        train_index, test_index = get_piq23_train_test(folder_path['piq23'])
        config.project_dir = os.path.join(config.project_dir, f'train_time_{i}')
        datasets_dir = [
            folder_path[config.dataset],
            folder_path['spaq'],
            folder_path['koniq10k'],
            folder_path['livew'],
            folder_path['bid'],
        ]
        solver = IQASolver(config, datasets_dir, train_index, test_index)
        srcc_all[i], plcc_all[i] = solver.train()

    # print(srcc_all)
    # print(plcc_all)
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))

    # return srcc_med, plcc_med


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ################## Model Config ##################
    parser.add_argument('--clip_model', default='openai/ViT-B-16')
    parser.add_argument('--clip_freeze', action="store_true")
    parser.add_argument('--ema_decay', dest='ema_decay', type=float, default=0.99, help='ema decay')
    ################## Resume/Finetune Config ##################
    parser.add_argument('--resume_from', type=str, default=None, help="resume from checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0, help="resume start epoch")
    parser.add_argument('--load_from', type=str, default=None, help="load from checkpoint")

    ################## Data Config ##################
    parser.add_argument('--dataset', dest='dataset', type=str, default='piq23', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--input_size', dest='input_size', type=int, default=672, help='Crop size for training & testing image')
    parser.add_argument('--train_data', nargs='+', type=int, default=[0,1,2,3,4], help="train data, 0:PIQ23, 1:SPAQ, 2:Koniq10K, 3:Livew, 4:BID")
    parser.add_argument('--val_data', nargs='+', type=int, default=[0], help="val data, 0:PIQ23, 1:SPAQ, 2:Koniq10K, 3:Livew, 4:BID")

    ################## Training Config ##################
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='Epochs for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--project_dir', type=str, default='PIQ2023/output/debug', help='project dir for tensorboard')
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

    config = parser.parse_args()

    # os.environ["http_proxy"] = 'http://10.147.18.225:7890'
    # os.environ["https_proxy"] = 'http://10.147.18.225:7890'
    main(config)

