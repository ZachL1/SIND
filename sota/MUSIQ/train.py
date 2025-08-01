import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


from option.config import Config
from model.model_main import IQARegression
from model.backbone import resnet50_backbone
from trainer import train_epoch, eval_epoch
from utils.util import RandHorizontalFlip, Normalize, ToTensor, RandShuffle

import sys


def run_train(train_json, test_json_dict, epoch):
    # config file
    config = Config({
        # device
        'gpu_id': "0",                          # specify GPU number to use
        'num_workers': 4,

        # data
        'db_name': 'KonIQ-10k',                                     # database type
        'db_path': 'data/',                                         # root path of database    
        # 'txt_file_name': './IQA_list/koniq-10k.txt',                # list of images in the database
        'train_size': 0.8,                                          # train/vaildation separation ratio
        'scenes': 'all',                                            # using all scenes
        'scale_1': 384,                                             
        'scale_2': 224,
        'batch_size': 4,
        'patch_size': 32,

        # ViT structure
        'n_enc_seq': 32*24 + 12*9 + 7*5,        # input feature map dimension (N = H*W) from backbone
        'n_layer': 14,                          # number of encoder layers
        'd_hidn': 384,                          # input channel of encoder (input: C x N)
        'i_pad': 0,
        'd_ff': 384,                            # feed forward hidden layer dimension
        'd_MLP_head': 1152,                     # hidden layer of final MLP
        'n_head': 6,                            # number of head (in multi-head attention)
        'd_head': 384,                          # channel of each head -> same as d_hidn
        'dropout': 0.1,                         # dropout ratio
        'emb_dropout': 0.1,                     # dropout ratio of input embedding
        'layer_norm_epsilon': 1e-12,
        'n_output': 1,                          # dimension of output
        'Grid': 10,                             # grid of 2D spatial embedding

        # optimization & training parameters
        'n_epoch': epoch,                         # total training epochs
        'learning_rate': 1e-4,                  # initial learning rate
        'weight_decay': 0,                      # L2 regularization weight
        'momentum': 0.9,                        # SGD momentum
        'T_max': 3e4,                           # period (iteration) of cosine learning rate decay
        'eta_min': 0,                           # minimum learning rate
        'save_freq': 10,                        # save checkpoint frequency (epoch)
        'val_freq': 5,                          # validation frequency (epoch)


        # load & save checkpoint
        'snap_path': './weights',               # directory for saving checkpoint
        'checkpoint': None,                     # load checkpoint
    })
    
    # device setting
    config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Using GPU %s' % config.gpu_id)
    else:
        print('Using CPU')


    # data selection
    # if config.db_name == 'KonIQ-10k':
    #     from dataset.koniq import IQADataset
    from dataset.koniq import JSONData

    # dataset separation (8:2)
    train_scene_list, test_scene_list = RandShuffle(config)
    print('number of train scenes: %d' % len(train_scene_list))
    print('number of test scenes: %d' % len(test_scene_list))

    # data load
    train_dataset = JSONData(
        db_path=config.db_path,
        txt_file_name=train_json,
        scale_1=config.scale_1,
        scale_2=config.scale_2,
        transform=transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), RandHorizontalFlip(), ToTensor()]),
        train_mode=True,
        scene_list=train_scene_list,
        train_size=config.train_size
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)

    test_loaders = {}
    for test_name, test_json in test_json_dict.items():
        test_dataset = JSONData(
            db_path=config.db_path,
            txt_file_name=test_json,
            scale_1=config.scale_1,
            scale_2=config.scale_2,
            transform= transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensor()]),
            train_mode=False,
            scene_list=test_scene_list,
            train_size=config.train_size
        )
        test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)
        test_loaders[test_name] = test_loader


    # create model
    model_backbone = resnet50_backbone().to(config.device)
    model_transformer = IQARegression(config).to(config.device)


    # loss function & optimization
    criterion = torch.nn.L1Loss()
    params = list(model_backbone.parameters()) + list(model_transformer.parameters())
    optimizer = torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)


    # load weights & optimizer
    if config.checkpoint is not None:
        checkpoint = torch.load(config.checkpoint)
        model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
        model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        start_epoch = 0

    # make directory for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    best_srcc = {test_name: 0 for test_name in test_json_dict.keys()}
    best_plcc = {test_name: 0 for test_name in test_json_dict.keys()}
    # train & validation
    for epoch in range(start_epoch, config.n_epoch):
        loss, rho_s, rho_p = train_epoch(config, epoch, model_transformer, model_backbone, criterion, optimizer, scheduler, train_loader)

        if epoch % config.val_freq != 0:
            continue
        for test_name, test_loader in test_loaders.items():
            print(f'Now testing on {test_name}')
            loss, rho_s, rho_p = eval_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader)
            if rho_s > best_srcc[test_name]:
                best_srcc[test_name] = rho_s
                best_plcc[test_name] = rho_p
            print(f'Best SROCC: {best_srcc[test_name]}')
            print(f'Best PLCC: {best_plcc[test_name]}')


if __name__ == '__main__':
    cnt = int(sys.argv[1])

    if cnt < 0: # cross dataset experiment
        cnt = -cnt

        train_json = [
            '',
            'data_json/for_cross_set/train/spaq_train.json',
            # 'data_json/for_cross_set/train/koniq10k_train.json',
            'data_json/for_cross_set/train/koniq10k_spaq_train.json',
        ][cnt]
        test_json_dict = {
            'koniq10k': 'data_json/for_cross_set/test/koniq10k_test.json',
            'spaq': 'data_json/for_cross_set/test/spaq_test.json',
            'livec': 'data_json/for_cross_set/test/livec.json',
            'cid2013': 'data_json/for_cross_set/test/cid2013.json',
            'bid': 'data_json/for_cross_set/test/bid.json',
            'live': 'data_json/for_cross_set/test/live.json',
            'csiq': 'data_json/for_cross_set/test/csiq.json',
        }

        run_train(train_json, test_json_dict, 100)

    else: # leave one out exp
        # dataname = ['eva', 'para']
        dataname = ['para']
        for name in dataname:
            folders = os.listdir(f'data_json/for_leave_one_out/{name}')[cnt:cnt+1]
            for folder in folders:
                print(f'\n\n================')
                print(f'Now processing {name}: {folder}')
                train_json = f'data_json/for_leave_one_out/{name}/{folder}/train.json'
                test_json = f'data_json/for_leave_one_out/{name}/{folder}/test.json'

                run_train(train_json, {name: test_json}, 50)


