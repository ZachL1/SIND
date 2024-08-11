import os
import sys

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models.aesclip import AesCLIP_reg
from dataset import AVA, JSONData
from utils import AverageMeter, InfoNCE
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

# import nni
# from nni.utils import merge_parameter

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import random  # 导入 random 模块


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set_seed(42)  # 设置随机种子


def init():
    parser = argparse.ArgumentParser(description="Tuned APDD_reg_weight")

    parser.add_argument('--path_to_images', type=str, default='data',
                        help='directory to images')
    parser.add_argument('--path_to_save_csv', type=str, default="data_json/for_leave_one_out/eva/test_for_1101_animals",
                        help='directory to csv_folder')
    parser.add_argument('--experiment_dir_name', type=str, default='exp_log/debug',
                        help='directory to project')


    parser.add_argument('--init_lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--num_comments', type=int, default=4, help='num of aesthetics comments')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='temperature for contrastive learning')
    parser.add_argument('--n_ctx', default=6, type=int,
                        help='length of context')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.8685, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--step_size', default=1, type=int,  # 改为 int 类型
                        help='Step size for SGD')
    parser.add_argument("--num_epoch", default=40, type=int,
                        help="epochs of training")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size of training")
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of workers used in dataloading')
    args = parser.parse_args()
    return args


# def adjust_learning_rate(params, optimizer, epoch, lr_decay_epoch=2):
#     """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
#     lr = params['init_lr'] * (0.5 ** (epoch // lr_decay_epoch))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer

def adjust_learning_rate(params, optimizer, epoch, lr_decay_epoch=1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    decay_rate = 0.5 ** (epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
    return optimizer


def get_score(opt, y_pred):
    score_np = y_pred.data.cpu().numpy()
    return y_pred, score_np


def create_data_part(opt):
    train_json_path = os.path.join(opt['path_to_save_csv'], 'AADB_train.csv')
    test_json_path = os.path.join(opt['path_to_save_csv'], 'AADB_test.csv')

    train_ds = AVA(train_json_path, opt['path_to_images'], if_train=True)
    test_ds = AVA(test_json_path, opt['path_to_images'], if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, test_loader


def create_jsondata_part(opt):
    train_json_path = os.path.join(opt['path_to_save_csv'], 'train.json')
    test_json_path = os.path.join(opt['path_to_save_csv'], 'test.json')

    train_ds = JSONData(train_json_path, opt['path_to_images'], if_train=True)
    test_ds = JSONData(test_json_path, opt['path_to_images'], if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, test_loader



def train(opt, epoch, model, loader, optimizer, scheduler, criterion_it, writer=None, global_step=None, name=None):
    model.train()
    train_losses_i = AverageMeter()
    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += int(np.prod(param.shape))
    print('Trainable params: %.4f million' % (param_num / 1e6))

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.cuda()
        y = y.type(torch.FloatTensor).cuda()

        y_pred = model(x).squeeze()  # 确保 y_pred 是标量

        # 对比loss
        # print(y_pred.size(), y.size())
        loss = criterion_it(y_pred, y)

        # loss回传
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            f.write(
                '| epoch:%d | Batch:%d | loss:%.3f \r\n'
                % (epoch, idx, loss.item()))
            f.flush()

        train_losses_i.update(loss.item(), x.size(0))

    scheduler.step()
    return train_losses_i.avg


def validate(opt, model, loader, optimizer, criterion_it, writer=None, global_step=None, name=None):
    model.eval()
    validate_losses_i = AverageMeter()
    total_pred_scores = []
    total_true_scores = []
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.cuda()
        y = y.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            y_pred = model(x).squeeze()
        # 对比loss
        loss = criterion_it(y_pred, y)
        validate_losses_i.update(loss.item(), x.size(0))

        # 计算预测分数和真实分数
        pred_score, pred_score_np = get_score(opt, y_pred)
        true_score, true_score_np = get_score(opt, y)
        # 累加预测分数和真实分数
        total_pred_scores.append(pred_score_np)
        total_true_scores.append(true_score_np)

    # print('total_pred_scores', total_pred_scores)
    # print('total_true_scores', total_true_scores)
    # 计算整个轮次的指标
    total_pred_scores = np.concatenate(total_pred_scores)
    total_true_scores = np.concatenate(total_true_scores)
    mse = mean_squared_error(total_true_scores, total_pred_scores)
    srocc, _ = spearmanr(total_true_scores, total_pred_scores)
    wasd = np.mean(np.abs(total_true_scores - total_pred_scores))
    # 输出当前轮验证的WASD、MSE和SROCC
    print(f'WASD: {wasd}, MSE: {mse}, SROCC: {srocc}')

    lcc_mean = pearsonr(total_pred_scores, total_true_scores)
    srcc_mean = spearmanr(total_pred_scores, total_true_scores)
    print('PLCC', lcc_mean[0])
    print('SRCC', srcc_mean[0])

    # 计算准确率
    threshold = 5.0
    total_true_scores_label = np.where(total_true_scores <= threshold, 0, 1)
    total_pred_scores_label = np.where(total_pred_scores <= threshold, 0, 1)
    acc = accuracy_score(total_true_scores_label, total_pred_scores_label)
    print('ACC', acc)

    # return validate_losses_i.avg
    return srcc_mean[0], lcc_mean[0]


def start_train(opt):
    log_name = 'AesCLIP.txt'
    log_txt = os.path.join(opt['experiment_dir_name'], log_name)
    global f
    f = open(log_txt, 'w')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_jsondata_part(opt)
    model = AesCLIP_reg(clip_name='ViT-B/16',
                        weight='pretrained_weights/AesCLIP')
    model.to(device)     #'/home/ubuntu/extend/qq/project/AesCLIP/pretrained_weights/5.21-4-pth-2/AesCLIP_weight--e11-train2.4314-test4.0253_best.pth'
    model.train()
    # TODO: change optimizer
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-3, betas=(0.9, 0.999), lr=opt['init_lr'],
                                  weight_decay=0.01)

    # 使用StepLR学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt['step_size'], gamma=opt['gamma'])

    # criterion_it = InfoNCE(temperature=opt['temperature'])
    criterion_it = torch.nn.MSELoss()  # 使用均方误差作为损失函数
    criterion_it.cuda()
    # criterion_ix.cuda()
    best_test_srcc = 0
    best_test_plcc = 0

    writer = SummaryWriter(log_dir=os.path.join(opt['experiment_dir_name'], 'logs_a'))

    for e in range(opt['num_epoch']):
        # optimizer = adjust_learning_rate(opt, optimizer, e)
        print("*******************************************************************************************************")
        print("第%d个epoch的学习率：%f" % (1 + e, optimizer.param_groups[0]['lr']))
        train_loss = train(opt, epoch=e, model=model, loader=train_loader, optimizer=optimizer,
                           scheduler=scheduler, criterion_it=criterion_it,
                           writer=writer, global_step=len(train_loader) * e,
                           name=f"{opt['experiment_dir_name']}_by_batch")

        test_srcc, test_plcc = validate(opt, model=model, loader=test_loader, optimizer=optimizer,
                             criterion_it=criterion_it,
                             writer=writer, global_step=len(test_loader) * e,
                             name=f"{opt['experiment_dir_name']}_by_batch")
        if test_srcc > best_test_srcc:
            best_test_srcc = test_srcc
            best_test_plcc = test_plcc
            model_name = 'AesCLIP_reg_weight--e{:d}-train{:.4f}-test{:.4f}'.format(e + 1, train_loss, test_srcc)

            # torch.save(model.state_dict(), os.path.join(opt['experiment_dir_name'], model_name + '_best.pth'))

        print('Best Test SRCC:', best_test_srcc)
        print('Best Test PLCC:', best_test_plcc)
        f.write(
            'epoch:%d, train_loss:%.5f,test_srcc:%.5f,test_plcc:%.5f\r\n'
            % (e, train_loss, test_srcc, test_plcc))
        f.write(f'Best Test SRCC: {best_test_srcc}, Best Test PLCC: {best_test_plcc}\r\n')
        f.flush()

        writer.add_scalars("epoch_loss", {'train': train_loss, 'test_srcc': test_srcc},
                           global_step=e)
    writer.close()
    f.close()


if __name__ == "__main__":
    opt = init()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    warnings.filterwarnings('ignore')
    # tuner_params = nni.get_next_parameter()
    # params = vars(merge_parameter(opt, tuner_params))
    params = vars(opt)
    print(params)

    start_train(params)

    # start_train(opt)
    # f.close()