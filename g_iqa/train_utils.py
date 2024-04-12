import math
import numpy as np
import torch


################## lr scheduler #######################
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    '''
    Warmup + Cosine Anneal

    '''
    def __init__(self, optimizer, warmup_step, T_max, last_epoch=-1, verbose=False):
        self.warmup_step = warmup_step
        self.T_max = T_max
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_step:
            return [base_lr * self.last_epoch / self.warmup_step for base_lr in self.base_lrs]
        else:
            return [0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_step) / self.T_max)) * base_lr for base_lr in self.base_lrs]

# not used
def get_cosine_warmup_scheduler(optimizer:torch.optim, warmup_step:int, T_max:int, lr_max:list[float], lr_min:list[float]=[], warmup_begin_lr:float=0.0):
    '''
    Args:
        optimizer: optimizer of the model
        warmup_step: number of steps for warm up
        T_max: number of steps for cosine anneal, T_max = total_steps - warmup_step, (set to total_steps for default lr_min=0)
        lr_max: maximum learning rate
        lr_min: minimum learning rate
    '''
    if len(lr_max) != 2:
        raise ValueError('lr_max must be a list of 2 elements')
    if len(lr_min) != 0:
        assert len(lr_max) == len(lr_min)
    else:
        lr_min = [0.0] * len(lr_max)

    # Warm up + Cosine Anneal
    cosine_warmup_lambda = []
    cosine_warmup_lambda.append(lambda cur_iter: (lr_max[0] - warmup_begin_lr) * cur_iter / warmup_step if  cur_iter < warmup_step else \
            lr_min[0] + (lr_max[0] - lr_min[0]) * (1 + math.cos(math.pi * (cur_iter - warmup_step) / T_max)) / 2)
    cosine_warmup_lambda.append(lambda cur_iter: (lr_max[1] - warmup_begin_lr) * cur_iter / warmup_step if  cur_iter < warmup_step else \
            lr_min[1] + (lr_max[1] - lr_min[1]) * (1 + math.cos(math.pi * (cur_iter - warmup_step) / T_max)) / 2)

    # python lambda 引用捕获，以下不可行
    # for i, lr_m in enumerate(lr_max):
    #     if len(lr_min) != 0:
    #         assert len(lr_max) == len(lr_min)
    #     else:
    #         lr_min = [0.0] * len(lr_max)
            
    #     cosine_warmup_l = lambda cur_iter: (lr_max[i] - warmup_begin_lr) * cur_iter / warmup_step if  cur_iter < warmup_step else \
    #             lr_min[i] + (lr_max[i] - lr_min[i]) * (1 + math.cos(math.pi * (cur_iter - warmup_step) / T_max)) / 2
    #     cosine_warmup_lambda.append(cosine_warmup_l)
    
    # plot the learning rate schedule
    import matplotlib.pyplot as plt
    x = list(range(0, int(T_max)))
    y = [cosine_warmup_lambda[0](i) for i in x]
    plt.plot(x, y)
    plt.savefig('lr_schedule.png')
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup_lambda)
    return scheduler


################## losses scheduler #######################
def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / scale
    ).float()


def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    # loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    # rho = torch.mean(y_pred * y)
    # loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    # return ((loss0 + loss1) / 2).float()
    return torch.nn.functional.l1_loss(y_pred, y).float()

def scale_shift_loss(y_pred, y):
    # linear fitting y = a*y_pred + b
    a, b = np.polyfit(y_pred.tolist(), y.tolist(), 1)
    if a < 0:
        # return torch.nn.functional.l1_loss(y_pred, y).float()
        # return rank_loss(y_pred, y)
        return plcc_loss(y_pred, y)
    y_pred = a * y_pred + b
    return torch.nn.functional.l1_loss(y_pred, y).float()

def robust_scale_shift_loss(y_pred, y):
    t_pred = torch.median(y_pred)
    t = torch.median(y)
    s_pred = torch.mean(torch.abs(y_pred - t_pred))
    s = torch.mean(torch.abs(y - t))
    y_pred = (y_pred - t_pred) / (s_pred + 1e-8)
    y = (y - t) / (s + 1e-8)
    return torch.nn.functional.l1_loss(y_pred, y).float()

def loss_by_scene(y_pred, y, scene, loss_type):
    scene_id = torch.unique(scene)
    loss = 0
    for s in scene_id:
        idx = scene == s
        y_pred_s = y_pred[idx]
        y_s = y[idx]
        if loss_type == 'l1':
            loss += torch.nn.functional.l1_loss(y_pred_s, y_s)
        elif loss_type == 'huber':
            loss += torch.nn.functional.smooth_l1_loss(y_pred_s, y_s)
        elif loss_type == 'plcc':
            loss += plcc_loss(y_pred_s, y_s)
        elif loss_type == 'rank':
            loss += rank_loss(y_pred_s, y_s)
        elif loss_type == 'plcc+rank':
            loss += plcc_loss(y_pred_s, y_s) + rank_loss(y_pred_s, y_s)
        elif loss_type == 'scale_shift':
            # loss += scale_shift_loss(y_pred_s, y_s)
            loss += robust_scale_shift_loss(y_pred_s, y_s)
    
    return loss / len(scene_id)
