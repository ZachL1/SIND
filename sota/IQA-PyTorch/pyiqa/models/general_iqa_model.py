import torch
import os
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
from pyiqa.archs import build_network
from pyiqa.losses import build_loss
from pyiqa.metrics import calculate_metric
from pyiqa.utils import get_root_logger, imwrite, tensor2img
from pyiqa.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class GeneralIQAModel(BaseModel):
    """General module to train an IQA network."""

    def __init__(self, opt):
        super(GeneralIQAModel, self).__init__(opt)

        # define network
        self.net = build_network(opt['network'])
        self.net = self.model_to_device(self.net)
        self.print_network(self.net)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net, load_path, self.opt['path'].get('strict_load', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net.train()
        train_opt = self.opt['train']

        self.net_best = build_network(self.opt['network']).to(self.device)

        # define losses
        if train_opt.get('mos_loss_opt'):
            self.cri_mos = build_loss(train_opt['mos_loss_opt']).to(self.device)
        else:
            self.cri_mos = None

        # define metric related loss, such as plcc loss
        if train_opt.get('metric_loss_opt'):
            self.cri_metric = build_loss(train_opt['metric_loss_opt']).to(self.device)
        else:
            self.cri_metric = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_opt = train_opt['optim']

        param_dict = {k: v for k, v in self.net.named_parameters()}
        param_keys = list(param_dict.keys())
        # set different lr for different modules if needed, e.g., lr_backbone, lr_head
        lr_keys = [i for i in optim_opt.keys() if i.startswith('lr_')]

        optim_params = []
        for key in lr_keys:
            if key.startswith('lr_'):
                module_key = key.replace('lr_', '')
                logger = get_root_logger()
                logger.info(f'Set optimizer for {module_key} with lr: {optim_opt[key]}, weight_decay: {optim_opt.get(f"weight_decay_{module_key}", 0.)}')

                optim_params.append({
                    'params': [param_dict[k] for k in param_keys if module_key in k and param_dict[k].requires_grad],
                    'lr': optim_opt.pop(key, 0.),
                    'weight_decay': optim_opt.pop(f'weight_decay_{module_key}', 0.),
                })

                # should use param_keys[:] to avoid iteration error
                for k in param_keys[:]:
                    if module_key in k:
                        param_keys.remove(k)
        
        # append the rest of the parameters
        optim_params.append({
            'params': [param_dict[k] for k in param_keys if param_dict[k].requires_grad],
        })

        # log params that will not be optimized
        for k, v in param_dict.items():
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        
        # remove blank param list
        for k in optim_params:
            if len(k['params']) == 0:
                optim_params.remove(k)

        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt['optim'])
        self.optimizers.append(self.optimizer)
    
    def feed_data(self, data):
        self.img_input = data['img'].to(self.device)

        if 'mos_label' in data:
            self.gt_mos = data['mos_label'].to(self.device)

        if 'ref_img' in data:
            self.use_ref = True
            self.ref_input = data['ref_img'].to(self.device)
        else:
            self.use_ref = False

        if 'use_ref' in self.opt['train']:
            self.use_ref = self.opt['train']['use_ref']

    def net_forward(self, net):
        if self.use_ref:
            return net(self.img_input, self.ref_input)
        else:
            return net(self.img_input)

    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()
        self.output_score = self.net_forward(self.net)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_mos:
            l_mos = self.cri_mos(self.output_score, self.gt_mos)
            l_total += l_mos
            loss_dict['l_mos'] = l_mos

        if self.cri_metric:
            l_metric = self.cri_metric(self.output_score, self.gt_mos)
            l_total += l_metric
            loss_dict['l_metric'] = l_metric

        l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # log metrics in training batch
        pred_score = self.output_score.squeeze(1).cpu().detach().numpy()
        gt_mos = self.gt_mos.squeeze(1).cpu().detach().numpy()
        for name, opt_ in self.opt['val']['metrics'].items():
            self.log_dict[f'train_metrics/{name}'] = calculate_metric([pred_score, gt_mos], opt_)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.output_score = self.net_forward(self.net)
        self.net.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        pred_score = []
        gt_mos = []
        for idx, val_data in enumerate(dataloader):
            img_name = osp.basename(val_data['img_path'][0])
            self.feed_data(val_data)
            self.test()
            pred_score.append(self.output_score)
            gt_mos.append(self.gt_mos)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name:>20}')
        if use_pbar:
            pbar.close()

        pred_score = torch.cat(pred_score, dim=0).squeeze(1).cpu().numpy()
        gt_mos = torch.cat(gt_mos, dim=0).squeeze(1).cpu().numpy()

        if with_metrics:
            # calculate all metrics
            for name, opt_ in self.opt['val']['metrics'].items():
                self.metric_results[name] = calculate_metric([pred_score, gt_mos], opt_)

            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    pred_gt = np.stack([pred_score, gt_mos], axis=1)
                    os.makedirs(f'{self.opt["path"]["experiments_root"]}/val', exist_ok=True)
                    np.savetxt(f'{self.opt["path"]["experiments_root"]}/val/pred_gt_{dataset_name}.txt', pred_gt, fmt='%.4f')
                    self.copy_model(self.net, self.net_best)
                    # self.save_network(self.net_best, 'net_best')
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net, self.net_best)
                    # self.save_network(self.net_best, 'net_best')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'val_metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter, save_net_label='net'):
        self.save_network(self.net, save_net_label, current_iter)
        self.save_training_state(epoch, current_iter)
