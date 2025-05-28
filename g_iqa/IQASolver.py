import math
import os
import numpy as np
from scipy import stats
from tqdm import tqdm
import gc

from accelerate import Accelerator
from accelerate import utils as autils
from accelerate import DistributedDataParallelKwargs
from safetensors.torch import load_file, load_model

import torch
from torch_ema import ExponentialMovingAverage

from g_iqa.models.iqa_clip import LocalGlobalClipIQA, SimpleClip, SimpleResNet
from g_iqa.g_datasets.data_loader import DataGenerator
from g_iqa.train_utils import loss_by_scene, WarmupCosineLR, plcc_loss

class IQASolver(object):
    """Solver for training and testing IQA"""
    def __init__(self, config, path, train_json, test_json):

        self.epochs = config.epochs
        self.start_epoch = config.start_epoch if "start_epoch" in config else 0
        self.loss_type = config.loss_type
        self.project_dir = config.project_dir
        self.scene_sampling = config.scene_sampling > 0
        self.eval_every = config.eval_every if "eval_every" in config else 1

        # accelerator implementation distributed training
        proj_config = autils.ProjectConfiguration(
            project_dir=config.project_dir,
            automatic_checkpoint_naming=True,
            iteration=math.ceil(self.start_epoch/2),
            total_limit=30,
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(log_with='tensorboard', project_config=proj_config, mixed_precision=config.mixed_precision, kwargs_handlers=[ddp_kwargs])
        self.accelerator.init_trackers('tb')
        self.device = self.accelerator.device
        
        train_loader = DataGenerator(config.train_dataset, path, train_json, config.input_size, batch_size=config.batch_size, istrain=True, scene_sampling=config.scene_sampling)
        self.train_data = train_loader.get_data()
        self.test_data = {td: DataGenerator(td, path, test_json, config.input_size, batch_size=1, istrain=False, testing_aug=True).get_data() for td in config.test_dataset}

        ############### Model ###############
        if config.clip_model == 'resnet50':
            self.model = SimpleResNet(clip_model=config.clip_model, clip_freeze=config.clip_freeze, precision='fp32')
        elif config.local_global:
            # if use our local global complementary token combination
            self.model = LocalGlobalClipIQA(clip_model=config.clip_model, clip_freeze=config.clip_freeze, precision='fp32', all_global=config.all_global)
        else:
            # use simple clip model
            self.model = SimpleClip(clip_model=config.clip_model, clip_freeze=config.clip_freeze, precision='fp32')

        self.model.train(True)
        paras = [{'params': filter(lambda p: p.requires_grad, self.model.clip_model.parameters()), 'lr': config.lr / config.lr_ratio},
                    {'params': self.model.head.parameters(), 'lr': config.lr}]

        # load model from pretrained checkpoint
        if "load_from" in config and config.load_from is not None:
            if config.load_from.endswith('.pth'):
                self.model.load_state_dict(torch.load(config.load_from))
            elif config.load_from.endswith('.safetensors'):
                load_model(self.model, config.load_from)

        self.optimizer = torch.optim.Adam(paras, weight_decay=config.weight_decay)
        self.epoch_step = math.ceil(len(self.train_data))
        self.lr_scheduler = WarmupCosineLR(optimizer=self.optimizer, warmup_step=config.warmup_epoch*self.epoch_step, T_max=config.epochs*self.epoch_step)
        
        self.model, self.optimizer, self.train_data, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_data, self.lr_scheduler
        )

        self.ema_model = ExponentialMovingAverage(self.model.parameters(), decay=config.ema_decay)

        # # resume training from some checkpoint of some epoch
        # if "resume_from" in config and config.resume_from is not None:
        #     self.accelerator.load_state(config.resume_from)
        #     self.ema_model.load_state_dict(torch.load(f'{config.resume_from}/ema_model.pth'))

    def train(self):
        """Training"""
        best_srcc = {data_name: 0 for data_name in self.test_data.keys()}
        best_plcc = {data_name: 0 for data_name in self.test_data.keys()}
        best_epoch = {data_name: 0 for data_name in self.test_data.keys()}
        srcc_by_epoch = {data_name: [] for data_name in self.test_data.keys()}
        plcc_by_epoch = {data_name: [] for data_name in self.test_data.keys()}


        global_step = self.start_epoch * len(self.train_data)
        for t in range(self.start_epoch, self.epochs+1):
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            scene_list = []

            gc.collect()
            torch.cuda.empty_cache()

            for sample in tqdm(self.train_data, desc=f'Epoch {t+1}/{self.epochs}'):
            # it_train_data = iter(self.train_data)
            # for _ in tqdm(range(len(self.train_data)), desc=f'Epoch {t+1}/{self.epochs}'):
            #     try:
            #         sample = next(it_train_data)
            #     except Exception as e:
            #         print(e)
            #         continue
                img = sample['img']
                label = sample['label'].to(self.device).float()
                scene = sample['scene'].to(self.device)

                self.optimizer.zero_grad()

                data, data_pt = img, sample['img_pt']
                pred = self.model(data, data_pt).squeeze(-1)

                # If there is domain sampling, the loss is calculated within the domain, otherwise calculated directly.
                if self.scene_sampling:
                    loss = loss_by_scene(pred, label, scene, self.loss_type)
                elif self.loss_type == 'l1':
                    loss = torch.nn.functional.l1_loss(pred, label)
                elif self.loss_type == 'l2':
                    loss = torch.nn.functional.mse_loss(pred, label)
                elif self.loss_type == 'plcc':
                    loss = plcc_loss(pred, label)
                else:
                    raise NotImplementedError(f'Loss type {self.loss_type} not implemented')
                
                # loss.backward()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                global_step += 1

                self.ema_model.update()

                # log training process with accelerator
                if self.is_main_process():
                    self.accelerator.log({"train/loss": loss}, step=global_step)
                    self.accelerator.log({f"lr/{i}": self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))}, step=global_step)

                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()
                    scene_list = scene_list + scene.cpu().tolist()
                    epoch_loss.append(loss.item())

            if t % self.eval_every == 0:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    print('[INFO] Use Distributed Training, wait for all processes to synchronize...')
                    self.accelerator.wait_for_everyone()
                    torch.distributed.barrier()
                if self.is_main_process():
                    print(f'=========Epoch:{t:3d}=========')
                    _, _ = self.log_metrics(pred_scores, gt_scores, scene_list, t, f"train")

                    with self.ema_model.average_parameters():
                        # os.makedirs(f'{self.project_dir}/ema_ckpts', exist_ok=True)
                        # torch.save(self.ema_model.state_dict(), f'{self.project_dir}/ema_ckpts/model_epoch{t:03}.pth')
                        # os.makedirs(f'{self.project_dir}/ckpts', exist_ok=True)
                        # torch.save(self.accelerator.unwrap_model(self.model).state_dict(), f'{self.project_dir}/ckpts/model_epoch{t:03}.pth')

                        for data_name, test_data in self.test_data.items():
                            pred_scores, gt_scores, scene_list = self.val(test_data)
                            ema_srcc, ema_plcc = self.log_metrics(pred_scores, gt_scores, scene_list, t, f"{data_name}/test")
                            srcc_by_epoch[data_name].append(ema_srcc)
                            plcc_by_epoch[data_name].append(ema_plcc)
                            if ema_srcc > best_srcc[data_name]:
                                best_srcc[data_name] = ema_srcc
                                best_plcc[data_name] = ema_plcc
                                best_epoch[data_name] = t
                                # save pred and gt scores to txt
                                pred_gt = np.stack([np.array(pred_scores), np.array(gt_scores)], axis=1)
                                np.savetxt(f'{self.project_dir}/pred_gt_{data_name}.txt', pred_gt, fmt='%.4f')
                        
                            print(f'Best SRCC: {best_srcc[data_name]:.4f}, Best PLCC: {best_plcc[data_name]:.4f}, Best epoch: {best_epoch[data_name]} \n')
                
                # # fast evaluation
                # with self.ema_model.average_parameters():
                #     rank_idx = torch.distributed.get_rank()
                #     while rank_idx < len(self.test_data):
                #         data_name = sorted(self.test_data.keys())[rank_idx]
                #         test_data = self.test_data[data_name]
                #         rank_idx += torch.distributed.get_world_size()

                #         pred_scores, gt_scores, scene_list = self.val(test_data)
                #         ema_srcc, ema_plcc = self.log_metrics(pred_scores, gt_scores, scene_list, t, f"{data_name}/test")
                #         srcc_by_epoch[data_name].append(ema_srcc)
                #         plcc_by_epoch[data_name].append(ema_plcc)
                #         if ema_srcc > best_srcc[data_name]:
                #             best_srcc[data_name] = ema_srcc
                #             best_plcc[data_name] = ema_plcc
                #             best_epoch[data_name] = t
                #             # save pred and gt scores to txt
                #             pred_gt = np.stack([np.array(pred_scores), np.array(gt_scores)], axis=1)
                #             np.savetxt(f'{self.project_dir}/pred_gt_{data_name}.txt', pred_gt, fmt='%.4f')
                    
                #         print(f'Best SRCC: {best_srcc[data_name]:.4f}, Best PLCC: {best_plcc[data_name]:.4f}, Best epoch: {best_epoch[data_name]} \n')
                
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    # print('[INFO] Use Distributed Training, wait for all processes to synchronize...')
                    self.accelerator.wait_for_everyone()
                    torch.distributed.barrier()

        return best_srcc, best_plcc, srcc_by_epoch, plcc_by_epoch

    @torch.no_grad()
    def val(self, test_data):
        """Testing"""
        self.model.train(False)
        val_model = self.accelerator.unwrap_model(self.model)
        pred_scores = []
        gt_scores = []
        scene_list = []

        for sample in tqdm(test_data):
            # Data.
            img = sample['img'].to(self.device)
            label = sample['label']
            scene = sample['scene']

            data, data_pt = img, sample['img_pt'].to(self.device)
            if len(data.shape) == 5:
                B, T, C, H, W = data.shape
                data = data.view(B*T, C, H, W)
                data_pt = data_pt.view(B*T, C, H, W)
            pred = val_model(data, data_pt)
            if pred.size(0) != label.size(0):
                pred = pred.view(B, T)
                pred = torch.mean(pred, dim=1, keepdim=False)

            pred = pred.squeeze(1) if len(pred.shape) == 2 else pred
            pred_scores = pred_scores + pred.cpu().tolist()
            gt_scores = gt_scores + label.cpu().tolist()
            scene_list = scene_list + scene.cpu().tolist()

        self.model.train(True)
        return pred_scores, gt_scores, scene_list
    
    def is_main_process(self):
        return self.accelerator.is_main_process
    
    def log_metrics(self, pred_scores, gt_scores, scene_list, epoch, prefix="", scene_independent=False):
        srcc, plcc = stats.spearmanr(pred_scores, gt_scores)[0], stats.pearsonr(pred_scores, gt_scores)[0]
        self.accelerator.log({f"{prefix}eval/srcc": srcc, f"{prefix}eval/plcc": plcc}, step=epoch)
        print(f'{prefix} srcc: {srcc:.4f}, plcc: {plcc:.4f}')

        if not scene_independent:
            return abs(srcc), abs(plcc)

        # computer srcc, plcc by scene
        # for scene independent dataset, we should treat each scene independently
        # e.g. pipal, tid2013, piq2023
        scene_dict = {}
        for i, scene in enumerate(scene_list):
            if scene not in scene_dict.keys():
                scene_dict[scene] = dict(
                    pred_scores = [],
                    gt_scores=[],
                )
            scene_dict[scene]['pred_scores'].append(pred_scores[i])
            scene_dict[scene]['gt_scores'].append(gt_scores[i])

        srcc_by_scene = []
        plcc_by_scene = []
        acc_by_scene = []
        for k, scene_item in scene_dict.items():
            scene_srcc, _ = stats.spearmanr(scene_item['pred_scores'], scene_item['gt_scores'])
            scene_plcc, _ = stats.pearsonr(scene_item['pred_scores'], scene_item['gt_scores'])
            scene_acc, _, _ = self.calculate_acc(scene_item['pred_scores'], scene_item['gt_scores'], threshold=150)
            srcc_by_scene.append(scene_srcc)
            plcc_by_scene.append(scene_plcc)
            acc_by_scene.append(scene_acc)
        mean_srcc, std_srcc = np.mean(srcc_by_scene), np.std(srcc_by_scene)
        mean_plcc, std_plcc = np.mean(plcc_by_scene), np.std(plcc_by_scene)
        mean_acc, std_acc = np.mean(acc_by_scene), np.std(acc_by_scene)

        self.accelerator.log({f"{prefix}eval/mean_srcc": mean_srcc}, step=epoch)
        self.accelerator.log({f"{prefix}eval/mean_plcc": mean_plcc}, step=epoch)
        self.accelerator.log({f"{prefix}eval/mean_acc": mean_acc}, step=epoch)

        print(f'{prefix} mean srcc: {mean_srcc:.4f}, std srcc: {std_srcc:.4f}')
        print(f'{prefix} mean plcc: {mean_plcc:.4f}, std plcc: {std_plcc:.4f}')
        print(f'{prefix} mean acc: {mean_acc:.4f}, std acc: {std_acc:.4f}')

        return abs(mean_srcc), abs(mean_plcc)


    def calculate_acc(self, pred_scores, gt_scores, threshold=10000):
        """
        计算阈值过滤后的pair准确率
        
        Args:
            pred_scores: 预测分数列表
            gt_scores: 真实分数列表
            threshold: gt差值阈值
            
        Returns:
            accuracy: 准确率
            total_pairs: 总配对数
            correct_pairs: 正确预测数
        """
        assert len(pred_scores) == len(gt_scores), "预测分数和真实分数长度必须相同"
        
        n = len(pred_scores)
        total_pairs = 0
        correct_pairs = 0
        
        # 构建所有可能的配对
        for i in range(n):
            for j in range(i+1, n):
                # 获取一对图像的预测分数和真实分数
                pred1, pred2 = pred_scores[i], pred_scores[j]
                gt1, gt2 = gt_scores[i], gt_scores[j]
                
                # 计算gt差值
                gt_diff = abs(gt1 - gt2)
                
                # 如果gt差值小于等于阈值，跳过该配对
                if gt_diff > threshold:
                    continue
                    
                # 确定预测分类和真实分类
                pred_class = pred1 > pred2
                gt_class = gt1 > gt2
                
                # 统计正确预测
                total_pairs += 1
                if pred_class == gt_class:
                    correct_pairs += 1
        
        # 计算准确率
        accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
        
        return accuracy, total_pairs, correct_pairs