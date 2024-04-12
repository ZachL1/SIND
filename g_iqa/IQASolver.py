import math
import numpy as np
from scipy import stats
from tqdm import tqdm

from accelerate import Accelerator
from accelerate import utils as autils
from accelerate import DistributedDataParallelKwargs
from safetensors.torch import load_file, load_model

import torch
from torch_ema import ExponentialMovingAverage

from models.iqa_clip import LocalGlobalClipIQA
from g_datasets.data_loader import DataGenerator
from train_utils import loss_by_scene, WarmupCosineLR

class IQASolver(object):
    """Solver for training and testing IQA"""
    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.start_epoch = config.start_epoch if "start_epoch" in config else 0
        self.loss_type = config.loss_type
        self.project_dir = config.project_dir

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
        
        train_path = [path[i] for i in config.train_data]
        val_path = [path[i] for i in config.val_data]
        train_loader = DataGenerator(config.dataset, train_path, train_idx, config.input_size, batch_size=config.batch_size, istrain=True, scene_sampling=config.scene_sampling)
        test_loader = DataGenerator(config.dataset, val_path, test_idx, config.input_size, batch_size=1, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        ############### Model ###############
        self.model = LocalGlobalClipIQA(clip_model=config.clip_model, clip_freeze=config.clip_freeze, precision='fp32', all_global=config.all_global)
        # self.model.load_state_dict(torch.load('/home/dzc/workspace/ntire/contrast_weight'), strict=False)
        self.model.train(True)
        paras = [{'params': filter(lambda p: p.requires_grad, self.model.clip_model.parameters()), 'lr': config.lr / config.lr_ratio},
                    {'params': self.model.head.parameters(), 'lr': config.lr}]

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

        if "resume_from" in config and config.resume_from is not None:
            self.accelerator.load_state(config.resume_from)
            self.ema_model.load_state_dict(torch.load(f'{config.resume_from}/ema_model.pth'))

            # temp_model = LocalGlobalClipIQA(clip_model=config.clip_model, clip_freeze=config.clip_freeze, precision='fp32')
            # temp_model.load_state_dict(torch.load(f'{config.resume_from}/ema_model.pth'))
            # ema_state = dict()
            # ema_state['shadow_params'] = list(temp_model.parameters())
            # ema_state['decay'] = config.ema_decay
            # ema_state['num_updates'] = len(self.train_data) * config.start_epoch
            # ema_state['collected_params'] = None
            # self.ema_model.load_state_dict(ema_state)
            # del temp_model

        # self.ema_model = ExponentialMovingAverage(self.model.parameters(), decay=config.ema_decay)

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_median_srcc = 0.0
        global_step = self.start_epoch * len(self.train_data)
        for t in range(self.start_epoch, self.epochs+1):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            torch.cuda.empty_cache()
            for sample in tqdm(self.train_data, desc=f'Epoch {t+1}/{self.epochs}'):
                img = sample['img']
                label = sample['label'].to(self.device)
                scene = sample['scene'].to(self.device)

                self.optimizer.zero_grad()

                data, data_pt = img, sample['img_pt']
                # with torch.cuda.amp.autocast():
                pred = self.model(data, data_pt)

                loss = loss_by_scene(pred.squeeze(), label.float(), scene, self.loss_type)
                
                # loss.backward()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                global_step += 1

                self.ema_model.update()

                # log with accelerator
                if self.is_main_process():
                    self.accelerator.log({"train/loss": loss}, step=global_step)
                    self.accelerator.log({f"lr/{i}": self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))}, step=global_step)

                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()
                    epoch_loss.append(loss.item())

            if t % 2 == 0:
                # self.accelerator.wait_for_everyone()
                if self.is_main_process():
                    with self.ema_model.average_parameters():
                        # os.makedirs(f'{self.project_dir}/ema_ckpts', exist_ok=True)
                        # torch.save(self.ema_model.state_dict(), f'{self.project_dir}/ema_ckpts/model_epoch{t:03}.pth')
                        
                        _, _ = self.ema_val(t)
                    
                    test_srcc, test_median_srcc = self.val(t)
                    train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
                    if test_srcc > best_srcc:
                        best_srcc = test_srcc
                        best_median_srcc = test_median_srcc

                    print(f'Epoch {t+1}/{self.epochs}, Train_Loss: {sum(epoch_loss) / len(epoch_loss):.3f}')
                    print(f'Train_SRCC: {train_srcc:.4f}, Test_SRCC: {test_srcc:.4f}, Test_Median_SRCC: {test_median_srcc:.4f}, Best_SRCC: {best_srcc:.4f}, Best_Median_SRCC: {best_median_srcc:.4f}')

                # self.accelerator.save_state()

        print('Best test mean-SRCC %f, median-SRCC %f' % (best_srcc, best_median_srcc))

        return best_srcc, best_median_srcc

    def ema_val(self, epoch):
        """Testing"""
        self.model.train(False)
        val_model = self.accelerator.unwrap_model(self.model)
        pred_scores = []
        gt_scores = []
        scene_list = []

        for sample in tqdm(self.test_data):
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
                pred = torch.mean(pred, dim=1, keepdim=True)

            pred_scores = pred_scores + pred.squeeze(-1).tolist()
            gt_scores = gt_scores + label.tolist()
            scene_list = scene_list + scene.tolist()

        # pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        # gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        # computer srcc by scene
        # scene_list = np.mean(np.reshape(np.array(scene_list), (-1, self.test_patch_num)), axis=1, dtype=np.int32).tolist()
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
        for k, scene_item in scene_dict.items():
            scene_srcc, _ = stats.spearmanr(scene_item['pred_scores'], scene_item['gt_scores'])
            srcc_by_scene.append(scene_srcc)
        mean_srcc, med_srcc = np.mean(srcc_by_scene), np.median(srcc_by_scene)

        # print(f'mean srcc: {mean_srcc}, median srcc: {med_srcc}')
        self.accelerator.log({"ema_eval/mean_srcc": mean_srcc, "ema_eval/median_srcc": med_srcc}, step=epoch)

        self.model.train(True)
        return mean_srcc, med_srcc

    def val(self, epoch):
        """Testing"""
        self.model.train(False)
        val_model = self.accelerator.unwrap_model(self.model)
        pred_scores = []
        gt_scores = []
        scene_list = []

        for sample in tqdm(self.test_data):
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
                pred = torch.mean(pred, dim=1, keepdim=True)

            pred_scores = pred_scores + pred.squeeze(-1).tolist()
            gt_scores = gt_scores + label.tolist()
            scene_list = scene_list + scene.tolist()

        # computer srcc by scene
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
        for k, scene_item in scene_dict.items():
            scene_srcc, _ = stats.spearmanr(scene_item['pred_scores'], scene_item['gt_scores'])
            srcc_by_scene.append(scene_srcc)
        mean_srcc, med_srcc = np.mean(srcc_by_scene), np.median(srcc_by_scene)

        # print(f'mean srcc: {mean_srcc}, median srcc: {med_srcc}')
        self.accelerator.log({"eval/mean_srcc": mean_srcc, "eval/median_srcc": med_srcc}, step=epoch)

        self.model.train(True)
        return mean_srcc, med_srcc
    
    def is_main_process(self):
        return self.accelerator.is_main_process