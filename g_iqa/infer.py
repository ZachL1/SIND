import os
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from scipy import stats

import PIQ2023.src.datasets.data_loader as data_loader
from models.iqa_clip import LocalGlobalClipIQA
from safetensors.torch import load_file, load_model

from torch_ema import ExponentialMovingAverage
import gzip

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

def computer_srcc_by_scene(scene_list, pred_scores, gt_scores):
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

    print(f'mean srcc: {mean_srcc}, median srcc: {med_srcc}')

if __name__ == '__main__':
  # config
  patch_size = 224
  test_patch_num = 1
  clip_model = 'openai/ViT-B-16'
#   model_path = '/home/dzc/workspace/ntire/PIQ2023/output/clip_base_epoch70_bs128_exdata_pretrain/train_time_0/checkpoints/checkpoint_21/model.safetensors'
#   model_path = '/home/dzc/workspace/ntire/PIQ2023/output/clip_base_epoch70_bs128_exdata_pretrain_wd5/train_time_0/checkpoints/checkpoint_17/model.safetensors'
#   model_path = '/home/dzc/workspace/ntire/PIQ2023/output/clip_base_epoch35_bs128_finetune_from70-42/train_time_0/checkpoints/checkpoint_0/model.safetensors'
#   model_path = '/home/dzc/workspace/ntire/PIQ2023/good_model/train60_50_wd5.safetensors'
  
  # model_path = '/home/dzc/workspace/ntire/model_temp/finetune35_2_from_70_34_833.pth'
  model_path = '/home/dzc/workspace/ntire/PIQ2023/output/clip_base_epoch35_bs128_finetune_from70-34_ema_test2/train_time_0/ema_ckpts/model_epoch012.pth'
  device = 'cuda:1'

  save_test_to = '/home/dzc/workspace/ntire/model_temp/finetune35_12_from_70_34_m2.csv'
  save_model_to = '/home/dzc/workspace/ntire/PIQ2023/NTIRE24/Submission Kit/weights/finetune35_12_from_70_34_m2.pth'

  piq_data_dir = '/home/dzc/workspace/ntire/data/PIQ23/'
  train_index, test_index = get_piq23_train_test(piq_data_dir)
  test_data = data_loader.DataLoader('piq23', piq_data_dir, test_index, patch_size, test_patch_num, istrain=False).get_data()

  model = LocalGlobalClipIQA(clip_model, clip_freeze=False, precision='fp32')
  model.to(device)
  model.eval()
#   load_model(model, model_path)
  

  def test(val_model):
    pred_scores = []
    gt_scores = []
    scene_list = []

    for sample in tqdm(test_data):
      # Data.
      img = sample['img'].to(device)
      label = sample['label']
      scene = sample['scene']

      data, data_pt = img, sample['img_pt'].to(device)
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
    

    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, test_patch_num)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, test_patch_num)), axis=1)
    scene_list = np.mean(np.reshape(np.array(scene_list), (-1, test_patch_num)), axis=1, dtype=np.int32).tolist()

    computer_srcc_by_scene(scene_list, pred_scores, gt_scores)

    # save pred_scores, gt_scores, scene_list to csv
    # Prediction,Class
    with open(save_test_to, 'w') as f:
      f.write('Prediction,Class,GT\n')
      for i in range(len(pred_scores)):
        f.write(f'{pred_scores[i]},{scene_list[i]},{gt_scores[i]}\n')


  # with gzip.open(save_model_to, 'wb') as f:
  with open(save_model_to, 'wb') as f:
    model_state = torch.load(model_path, map_location=device)
    if 'decay' in model_state.keys():
      ema_model = ExponentialMovingAverage(model.parameters(), decay=model_state['decay'])
      ema_model.load_state_dict(model_state)
      with ema_model.average_parameters():
        torch.save(model.state_dict(), f)
        test(model)
    else: 
      model.load_state_dict(model_state, strict=False)
      torch.save(model.state_dict(), f)
      test(model)

  # save model to .pth
#   torch.save(model.state_dict(), '/home/dzc/workspace/ntire/PIQ2023/NTIRE24/Submission Kit/weights/finetune35_0_from70_42.pth')
