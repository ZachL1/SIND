import os
from tqdm import tqdm
import json
import argparse
import blobfile as bf

import torch
import numpy as np
import pandas as pd
from scipy import stats

from g_iqa.g_datasets.data_loader import DataGenerator
from g_iqa.models.iqa_clip import LocalGlobalClipIQA

def _list_image_files_recursively(data_dir):
  results = []
  for entry in sorted(bf.listdir(data_dir)):
    full_path = bf.join(data_dir, entry)
    ext = entry.split(".")[-1]
    if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "tif", "tiff"]:
      results.append(full_path)
    elif bf.isdir(full_path):
      results.extend(_list_image_files_recursively(full_path))
  return results

def load_sind(path, device):
  clip_model = 'openai/ViT-B-16'
  model = LocalGlobalClipIQA(clip_model, clip_freeze=False, precision='fp32')
  model.load_state_dict(torch.load(path, map_location='cpu'))
  model.to(device)
  model.eval()
  return model

def load_json_data(data_dir, batch_size, json_file, input_size):
  """
  Create a data loader from a json file
  Args:
    data_dir: the directory that contains images, os.path.join(data_dir, "path/to/image1.jpg") should be the path of the image
    json_file: the json file of the data, the json file should be like this:
      {
        "files": [
          {"image": "path/to/image1.jpg", "score": 1},
          {"image": "path/to/image2.jpg", "score": 0}
        ]
      }
  """
  with open(json_file, 'r') as f:
    data = json.load(f)['files']
  for item in data:
    if 'score' not in item.keys():
      item['score'] = 0 # score is not necessary when testing, set it to 0 for convenience
    if 'domain_id' not in item.keys():
      item['domain_id'] = 0 # domain_id is not necessary when testing, set it to 0 for convenience

  data_loader = DataGenerator(dataset='common_json', path=data_dir, data_json=data, input_size=input_size, batch_size=batch_size, istrain=False, testing_aug=True, local=True).get_data()
  return data_loader

def load_directory_data(data_dir, batch_size, input_size):
  """
  Create a data loader from a directory, containing images as:
    data_dir/image1.jpg
    data_dir/image2.jpg
    ...
  """
  # create a json dict
  data = []
  # for file in os.listdir(data_dir):
  for file in _list_image_files_recursively(data_dir)[:650000]:
    data.append({"image": file, "score": 0, "domain_id": 0})
  
  data_loader = DataGenerator(dataset='common_json', path='.', data_json=data, input_size=input_size, batch_size=batch_size, istrain=False, testing_aug=True, local=True).get_data()
  return data_loader

@torch.no_grad()
def do_inference(model, data_loader, device):
  pred_scores = []
  gt_scores = []
  scene_list = []
  img_names = []

  for sample in tqdm(data_loader):
    # Data.
    data, data_pt = sample['img'].to(device), sample['img_pt'].to(device)
    label = sample['label'] # default is 0 if not provided
    scene = sample['scene'] # default is 0 if not provided
    img_name = sample['img_name']

    # Inference.
    if len(data.shape) == 5:
        B, T, C, H, W = data.shape
        data = data.view(B*T, C, H, W)
        data_pt = data_pt.view(B*T, C, H, W)
    pred = model(data, data_pt)
    if pred.size(0) != label.size(0):
        pred = pred.view(B, T)
        pred = torch.mean(pred, dim=1, keepdim=True)

    # Save.
    pred_scores = pred_scores + pred.squeeze(-1).tolist()
    gt_scores = gt_scores + label.tolist()
    scene_list = scene_list + scene.tolist()
    img_names = img_names + img_name

  return pred_scores, gt_scores, scene_list, img_names


if __name__ == '__main__':
  # config
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='', help='the directory that contains images')
  parser.add_argument('--json_file', type=str, default=None, help='the json file of well organized image paths')
  parser.add_argument('--input_size', type=int, default=224, help='input size of the model, should be 224 consistent with the training')
  parser.add_argument('--model_path', type=str, default='', help='the trained checkpoint path')
  parser.add_argument('--save_dir', type=str, default=None, help='the directory to save the results')
  parser.add_argument('--batch_size', type=int, default=8, help='batch size for inference')

  args = parser.parse_args()
  
  # load data
  if args.json_file is not None:
    test_data = load_json_data(args.data_dir, args.batch_size, args.json_file, args.input_size)
  else:
    test_data = load_directory_data(args.data_dir, args.batch_size, args.input_size)

  # load model
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = load_sind(args.model_path, device)

  # do test
  pred_scores, gt_scores, _, img_names = do_inference(model, test_data, device)

  # compute srcc and plcc if gt_scores are provided
  if sum(gt_scores) > 0:
    srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    plcc, _ = stats.pearsonr(pred_scores, gt_scores)
    print(f'SRCC: {srcc}, PLCC: {plcc}')
  
  # save results
  if args.save_dir is not None:
    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.DataFrame({'img_name': img_names, 'gt_score': gt_scores, 'pred_score': pred_scores})
    df.to_csv(os.path.join(args.save_dir, 'imagenet_results_0_650000.csv'), index=False)

  # with open('data/HUAWEI_test/results.json', 'r') as f:
  #   results = json.load(f)
  # for img_name, pred_score in zip(img_names, pred_scores):
  #   results[img_name.split('/HUAWEI_test/')[-1]]['SIND_score'] = pred_score
  # with open('data/HUAWEI_test/results.json', 'w') as f:
  #   json.dump(results, f, indent=2)