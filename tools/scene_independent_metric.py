def process_files(pred_gt_file, val_label_file, threshold=10000):
    # 读取文件并解析数据
    pred_gt_data = {}
    with open(pred_gt_file, 'r') as f:
        for line in f:
            # if '|' in line:  # 跳过可能的空行
            parts = line.strip().split(' ')
            pred = float(parts[0])
            gt = float(parts[1])
            pred_gt_data[len(pred_gt_data) + 1] = (pred, gt)
    
    # 从val_label文件读取图像名称
    image_data = {}
    with open(val_label_file, 'r') as f:
        for idx, line in enumerate(f, 1):
            if line.strip():  # 跳过空行
                img_name, _ = line.strip().split(',')
                prefix = img_name.split('_')[0]  # 获取前缀(如A0005)
                if prefix not in image_data:
                    image_data[prefix] = []
                image_data[prefix].append((idx, img_name))

    # 计算准确率
    total_pairs = 0
    correct_pairs = 0

    # 对每个组进行处理
    for prefix, images in image_data.items():
        n = len(images)
        # 构建所有可能的配对
        for i in range(n):
            for j in range(i+1, n):
                idx1, img1 = images[i]
                idx2, img2 = images[j]
                
                pred1, gt1 = pred_gt_data[idx1]
                pred2, gt2 = pred_gt_data[idx2]
                
                # 过滤掉gt差值大于10的pair
                if abs(gt1 - gt2) > threshold:
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


import json
import os
from collections import defaultdict
import numpy as np
from scipy import stats
from tqdm import tqdm
import sys
sys.path.append('/home/dzc/workspace/G-IQA/sota/IQA-PyTorch')
import pyiqa

os.environ['HTTP_PROXY'] = 'http://192.168.195.225:7890'
os.environ['HTTPS_PROXY'] = 'http://192.168.195.225:7890'

# 读取数据
def load_data(json_file, pred_gt_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # 读取预测结果
    pred_gt_pairs = []
    with open(pred_gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            pred = float(parts[0])
            gt = float(parts[1])
            pred_gt_pairs.append((pred, gt))
    
    # 组合数据
    grouped_data = defaultdict(list)
    for (pred, gt), item in zip(pred_gt_pairs, json_data['files']):
        grouped_data[item['domain_id']].append({
            'pred': pred,
            'gt': gt,
            'image': item['image']
        })
    
    return grouped_data

def load_data_pyiqa(json_file, method='niqe'):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    iqa_metric = pyiqa.create_metric(method, device='cuda')
    iqa_metric.eval()

    # 组合数据
    grouped_data = defaultdict(list)
    for item in tqdm(json_data['files']):
        img_path = os.path.join('data', item['image'])
        pred = iqa_metric(img_path).item()
        grouped_data[item['domain_id']].append({
            'pred': pred,
            'gt': item['score'],
            'image': item['image']
        })
    
    return grouped_data

# 在组内构建pairs并评估
def evaluate_pairs(grouped_data, threshold=10):
    total_pairs = 0
    correct_pairs = 0
    
    srcc_list = []
    plcc_list = []
    for domain_id, images in grouped_data.items():
        n = len(images)
        for i in range(n):
            for j in range(i+1, n):
                gt_diff = abs(images[i]['gt'] - images[j]['gt'])
                
                # 过滤掉gt差值大于10的pairs
                if gt_diff > threshold:
                    continue
                    
                total_pairs += 1
                
                # 确定预测分类
                pred_result = images[i]['pred'] > images[j]['pred']
                # 确定真实分类
                gt_result = images[i]['gt'] > images[j]['gt']
                
                # 检查预测是否正确
                if pred_result == gt_result:
                    correct_pairs += 1

        # 计算SRCC和PLCC
        pred_scores = [image['pred'] for image in images]
        gt_scores = [image['gt'] for image in images]
        srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        srcc_list.append(srcc)
        plcc_list.append(plcc)
    
    accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
    return accuracy, total_pairs, correct_pairs, srcc_list, plcc_list

if __name__ == "__main__":
    json_file = 'data_json/for_cross_set/test/pipal_test.json'
    pred_gt_file = 'exp_log/pipal/our_epoch15_bs128_plcc+kld/pred_gt_pipal.txt'
    pred_gt_file = '/home/dzc/workspace/G-IQA/sota/IQA-PyTorch/experiments/hypernet_pipal/val/pred_gt_pipal.txt'
    pred_gt_file = '/home/dzc/workspace/G-IQA/sota/IQA-PyTorch/experiments/clipiqa_pipal/val/pred_gt_pipal.txt'
    pred_gt_file = '/home/dzc/workspace/G-IQA/sota/IQA-PyTorch/experiments/dbcnn_pipal/val/pred_gt_pipal.txt'
    pred_gt_file = '/home/dzc/workspace/G-IQA/sota/MetaIQA/exp_results/cross_set/pipal/pred_gt_pipal.txt'
    threshold = 150

    # json_file = 'data_json/for_cross_set/test/tid2013_test.json'
    # pred_gt_file = 'exp_log/tid2013/our_epoch15_bs128_plcc+kld/pred_gt_tid2013.txt'
    # pred_gt_file = '/home/dzc/workspace/G-IQA/sota/IQA-PyTorch/experiments/hypernet_tid2013/val/pred_gt_tid2013.txt'
    # pred_gt_file = '/home/dzc/workspace/G-IQA/sota/IQA-PyTorch/experiments/clipiqa_tid2013/val/pred_gt_tid2013.txt'
    # pred_gt_file = '/home/dzc/workspace/G-IQA/sota/IQA-PyTorch/experiments/dbcnn_tid2013/val/pred_gt_tid2013.txt'
    # pred_gt_file = '/home/dzc/workspace/G-IQA/sota/MetaIQA/exp_results/cross_set/tid2013/pred_gt_tid2013.txt'
    # threshold = 0.6

    # grouped_data = load_data(json_file, pred_gt_file)


    json_file = 'data_json/for_cross_set/test/pipal_test.json'
    threshold = 150
    json_file = 'data_json/for_cross_set/test/tid2013_test.json'
    threshold = 0.6
    method = 'niqe'
    method = 'ilniqe'
    method = 'brisque'
    grouped_data = load_data_pyiqa(json_file, method=method)


    accuracy, total_pairs, correct_pairs, srcc_list, plcc_list = evaluate_pairs(grouped_data, threshold=threshold)
    
    print(f"总共评估的pairs数量: {total_pairs}")
    print(f"正确预测的pairs数量: {correct_pairs}")
    print(f"准确率: {accuracy:.4f}")

    print(f"SRCC: {np.mean(srcc_list):.4f} ± {np.std(srcc_list):.4f}")
    print(f"PLCC: {np.mean(plcc_list):.4f} ± {np.std(plcc_list):.4f}")


