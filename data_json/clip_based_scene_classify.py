import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


os.environ["HTTP_PROXY"] = "http://192.168.195.225:7890"
os.environ["HTTPS_PROXY"] = "http://192.168.195.225:7890"

# 加载预训练的CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# 提取图像特征向量的函数
def extract_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

# 为每张图片分配标签的函数
def classify_image(image_path, labels, text_features):
    image_features = extract_features(image_path)
    
    # 计算相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # 找到最相似的标签
    best_label_index = similarity.argmax().item()
    return labels[best_label_index]

# 处理整个数据集的函数
def classify_images(image_items, data_dir='./data'):
    # 候选场景标签
    # labels = ["Animal", "Cityscape", "Human", "Indoor scene", "Landscape", "Night scene", "Plant", "Still-life", "Others"]
    labels = ["Animal", "Cityscape", "Human", "Indoor scene", "Landscape", "Night scene", "Plant", "Still-life"]

    # 将标签转换为CLIP特征向量
    text_inputs = torch.cat([clip.tokenize(f"a photo of {label}") for label in labels]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # 为每张图片分配标签
    for item in tqdm(image_items):
        if 'scene' in item and len(item['scene']) == 1 and "Others" in item['scene']:
            item['clip_scene'] = "Others"
            continue
        
        image_path = os.path.join(data_dir, item['image'])
        label = classify_image(image_path, labels, text_features)
        item['scene'] = label
        item['domain_id'] += labels.index(label)
    return image_items


def cluster_images(image_items, data_dir='./data', n_clusters=9, domain_id_base=100):
    features = []
    # 提取所有图像的特征向量
    for item in tqdm(image_items):
        image_path = os.path.join(data_dir, item['image'])
        image_feature = extract_features(image_path)

        # 对特征向量进行归一化，kmeans聚类时最小化欧氏距离即等效于最大化余弦相似度
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        
        features.append(image_feature.cpu().numpy().flatten())
    
    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=2024)
    labels = kmeans.fit_predict(features)

    # 为每张图片分配标签
    for i, item in enumerate(image_items):
        item['clip_domain_id'] = int(domain_id_base + labels[i])

    return image_items




with open("data_json/all/koniq10k_all.json", "r") as f:
    data = json.load(f)

updated_data_json = classify_images(data['files'])
# updated_data_json = cluster_images(data['files'])

with open("data_json/all/koniq10k_all.json", "w") as f:
    json.dump(data, f, indent=2)



# def merge_clip_json(target_json, clip_json):
#     with open(target_json, "r") as f:
#         target_data = json.load(f)
#     with open(clip_json, "r") as f:
#         clip_data = json.load(f)
    
#     for target_item, clip_item in zip(target_data['files'], clip_data):
#         assert target_item['image'] == clip_item['image']
#         target_item['clip_domain_id'] = clip_item['clip_domain_id']
    
#     with open(target_json, "w") as f:
#         json.dump(target_data, f, indent=2)

# merge_clip_json("data_json/for_leave_one_out/spaq_all.json", "data_json/for_leave_one_out/spaq_all_clip.json")