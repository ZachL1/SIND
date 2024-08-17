import torch
import torchvision
import torch.nn as nn
import open_clip

def load_clip_model(clip_model="openai/ViT-B-16", clip_freeze=True, precision='fp16'):
    pretrained, model_tag = clip_model.split('/')
    clip_model = open_clip.create_model(model_tag, precision=precision, pretrained=pretrained)
    if clip_freeze:
        for param in clip_model.parameters():
            param.requires_grad = False

    if model_tag == 'ViT-B-16':
        feature_size = dict(global_feature=512, local_feature=[196, 768])
    elif model_tag == 'ViT-L-14-quickgelu' or model_tag == 'ViT-L-14':
        feature_size = dict(global_feature=768, local_feature=[256, 1024])
    else:
        raise ValueError(f"Unknown model_tag: {model_tag}")

    return clip_model, feature_size


class QualityFusionHead(nn.Module):
    def __init__(self, global_feature_size, local_feature_size, output_size=1):
        super(QualityFusionHead, self).__init__()
        self.global_feature_size = global_feature_size
        self.local_feature_size = local_feature_size
        self.output_size = output_size

        # crop_patch = 7 * 7
        # self.local_proj = nn.Sequential(
        #     nn.Conv1d(local_feature_size[0], crop_patch, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(crop_patch, 1, 1),
        #     nn.ReLU(),
        #     nn.Linear(local_feature_size[1], global_feature_size),
        #     nn.ReLU(),
        # )

        self.quality_predictor = nn.Sequential(
            # nn.TransformerEncoderLayer(d_model=1024, nhead=8),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, output_size),
            # nn.Linear(1024, output_size),
            nn.Linear(global_feature_size + local_feature_size[1], output_size),
        )

    def forward(self, global_features, local_features):
        # local_features = self.local_proj(local_features).squeeze(1)
        local_features = torch.mean(local_features, dim=1)

        features = torch.cat([global_features, local_features], dim=1)
        quality = self.quality_predictor(features)

        return quality


class LocalGlobalClipIQA(nn.Module):
    def __init__(self, clip_model="openai/ViT-B-16", clip_freeze=True, precision='fp16', all_global=False):
        super(LocalGlobalClipIQA, self).__init__()
        self.clip_freeze = clip_freeze
        self.all_global = all_global


        self.clip_model, feature_size = load_clip_model(clip_model, clip_freeze, precision)
        if self.all_global:
            feature_size['local_feature'] = [1, feature_size['global_feature']]
        self.head = QualityFusionHead(feature_size['global_feature'], feature_size['local_feature'])

    def forward(self, x_global, x_local):
        global_features, _ = self.clip_model.encode_image(x_global) # B, 512
        if self.all_global:
            local_features, _ = self.clip_model.encode_image(x_local) # B, 512
            local_features = local_features.unsqueeze(1) # B, 1, 512
        else:
            _, local_features = self.clip_model.encode_image(x_local) # B, 196, 768

        quality = self.head(global_features, local_features)

        local_features = torch.mean(local_features, dim=1)
        features = torch.cat([global_features, local_features], dim=1)
        # return quality, features
        return quality

class SimpleClip(nn.Module):
    def __init__(self, clip_model="openai/ViT-B-16", clip_freeze=True, precision='fp16'):
        super(SimpleClip, self).__init__()
        self.clip_freeze = clip_freeze

        self.clip_model, feature_size = load_clip_model(clip_model, clip_freeze, precision)
        self.head = nn.Linear(feature_size['global_feature'], 1)

    def forward(self, x, x_local):
        features, _ = self.clip_model.encode_image(x)
        quality = self.head(features)

        # return quality, features
        return quality

class SimpleResNet(nn.Module):
    def __init__(self, clip_model="resnet50", clip_freeze=True, precision='fp16'):
        super(SimpleResNet, self).__init__()

        resnet_model = torchvision.models.resnet50(pretrained=True)
        fc_in_features = resnet_model.fc.in_features

        self.clip_model = nn.Sequential(*list(resnet_model.children())[:-1])
        self.head = nn.Linear(fc_in_features, 1)
    
    def forward(self, x, x_local):
        features = self.clip_model(x).squeeze()
        quality = self.head(features)

        return quality