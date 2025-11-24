# SLIP encoder
import torch
import torch.nn as nn
import torchvision.models as models

class SLIPVideoEncoder(nn.Module):
    def __init__(self, pretrained=True, embed_dim=512):
        super(SLIPVideoEncoder, self).__init__()
        
        # ResNet50 Backbone
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features 
        
        # 차원 축소 (2048 -> 512)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x: (Batch, Channels, Frames, Height, Width)
        b, c, f, h, w = x.shape
        
        # CNN 처리를 위해 (Batch * Frames)로 펼침
        x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        
        # 특징 추출 (B*F, 2048)
        features = self.backbone(x).flatten(1) 
        
        # 차원 축소 (B*F, 512)
        features = self.projection(features)
        
        # 다시 비디오 형태인 (Batch, Frames, Dim)으로 복구
        # Temporal Pooling을 하지 않고 그대로 반환합니다!
        out = features.view(b, f, -1) 
        
        return out