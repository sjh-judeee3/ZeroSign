import torch
import torch.nn as nn
import timm

class SLIPVisualEncoder(nn.Module):
    """
    SLIP(ViT-B/16) 기반의 Visual Encoder입니다.
    비디오 프레임 시퀀스를 입력받아 프레임별 특징 벡터(Feature Vector)를 추출합니다.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        print(f"🔄 Loading SLIP Backbone ({model_name})...")
        
        # timm을 사용하여 ViT 모델 로드 (SLIP/CLIP은 주로 ViT-B/16 사용)
        # num_classes=0으로 설정하여 분류 헤드(Classifier)를 제외하고 특징만 뽑습니다.
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # ViT-B의 출력 차원 (보통 768)
        self.output_dim = self.backbone.num_features
        print(f"✅ Encoder Loaded! Output Dim: {self.output_dim}")

    def forward(self, x):
        """
        Args:
            x: [Batch, Channel, Frames, Height, Width] 형태의 5D 텐서
               (예: [4, 3, 16, 224, 224])
        Returns:
            features: [Batch, Frames, Output_Dim]
               (예: [4, 16, 768])
        """
        # 입력 차원 확인
        if x.dim() == 4: # [C, T, H, W] -> 배치 차원 추가
            x = x.unsqueeze(0)
            
        b, c, t, h, w = x.shape
        
        # CNN/ViT는 보통 이미지(4D)를 처리하므로, 배치와 프레임을 합칩니다.
        # [Batch, C, T, H, W] -> [Batch, T, C, H, W] -> [Batch * T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, c, h, w)
        
        # Backbone 통과 (이미지 인코딩)
        features = self.backbone(x) # 결과: [Batch * T, Feature_Dim]
        
        # 다시 비디오 시퀀스 형태로 복원
        # [Batch * T, Dim] -> [Batch, T, Dim]
        features = features.view(b, t, -1)
        
        return features

if __name__ == "__main__":
    # 테스트 코드
    model = SLIPVisualEncoder()
    dummy_video = torch.randn(2, 3, 16, 224, 224) # [B, C, T, H, W]
    output = model(dummy_video)
    print(f"Input shape: {dummy_video.shape}")
    print(f"Output shape: {output.shape}") # [2, 16, 768] 이어야 함