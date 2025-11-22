## SLIP.py 끝나면 DummyResNetEncoder 대체하기

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models_vision

# 우리가 만든 모듈들
from src.dataset import SignLanguageDataset
from src.models import HybridTemporalModel, ProtoNetClassifier


# [1] 임시 인코더

class DummyResNetEncoder(nn.Module):
    """
    SLIP 대신 이미지를 받아서 512차원 특징을 뽑아주는 임시 모델\
    """
    def __init__(self):
        super().__init__()
        # 가벼운 ResNet18 사용
        resnet = models_vision.resnet18(pretrained=True)
        # 마지막 분류기(FC)를 떼어내고 특징만 뽑도록 수정
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 512) # 차원 맞추기용

    def forward(self, x):
        # x: [Batch, Frames, 3, 224, 224]
        batch, frames, c, h, w = x.shape
        
        # CNN은 4차원만 받으므로 [Batch*Frames, 3, 224, 224]로 펼침
        x_flat = x.view(batch * frames, c, h, w)
        
        # 특징 추출
        feat = self.backbone(x_flat) # [B*F, 512, 1, 1]
        feat = feat.view(batch * frames, -1) # [B*F, 512]
        feat = self.fc(feat) # [B*F, 512]
        
        # 다시 시간 축 복구: [Batch, Frames, 512]
        return feat.view(batch, frames, -1)


# [2] 유틸리티: 라벨 사전 만들기 & 배치 샘플링

def build_label_map(label_dir):
    """
    JSON 파일들을 읽어서 {파일명: 라벨인덱스} 사전을 만듭니다.
    """
    print(f" 라벨 데이터 읽는 중... ({label_dir})")
    video_to_label = {}
    label_to_idx = {}
    
    # 예: JSON 파일명이 비디오 파일명과 대응된다고 가정
    # 실제 데이터 구조에 따라 수정 필요할 수 있음
    files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    
    for json_file in files:
        # 1. JSON 읽기
        with open(os.path.join(label_dir, json_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 2. 단어(Gloss) 추출 (AI Hub 구조: attributes -> name)
        # 구조가 복잡하면 print(data)로 확인 필요
        try:
            # 데이터 구조에 따라 경로가 다를 수 있음. 가장 흔한 패턴 시도:
            word = data.get('attributes', [{}])[0].get('name')
            if not word: continue
            
            # 3. 라벨 인덱싱 (화장실 -> 0, 가다 -> 1 ...)
            if word not in label_to_idx:
                label_to_idx[word] = len(label_to_idx)
            
            # 4. 비디오 파일명 매핑 (확장자만 json -> mp4로 변경 가정)
            video_name = json_file.replace('.json', '.mp4')
            video_to_label[video_name] = label_to_idx[word]
            
        except Exception as e:
            continue # 에러 난 파일은 건너뜀

    print(f"✅ 총 {len(label_to_idx)}개 단어 클래스 발견!")
    print(f"✅ 총 {len(video_to_label)}개 학습용 데이터 매핑 완료.")
    return video_to_label, len(label_to_idx)

def get_episodic_batch(dataset, n_way, k_shot, q_query):
    """
    데이터셋에서 N개의 클래스를 골라 K+Q개의 샘플을 뽑아 배치를 만듭니다.
    (복잡한 Sampler 대신 간단하게 구현)
    """
    # 데이터셋 전체 라벨 가져오기 (dataset.label_dict 역참조가 필요하지만 간단히 처리)
    # 실제로는 클래스별로 인덱스를 미리 정리해두는 게 효율적
    
    # 1. 이번 에피소드에서 사용할 N개 클래스 랜덤 선택
    available_classes = list(set(dataset.label_dict.values()))
    if len(available_classes) < n_way:
        # 클래스가 부족하면 있는 거 다 씀
        selected_classes = available_classes
        real_n_way = len(selected_classes)
    else:
        selected_classes = random.sample(available_classes, n_way)
        real_n_way = n_way

    support_images, support_labels = [], []
    query_images, query_labels = [], []

    # 2. 각 클래스별로 데이터 뽑기
    class_indices = {} # {label: [idx1, idx2...]}
    for idx, (name, label) in enumerate(dataset.label_dict.items()):
        if label in selected_classes:
            if label not in class_indices: class_indices[label] = []
            class_indices[label].append(idx)

    for i, cls in enumerate(selected_classes):
        indices = class_indices[cls]
        # 데이터가 모자르면 중복 허용해서 뽑기
        needed = k_shot + q_query
        if len(indices) >= needed:
            sampled_idxs = random.sample(indices, needed)
        else:
            sampled_idxs = random.choices(indices, k=needed)
            
        # Support Set 담기
        for idx in sampled_idxs[:k_shot]:
            img, _ = dataset[idx] # img: [Frames, 3, H, W] (dataset.py 수정 필요할 수 있음)
            support_images.append(img)
            support_labels.append(i) # 0 ~ N-1 로 재매핑
            
        # Query Set 담기
        for idx in sampled_idxs[k_shot:]:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(i)

    # 텐서로 변환
    support_images = torch.stack(support_images)
    query_images = torch.stack(query_images)
    support_labels = torch.tensor(support_labels)
    
    return support_images, support_labels, query_images, query_labels, real_n_way


# ====================================================
# [3] 메인 학습 실행 코드
# ====================================================
def train():
    # 설정
    DATA_DIR = "data/raw_videos"
    LABEL_DIR = "data/raw_labels"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): DEVICE = "mps" # 맥북 가속
    
    print(f"🚀 학습 시작! (Device: {DEVICE})")

    # 1. 데이터 준비
    video_label_map, num_classes = build_label_map(LABEL_DIR)
    if len(video_label_map) == 0:
        print("❌ 매핑된 데이터가 없습니다. 폴더 경로와 파일명을 확인하세요.")
        return

    dataset = SignLanguageDataset(
        video_dir=DATA_DIR,
        label_dict=video_label_map,
        max_frames=30,
        transform=None # 필요 시 transform 추가
    )
    
    # 2. 모델 준비
    # (A) Encoder: 임시 ResNet (나중에 SLIP으로 교체)
    encoder = DummyResNetEncoder().to(DEVICE)
    # (B) Time Model: 우리가 만든 Hybrid 모델
    time_model = HybridTemporalModel(input_dim=512).to(DEVICE)
    # (C) Classifier: ProtoNet
    classifier = ProtoNetClassifier().to(DEVICE)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(time_model.parameters()), 
        lr=1e-4
    )

    # 3. 학습 루프 (Episode 반복)
    N_WAY = 5   # 한 번에 5개 단어 구분 연습
    K_SHOT = 1  # 정답 예시는 1개만 봄
    Q_QUERY = 1 # 문제는 1개 풂
    MAX_EPISODES = 100 # 100번 반복

    for episode in range(MAX_EPISODES):
        # (1) 배치 만들기 (복잡한 과정은 함수가 처리)
        try:
            s_img, s_lbl, q_img, _, real_n_way = get_episodic_batch(
                dataset, N_WAY, K_SHOT, Q_QUERY
            )
        except ValueError as e:
            print("⚠️ 데이터 로딩 중 오류 (파일 부족 등):", e)
            continue
            
        s_img = s_img.to(DEVICE) # [N*K, Frames, 3, H, W]
        s_lbl = s_lbl.to(DEVICE)
        q_img = q_img.to(DEVICE) # [N*Q, Frames, 3, H, W]
        
        # (2) Forward Pass
        # 이미지 -> 특징 벡터 (Encoder)
        s_feat = encoder(s_img) # [N*K, Frames, 512]
        q_feat = encoder(q_img)
        
        # 시퀀스 -> 비디오 벡터 (Hybrid Model)
        s_emb = time_model(s_feat) # [N*K, 512]
        q_emb = time_model(q_feat)
        
        # 분류 (ProtoNet)
        # 주의: ProtoNet은 Logits(음수 거리)를 반환함
        logits = classifier(s_emb, s_lbl, q_emb, real_n_way)
        
        # (3) Loss 계산 (정답은 0, 1, 2... 순서대로 들어감)
        # Query 라벨 만들기 (0,0,0... 1,1,1... 식)
        target = torch.arange(real_n_way).repeat_interleave(Q_QUERY).to(DEVICE)
        
        loss = nn.CrossEntropyLoss()(logits, target)
        
        # (4) Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (episode+1) % 10 == 0:
            print(f"[{episode+1}/{MAX_EPISODES}] Loss: {loss.item():.4f}")

    print("🎉 학습 완료! (임시 테스트 성공)")

if __name__ == "__main__":
    train()