import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

# ✅ 모듈 임포트 (파일명 정확히 확인!)
from dataset import SignLanguageDataset
from encoder import SLIPVideoEncoder 
from models import HybridTemporalModel, ProtoNetClassifier

# ====================================================
# [설정] 경로 및 파라미터
# ====================================================
LABEL_DIR = "/content/drive/MyDrive/Capstone/morpheme/01"
VIDEO_DIR = "/content/drive/MyDrive/Capstone/fin_videos_extracted"

MAX_EPISODES = 100  
N_WAY = 5           # 5지 선다
K_SHOT = 1          # 정답지 1개
Q_QUERY = 1         # 문제 1개
LR = 1e-4           # 학습률 (Transformer라 조금 낮춤)

def get_episodic_batch(label_to_indices, dataset, n_way, k_shot, q_query):
    """에피소드(N-way K-shot) 배치를 생성하는 함수"""
    # 1. N개의 클래스 랜덤 선택
    valid_labels = [l for l, idxs in label_to_indices.items() if len(idxs) >= k_shot + q_query]
    if len(valid_labels) < n_way: return None, None, None, None
    
    selected_classes = random.sample(valid_labels, n_way)
    
    support_imgs, query_imgs = [], []
    support_labels, query_labels = [], []
    
    for i, class_label in enumerate(selected_classes):
        indices = label_to_indices[class_label]
        selected_indices = random.sample(indices, k_shot + q_query)
        
        # Support Set
        for idx in selected_indices[:k_shot]:
            img, _ = dataset[idx]
            support_imgs.append(img)
            support_labels.append(i)
            
        # Query Set
        for idx in selected_indices[k_shot:]:
            img, _ = dataset[idx]
            query_imgs.append(img)
            query_labels.append(i)
            
    return torch.stack(support_imgs), torch.tensor(support_labels), \
           torch.stack(query_imgs), torch.tensor(query_labels)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 시작! Device: {device}")

    # 1. 데이터셋 준비
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = SignLanguageDataset(LABEL_DIR, VIDEO_DIR, transform=transform)
    
    # 인덱싱 (속도 최적화)
    print("📊 데이터 분류 중...")
    label_to_indices = {}
    for idx in tqdm(range(len(dataset))):
        try:
            import json
            with open(dataset.json_paths[idx], 'r', encoding='utf-8') as f:
                label = json.load(f)['data'][0]['attributes'][0]['name']
            if label not in label_to_indices: label_to_indices[label] = []
            label_to_indices[label].append(idx)
        except: continue

    # 2. 모델 초기화 (3단 합체!)
    # (A) Encoder: 이미지 -> 프레임별 특징 (B, T, 512)
    encoder = SLIPVideoEncoder(pretrained=True, embed_dim=512).to(device)
    
    # (B) Temporal: 프레임별 특징 -> 비디오 벡터 (B, 512)
    # models.py의 HybridTemporalModel 사용
    temporal_model = HybridTemporalModel(input_dim=512, hidden_dim=512).to(device)
    
    # (C) Classifier: 비디오 벡터 -> 거리 계산 & 분류
    classifier = ProtoNetClassifier().to(device)
    
    # Optimizer (Encoder와 Temporal 모델 둘 다 학습)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(temporal_model.parameters()), 
        lr=LR
    )

    # 3. 학습 루프
    print("🔥 Training Loop Start...")
    model_save_path = "slip_protonet_final.pth"
    
    for episode in range(MAX_EPISODES):
        # 배치 생성
        s_imgs, s_lbls, q_imgs, q_lbls = get_episodic_batch(
            label_to_indices, dataset, N_WAY, K_SHOT, Q_QUERY
        )
        
        if s_imgs is None: 
            print("❌ 학습 가능한 데이터가 부족합니다."); break

        s_imgs, s_lbls = s_imgs.to(device), s_lbls.to(device)
        q_imgs, q_lbls = q_imgs.to(device), q_lbls.to(device)

        optimizer.zero_grad()

        # --- Forward Pass (모델 연결) ---
        # 1. Encoder (Frame Features)
        s_features = encoder(s_imgs) # Output: (N*K, T, 512)
        q_features = encoder(q_imgs) # Output: (N*Q, T, 512)
        
        # 2. Temporal Model (Video Embedding)
        s_emb = temporal_model(s_features) # Output: (N*K, 512)
        q_emb = temporal_model(q_features) # Output: (N*Q, 512)
        
        # 3. ProtoNet Classifier
        # Output: Logits (음수 거리값)
        logits = classifier(s_emb, s_lbls, q_emb, N_WAY)
        
        # --- Loss & Update ---
        loss = torch.nn.functional.cross_entropy(logits, q_lbls)
        loss.backward()
        optimizer.step()

        # 정확도
        acc = (logits.argmax(1) == q_lbls).float().mean()

        if (episode + 1) % 10 == 0:
            print(f"Episode [{episode+1}/{MAX_EPISODES}] Loss: {loss.item():.4f} | Acc: {acc.item()*100:.2f}%")

    print("🎉 학습 완료!")
    # 모델 저장 (Encoder와 Temporal 둘 다 저장해야 함)
    torch.save({
        'encoder': encoder.state_dict(),
        'temporal': temporal_model.state_dict()
    }, model_save_path)

if __name__ == "__main__":
    train()