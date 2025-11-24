import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# ✅ 우리가 만든 파일들과 정확히 매칭되는 임포트
from dataset import SignLanguageDataset
from models import SLIP_ProtoNet

# ====================================================
# [설정] 경로를 정확히 수정했습니다!
# ====================================================
LABEL_DIR = "/content/drive/MyDrive/Capstone/수어영상2/labels_01"
VIDEO_DIR = "/content/drive/MyDrive/Capstone/fin_videos_extracted"

MAX_EPISODES = 100  # 테스트용 (나중엔 10000 이상으로 늘리세요)
N_WAY = 5           # 5지 선다
K_SHOT = 1          # 정답지 1개
Q_QUERY = 1         # 문제 1개
LR = 0.001          # 학습률

def train():
    # 1. 디바이스 설정 (GPU/MPS/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    print(f"🚀 학습 시작! Device: {device}")

    # 2. 데이터셋 준비
    print("📂 데이터셋 로드 중...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # [수정] label_dir, video_dir 두 개를 넣어야 합니다!
    dataset = SignLanguageDataset(label_dir=LABEL_DIR, video_dir=VIDEO_DIR, transform=transform)
    
    # 3. Few-shot을 위한 라벨별 인덱스 정리
    print("📊 데이터를 라벨별로 분류 중... (시간이 좀 걸립니다)")
    label_to_indices = {}
    
    # tqdm으로 진행상황 표시
    for idx in tqdm(range(len(dataset))):
        try:
            # 데이터셋 내부 리스트에 접근해서 라벨만 빠르게 추출
            # (__getitem__을 쓰면 영상을 읽어서 느려짐 -> 최적화)
            import json
            with open(dataset.json_paths[idx], 'r', encoding='utf-8') as f:
                meta = json.load(f)
                label = meta['data'][0]['attributes'][0]['name']
            
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        except:
            continue

    # 데이터가 너무 적은 클래스 제외
    min_samples = K_SHOT + Q_QUERY
    valid_labels = [lbl for lbl, idxs in label_to_indices.items() if len(idxs) >= min_samples]
    print(f"✅ 학습 가능 단어 수: {len(valid_labels)}개 (총 라벨 {len(label_to_indices)}개 중)")

    if len(valid_labels) < N_WAY:
        print(f"❌ 에러: N_WAY({N_WAY})보다 학습 가능한 단어 수가 적습니다.")
        return

    # 4. 모델 준비 (SLIP_ProtoNet 하나로 해결)
    model = SLIP_ProtoNet(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    model.train()

    # 5. 학습 루프 (Episode Training)
    print("🔥 Training Loop Start...")
    for episode in range(MAX_EPISODES):
        optimizer.zero_grad()
        
        # (1) 이번 에피소드용 샘플링
        sampled_classes = random.sample(valid_labels, N_WAY)
        
        support_imgs = []
        query_imgs = []
        target_labels = [] 

        for i, class_label in enumerate(sampled_classes):
            indices = label_to_indices[class_label]
            # 중복 없이 K+Q개 뽑기
            selected_indices = random.sample(indices, K_SHOT + Q_QUERY)
            
            # Support Set
            for idx in selected_indices[:K_SHOT]:
                img, _ = dataset[idx]
                support_imgs.append(img)
                
            # Query Set
            for idx in selected_indices[K_SHOT:]:
                img, _ = dataset[idx]
                query_imgs.append(img)
                target_labels.append(i) # 0~4 사이 정답 라벨

        # 텐서 합치기 & 이동
        support_imgs = torch.stack(support_imgs).to(device)
        query_imgs = torch.stack(query_imgs).to(device)
        target_labels = torch.tensor(target_labels).to(device)

        # (2) 모델 예측 (Forward)
        # SLIP_ProtoNet이 내부에서 인코딩 -> 프로토타입 생성 -> 거리 계산까지 다 해줍니다.
        log_probs = model(support_imgs, query_imgs, N_WAY, K_SHOT)
        
        # (3) Loss 계산 & 업데이트
        loss = torch.nn.functional.nll_loss(log_probs, target_labels)
        loss.backward()
        optimizer.step()

        # (4) 정확도 출력
        y_pred = log_probs.argmax(1)
        acc = (y_pred == target_labels).float().mean()

        if (episode + 1) % 10 == 0:
            print(f"Episode [{episode+1}/{MAX_EPISODES}] Loss: {loss.item():.4f} | Acc: {acc.item()*100:.2f}%")

    print("🎉 학습 완료!")
    # 모델 저장
    torch.save(model.state_dict(), "slip_protonet_final.pth")

if __name__ == "__main__":
    train()