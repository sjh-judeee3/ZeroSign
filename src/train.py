import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ✅ 모듈 임포트 (수정됨: encoder -> encoder_test)
from src.dataset import SignLanguageDataset
from src.encoder_test import SLIPVisualEncoder  # 파일명 변경 반영
from src.models import HybridTemporalModel, ProtoNetClassifier

# ====================================================
# [1] 데이터셋 인덱싱 유틸리티 (텍스트 -> 숫자 매핑)
# ====================================================
def create_class_indices(dataset):
    """
    데이터셋을 한 번 훑어서 {단어(Text): [인덱스 리스트]} 맵을 만듭니다.
    Few-shot 배치를 만들 때 특정 단어의 데이터를 빨리 찾기 위함입니다.
    """
    print("📂 데이터셋 인덱싱 중... (클래스별 샘플 분류)")
    class_indices = {}
    
    # dataset 길이는 __len__으로 알 수 있음
    for idx in tqdm(range(len(dataset))):
        try:
            # __getitem__을 호출하면 영상 로딩 때문에 느리므로
            # dataset 내부의 json_paths를 직접 읽어 라벨만 빼오는 방식을 씁니다.
            
            json_path = dataset.json_paths[idx]
            # JSON 로드 로직 복사 (dataset.py 로직 참조)
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                label = data['data'][0]['attributes'][0]['name']
                
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
            
        except Exception as e:
            continue # 에러난 데이터는 스킵

    print(f"✅ 인덱싱 완료! 총 {len(class_indices)}개 클래스 발견.")
    return class_indices

# ====================================================
# [2] 에피소딕 배치 샘플러 (N-way K-shot)
# ====================================================
def get_episodic_batch(dataset, class_indices, n_way, k_shot, q_query):
    """
    dataset: 원본 데이터셋
    class_indices: {라벨: [idx1, idx2...]} 딕셔너리
    """
    # 1. N개의 클래스 랜덤 선택
    available_classes = list(class_indices.keys())
    if len(available_classes) < n_way:
        selected_classes = available_classes
        real_n_way = len(selected_classes)
    else:
        selected_classes = random.sample(available_classes, n_way)
        real_n_way = n_way

    support_images = []
    query_images = []
    
    # 2. 각 클래스에서 데이터 뽑기
    for i, cls_name in enumerate(selected_classes):
        indices = class_indices[cls_name]
        needed = k_shot + q_query
        
        # 데이터가 부족하면 중복 허용해서 뽑기
        if len(indices) >= needed:
            sampled_idxs = random.sample(indices, needed)
        else:
            sampled_idxs = random.choices(indices, k=needed)
            
        # Support Set (정답지)
        for idx in sampled_idxs[:k_shot]:
            # dataset[idx]는 (frames, label_text)를 반환
            img, _ = dataset[idx] 
            support_images.append(img)
            
        # Query Set (문제)
        for idx in sampled_idxs[k_shot:]:
            img, _ = dataset[idx]
            query_images.append(img)
            
    # 3. 텐서로 변환
    # dataset이 [C, T, H, W]를 주므로 stack하면 [Batch, C, T, H, W]가 됨
    support_images = torch.stack(support_images)
    query_images = torch.stack(query_images)
    
    # 라벨은 0부터 N-1까지 숫자로 새로 만듦 (이번 에피소드용)
    support_labels = torch.arange(real_n_way).repeat_interleave(k_shot)
    query_labels = torch.arange(real_n_way).repeat_interleave(q_query)
    
    return support_images, support_labels, query_images, query_labels, real_n_way

# ====================================================
# [3] 학습 실행
# ====================================================
def train():
    # 설정 (경로 수정하세요!)
    DATA_DIR = "/content/drive/MyDrive/Capstone/수어영상2/18" 
    
    # 맥북(MPS) / CUDA / CPU 자동 선택
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): DEVICE = "mps"
    
    print(f"🚀 학습 시작! Device: {DEVICE}")

    # 1. 데이터셋 & 전처리 준비
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # Normalize 등을 추가하면 더 좋음
    ])
    
    dataset = SignLanguageDataset(
        data_dir=DATA_DIR, 
        transform=transform, 
        num_frames=16 # SLIP Encoder 예시와 맞춤
    )
    
    # 클래스별 인덱스 정리 (Few-shot 배치를 위해 필수)
    class_indices = create_class_indices(dataset)
    if len(class_indices) == 0:
        print("❌ 유효한 데이터가 없습니다.")
        return

    # 2. 모델 초기화
    # (A) Encoder: SLIPVisualEncoder (파일명 encoder_test.py)
    encoder = SLIPVisualEncoder(model_name='vit_base_patch16_224').to(DEVICE)
    
    # (B) Hybrid Model: SLIP의 출력차원(768)에 맞춤
    time_model = HybridTemporalModel(input_dim=encoder.output_dim).to(DEVICE)
    
    # (C) Classifier
    classifier = ProtoNetClassifier().to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(time_model.parameters()), 
        lr=1e-5 # ViT는 학습률을 낮게 잡는 게 좋음
    )

    # 3. 학습 루프
    MAX_EPISODES = 100
    N_WAY = 5
    K_SHOT = 1
    Q_QUERY = 1
    
    print("\n🔥 Training Loop Start...")
    
    for episode in range(MAX_EPISODES):
        try:
            # 배치 생성 (이미지, 라벨)
            s_imgs, s_lbls, q_imgs, q_lbls, real_n = get_episodic_batch(
                dataset, class_indices, N_WAY, K_SHOT, Q_QUERY
            )
            
            s_imgs = s_imgs.to(DEVICE) # [N*K, C, T, H, W]
            s_lbls = s_lbls.to(DEVICE)
            q_imgs = q_imgs.to(DEVICE) # [N*Q, C, T, H, W]
            q_lbls = q_lbls.to(DEVICE) # 정답 라벨

            # --- Forward Pass ---
            # 1. Encoder (Video -> Frame Features)
            s_feat = encoder(s_imgs) # [N*K, T, 768]
            q_feat = encoder(q_imgs)
            
            # 2. Hybrid Model (Frame Features -> Video Vector)
            s_emb = time_model(s_feat) # [N*K, 768]
            q_emb = time_model(q_feat)
            
            # 3. ProtoNet (거리 계산)
            logits = classifier(s_emb, s_lbls, q_emb, real_n)
            
            # --- Loss & Backward ---
            loss = nn.CrossEntropyLoss()(logits, q_lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (episode+1) % 10 == 0:
                print(f"Episode [{episode+1}/{MAX_EPISODES}] Loss: {loss.item():.4f}")

        except Exception as e:
            print(f"⚠️ Episode {episode} Failed: {e}")
            continue

    print("🎉 학습 완료!")

if __name__ == "__main__":
    train()