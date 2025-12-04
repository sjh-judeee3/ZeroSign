import torch
import os
import json
from tqdm import tqdm
from dataset import SignLanguageDataset
from encoder import SLIPVideoEncoder
from models import HybridTemporalModel
from torchvision import transforms

# 설정
LABEL_DIR = "/content/drive/MyDrive/Capstone/morpheme/01"
VIDEO_DIR = "/content/drive/MyDrive/Capstone/fin_videos_extracted"
MODEL_PATH = "/content/drive/MyDrive/Capstone/slip_protonet_final.pth"
SAVE_PATH = "/content/drive/MyDrive/Capstone/prototypes.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_prototypes():
    print(f"🛠️ 프로토타입(클래스별 기준점) 생성 시작... Device: {DEVICE}")

    # 1. 모델 로드
    encoder = SLIPVideoEncoder(pretrained=False, embed_dim=512).to(DEVICE)
    temporal = HybridTemporalModel(input_dim=512, hidden_dim=512).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    encoder.load_state_dict(checkpoint['encoder'])
    temporal.load_state_dict(checkpoint['temporal'])
    
    encoder.eval()
    temporal.eval()

    # 2. 데이터셋 로드 (전체 데이터를 다 봅니다)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = SignLanguageDataset(LABEL_DIR, VIDEO_DIR, transform=transform)

    # 3. 클래스별 임베딩 모으기
    class_embeddings = {} # { "안녕하세요": [tensor1, tensor2...], "감사합니다": [...] }
    
    # 🌟🌟🌟 설정 변수 🌟🌟🌟
    MAX_SAMPLES_PER_CLASS = 10  # 각 클래스별로 최대 10개의 영상만 사용
    
    # 🚨🚨🚨 비상 테스트용: 전체 순회 횟수를 강제로 제한합니다. 🚨🚨🚨
    # 5013개를 모두 돌면 4시간이 걸리므로, 일단 500개만 처리하고 종료합니다.
    TOTAL_ITERATION_LIMIT = 500 # <--- 이 값을 변경하면 테스트 시간이 바뀝니다.
    
    print(f"📊 데이터 임베딩 추출 중... (클래스별 최대 {MAX_SAMPLES_PER_CLASS}개 / 총 {TOTAL_ITERATION_LIMIT}회 제한)")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            
            # -----------------------------------------------
            # 🚨 핵심 수정 1: 전체 순회 횟수 제한 (가장 빠르고 확실한 단축 방법)
            if i >= TOTAL_ITERATION_LIMIT:
                print(f"\n🚨 [비상 테스트] 총 순회 횟수 {TOTAL_ITERATION_LIMIT}회 도달. 강제 종료합니다.")
                break
            # -----------------------------------------------
            
            try:
                img, label_path = dataset[i]
                
                # 라벨 이름 추출 (JSON 파싱)
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_name = json.load(f)['data'][0]['attributes'][0]['name']
                
                # -----------------------------------------------
                # 핵심 수정 2: 이미 충분한 샘플을 모았다면 비디오 로딩 후라도 건너뛰기
                if label_name in class_embeddings and len(class_embeddings[label_name]) >= MAX_SAMPLES_PER_CLASS:
                    continue
                # -----------------------------------------------

                img = img.unsqueeze(0).to(DEVICE) # (1, T, C, H, W)
                
                # 특징 추출
                feats = encoder(img)
                emb = temporal(feats) # (1, 512)
                
                if label_name not in class_embeddings:
                    class_embeddings[label_name] = []
                class_embeddings[label_name].append(emb.cpu()) # CPU로 내려서 저장 (메모리 절약)
                
            except Exception as e:
                continue

    # 4. 평균(Prototype) 계산
    prototypes = {}
    print("🧮 클래스별 평균 계산 중...")
    for label, emb_list in class_embeddings.items():
        # 스택 후 평균 계산
        emb_stack = torch.cat(emb_list, dim=0) # (N, 512)
        proto = emb_stack.mean(dim=0) # (512,) -> 이게 바로 그 클래스의 기준점!
        prototypes[label] = proto

    # 5. 저장
    torch.save(prototypes, SAVE_PATH)
    print(f"✅ 프로토타입 저장 완료! 경로: {SAVE_PATH}")
    print(f"총 {len(prototypes)}개의 수어 단어 기준점이 생성되었습니다.")

if __name__ == "__main__":
    create_prototypes()