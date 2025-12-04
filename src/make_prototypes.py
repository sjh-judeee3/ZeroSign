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
    
    MAX_SAMPLES_PER_CLASS=10
    unique_classes = set()
    print(f"📊 전체 데이터 임베딩 추출 중... (클래스별 최대 {MAX_SAMPLES_PER_CLASS}개 제한 적용)")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            try:
                img, label_path = dataset[i]
                
                # 라벨 이름 추출 (JSON 파싱)
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_name = json.load(f)['data'][0]['attributes'][0]['name']
                unique_classes.add(label_name)
                
                if label_name in class_embeddings and len(class_embeddings[label_name]) >= MAX_SAMPLES_PER_CLASS:
                    continue
                img = img.unsqueeze(0).to(DEVICE) # (1, T, C, H, W)
                
                # 특징 추출
                feats = encoder(img)
                emb = temporal(feats) # (1, 512)
                
                if label_name not in class_embeddings:
                    class_embeddings[label_name] = []
                class_embeddings[label_name].append(emb.cpu()) # CPU로 내려서 저장 (메모리 절약)
            except Exception as e:
                continue
            all_classes_are_full = True
            for name in unique_classes:
                # 아직 이 클래스가 class_embeddings에 없거나, 샘플 수가 10개 미만이면
                if name not in class_embeddings or len(class_embeddings[name]) < MAX_SAMPLES_PER_CLASS:
                    all_classes_are_full = False
                    break
            # 모든 클래스가 10개 이상 모였다면 루프를 종료
            if all_classes_are_full and len(unique_classes) > 0:
                 print("\n🎉 모든 클래스가 최소 샘플 수를 충족했습니다. 프로토타입 생성 루프를 종료합니다.")
                 break

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