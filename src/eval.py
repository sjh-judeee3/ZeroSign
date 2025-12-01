import torch
import torch.nn as nn
import numpy as np
import os
import glob
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 사용자 정의 모듈 임포트
# (파일 구조가 src/encoder_test.py, src/models.py 라고 가정)
try:
    from src.encoder_test import SLIPVisualEncoder
    from src.models import HybridTemporalModel, ProtoNetClassifier
except ImportError:
    # src 폴더 내부에서 실행할 경우
    from encoder_test import SLIPVisualEncoder
    from models import HybridTemporalModel, ProtoNetClassifier

# --- [설정] ---
DATA_ROOT = "eval_data_resized"       # 전처리된 데이터 폴더
CHECKPOINT_PATH = "checkpoints/slip_protonet_final.pth" # 학습된 가중치 경로
N_SUPPORT = 3                         # 클래스당 기준 영상 개수 (Few-shot Support)
NUM_FRAMES = 16                       # 모델이 학습할 때 썼던 프레임 수
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps" # 맥북용

# ==========================================
# 1. 평가용 데이터 로더 (JSON 없이 MP4 직접 로드)
# ==========================================
def load_video_tensor(video_path, num_frames=16):
    """
    MP4 파일을 읽어 [1, C, T, H, W] 텐서로 변환
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Error: Empty video {video_path}")
        return None

    # 균등 간격 샘플링 (Uniform Sampling)
    if total_frames <= num_frames:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    current_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if current_idx in indices:
            # BGR -> RGB & Normalize (0~1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            if len(frames) == num_frames:
                break
        current_idx += 1
    cap.release()

    # 프레임 부족 시 마지막 프레임 복사 (Padding)
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224,224,3), dtype=np.float32))

    # Numpy -> Tensor 변환
    # frames: [T, H, W, C] -> [C, T, H, W] (Model Input)
    frames = np.array(frames)
    frames = np.transpose(frames, (3, 0, 1, 2)) 
    
    return torch.tensor(frames).unsqueeze(0) # Batch 차원 추가 [1, C, T, H, W]

# ==========================================
# 2. 모델 로드 및 가중치 복원
# ==========================================
def load_trained_models():
    print(f"🔄 Loading models on {DEVICE}...")
    
    # 모델 초기화
    encoder = SLIPVisualEncoder(model_name='vit_base_patch16_224').to(DEVICE)
    time_model = HybridTemporalModel(input_dim=encoder.output_dim).to(DEVICE) # output_dim=768
    
    # 체크포인트 로드
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoints not found at {CHECKPOINT_PATH}")
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # 저장 방식에 따라 분기 처리
    # Case 1: {'encoder': ..., 'time_model': ...} 딕셔너리로 저장된 경우 (권장)
    if isinstance(checkpoint, dict) and 'encoder_state_dict' in checkpoint:
        print("✅ Detected dictionary checkpoint format.")
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        time_model.load_state_dict(checkpoint['time_model_state_dict'])
        
    # Case 2: 모델 전체가 저장된 경우 or 다른 키값
    else:
        print("⚠️ Warning: Unknown checkpoint format. Trying direct load...")
        # 만약 키 값이 다르다면 아래를 수정해야 합니다. 
        # 예: checkpoint['model'] 등. 
        # 지금은 에러가 날 수 있으니 일단 패스하거나 사용자 확인 필요.
        try:
            encoder.load_state_dict(checkpoint['encoder']) # 키 이름 추측
            time_model.load_state_dict(checkpoint['hybrid'])
        except:
            print("❌ 가중치 로드 실패. train.py에서 저장 방식을 확인해주세요.")
            print(f"Available keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")

    encoder.eval()
    time_model.eval()
    
    return encoder, time_model

# ==========================================
# 3. 평가 실행 (ProtoNet Logic)
# ==========================================
def run_evaluation():
    # 1. 모델 준비
    encoder, time_model = load_trained_models()
    
    # 2. 클래스 탐색
    classes = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    print(f"📂 Found Classes: {classes}")
    
    support_embs = []
    support_lbls = []
    query_embs = []
    query_lbls = [] # 정답지

    print("\n🚀 Extracting Features & Split Data (Support vs Query)...")
    
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(DATA_ROOT, class_name)
        video_files = sorted(glob.glob(os.path.join(class_dir, "*.mp4")))
        
        if len(video_files) == 0:
            print(f"⚠️  Skipping empty class: {class_name}")
            continue
            
        # 데이터 분할 (앞의 N개는 Support, 나머지는 Query)
        # 만약 영상이 3개 이하라면? -> 1개를 Support, 나머지를 Query로 강제 조정
        cur_n_support = N_SUPPORT
        if len(video_files) <= N_SUPPORT:
            cur_n_support = 1
            print(f"⚠️  {class_name}: Not enough videos. Using 1 for support.")

        s_files = video_files[:cur_n_support]
        q_files = video_files[cur_n_support:]
        
        print(f"   [{class_name}] Support: {len(s_files)} | Query: {len(q_files)}")

        # --- Support Set 처리 ---
        for v_path in s_files:
            tensor = load_video_tensor(v_path, NUM_FRAMES)
            if tensor is None: continue
            tensor = tensor.to(DEVICE)
            
            with torch.no_grad():
                # Encoder (Video -> Frame Features)
                f_feat = encoder(tensor) # [1, T, 768]
                # Time Model (Frame Features -> Video Vector)
                vid_emb = time_model(f_feat) # [1, 768]
                
            support_embs.append(vid_emb.cpu())
            support_lbls.append(label_idx)

        # --- Query Set 처리 ---
        for v_path in q_files:
            tensor = load_video_tensor(v_path, NUM_FRAMES)
            if tensor is None: continue
            tensor = tensor.to(DEVICE)
            
            with torch.no_grad():
                f_feat = encoder(tensor)
                vid_emb = time_model(f_feat)
                
            query_embs.append(vid_emb.cpu())
            query_lbls.append(label_idx)

    # 리스트 -> 텐서 변환
    S = torch.cat(support_embs).to(DEVICE) # [Total_Support, Dim]
    S_Y = torch.tensor(support_lbls).to(DEVICE)
    Q = torch.cat(query_embs).to(DEVICE)   # [Total_Query, Dim]
    Q_Y = np.array(query_lbls)             # Metric 계산용 (numpy)

    # ==========================================
    # 4. ProtoNet 거리 계산 및 분류
    # ==========================================
    classifier = ProtoNetClassifier().to(DEVICE)
    
    # (1) 프로토타입 계산
    num_classes = len(classes)
    prototypes = classifier.compute_prototypes(S, S_Y, num_classes) # [N_Class, Dim]
    
    # (2) 거리 계산 (Query vs Prototypes)
    # Output: [N_Query, N_Classes]
    dists = classifier.euclidean_distance(Q, prototypes)
    
    # (3) 예측 (거리가 가장 짧은 클래스 선택)
    # dists가 작을수록 좋음 -> argmin
    predictions = torch.argmin(dists, dim=1).cpu().numpy()
    
    # ==========================================
    # 5. 결과 시각화 및 저장
    # ==========================================
    acc = accuracy_score(Q_Y, predictions)
    print(f"\n🏆 Final Accuracy: {acc * 100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(Q_Y, predictions, target_names=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(Q_Y, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Acc: {acc*100:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_path = 'evaluation_result.png'
    plt.savefig(save_path)
    print(f"\n✅ Result image saved at: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_evaluation()