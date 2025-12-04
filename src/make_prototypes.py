import torch
import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

# ✅ 사용자 정의 모델 임포트 (기존 코드와 동일)
try:
    from src.encoder import SLIPVideoEncoder 
    from src.models import HybridTemporalModel, ProtoNetClassifier
except ImportError:
    from encoder import SLIPVideoEncoder
    from models import HybridTemporalModel, ProtoNetClassifier

# --- [설정] ---
DATA_ROOT = "eval_data_resized"                   # 전처리된 데이터 폴더
CHECKPOINT_PATH = "checkpoint/slip_protonet_final.pth" # 학습된 가중치 경로
OUTPUT_PATH = "checkpoint/command_prototypes_demo.pt"       # 저장할 프로토타입 파일명
N_SUPPORT = 3                                     # 프로토타입 계산에 사용할 영상 개수
NUM_FRAMES = 16
EMBED_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps"

# ==========================================
# 1. 유틸리티 함수 (eval.py에서 재사용)
# ==========================================

# (load_video_tensor 함수는 길기 때문에 생략하고, eval.py에 있는 것을 사용한다고 가정합니다.)
# (load_trained_models 함수는 길기 때문에 생략하고, eval.py에 있는 것을 사용한다고 가정합니다.)
# [주의] 아래 코드의 함수들은 eval.py에서 복사해와야 합니다!

# 예시: eval.py에서 load_video_tensor, load_trained_models 함수를 복사해야 함.

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



def load_trained_models():
    print(f"🔄 Loading models on {DEVICE}...")
    
    # 모델 초기화 (train.py의 파라미터와 동일하게!)
    encoder = SLIPVideoEncoder(pretrained=False, embed_dim=EMBED_DIM).to(DEVICE)
    time_model = HybridTemporalModel(input_dim=EMBED_DIM, hidden_dim=EMBED_DIM).to(DEVICE)
    
    # 체크포인트 로드
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoints not found at {CHECKPOINT_PATH}")
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # train.py에서 저장한 키: 'encoder', 'temporal'
    print(f"✅ Checkpoint Keys Found: {list(checkpoint.keys())}")
    
    try:
        encoder.load_state_dict(checkpoint['encoder'])
        time_model.load_state_dict(checkpoint['temporal'])
        print("✅ Weights loaded successfully!")
    except KeyError as e:
        print(f"❌ Key Error loading checkpoint: {e}")
        print("train.py의 저장 코드와 키 값이 일치하는지 확인하세요.")
        return None, None
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None, None

    encoder.eval()
    time_model.eval()
    
    return encoder, time_model


# ==========================================
# 2. 프로토타입 계산 및 저장
# ==========================================
def save_prototypes_for_demo():
    print(f"🔄 Loading models and calculating Prototypes (N={N_SUPPORT})...")
    
    encoder, time_model = load_trained_models()
    if encoder is None: return

    # 1. 클래스 탐색
    all_classes = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    # Grab, Pinch, Point를 포함한 모든 클래스에 대해 계산 진행
    print(f"📂 Found Classes for Prototypes: {all_classes}")
    
    support_embs = []
    support_lbls = [] 
    class_map = [] # 클래스 이름과 인덱스 매핑

    # 2. Support Set 특징 추출
    for label_idx, class_name in enumerate(all_classes):
        class_dir = os.path.join(DATA_ROOT, class_name)
        video_files = sorted(glob.glob(os.path.join(class_dir, "*.mp4")))
        
        # 앞의 N_SUPPORT 개 파일만 사용
        s_files = video_files[:N_SUPPORT]
        
        if len(s_files) < N_SUPPORT:
            print(f"⚠️  Warning: Class {class_name} has only {len(s_files)} videos. Skipping or padding.")
            continue
            
        print(f"   [Processing] {class_name} with {len(s_files)} videos...")
        
        # 특징 추출
        for v_path in tqdm(s_files, desc=f"Feat. Extraction for {class_name}"):
            tensor = load_video_tensor(v_path, NUM_FRAMES)
            if tensor is None: continue
            tensor = tensor.to(DEVICE)
            
            with torch.no_grad():
                f_feat = encoder(tensor)
                vid_emb = time_model(f_feat)
                
            support_embs.append(vid_emb.cpu())
            support_lbls.append(label_idx)
        
        class_map.append(class_name)

    if not support_embs:
        print("❌ 추출된 Support 데이터가 없습니다. 경로를 확인하세요.")
        return

    # 3. Prototypes 계산
    S = torch.cat(support_embs)
    S_Y = torch.tensor(support_lbls)

    # ProtoNetClassifier의 compute_prototypes 로직을 재사용 (수동 계산)
    dim = S.size(1)
    num_classes = len(class_map)
    prototypes = torch.zeros(num_classes, dim)
    
    for c in range(num_classes):
        class_samples = S[S_Y == c]
        if class_samples.size(0) > 0:
            prototypes[c] = class_samples.mean(dim=0)
            
    # 4. 파일 저장
    data_to_save = {
        'classes': class_map,
        'prototypes': prototypes.cpu(),
        'embedding_dim': EMBED_DIM,
        'N_support': N_SUPPORT
    }

    torch.save(data_to_save, OUTPUT_PATH)
    print(f"\n✅ Success! Prototypes for {num_classes} classes saved.")
    print(f"💾 File Path: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":

    try:
        save_prototypes_for_demo()
    except NameError as e:
        print("\n--- 실행 오류 ---")
        print(f"❌ 함수 정의 오류: {e}")
        print("eval.py의 'load_video_tensor' 및 'load_trained_models' 함수를 복사하여 이 파일에 정의한 후 실행해 주세요.")