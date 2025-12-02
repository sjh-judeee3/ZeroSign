"""
eval metric
t-Sne visualization
"""

import torch
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

# ✅ 사용자 정의 모듈 (경로에 맞게 수정)
try:
    from src.encoder import SLIPVideoEncoder 
    from src.models import HybridTemporalModel
except ImportError:
    from encoder import SLIPVideoEncoder
    from models import HybridTemporalModel

# --- [설정] ---
DATA_ROOT = "eval_data_resized"
CHECKPOINT_PATH = "checkpoint/slip_protonet_final.pth" # 경로 확인!
NUM_FRAMES = 16
EMBED_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps"

# (기존 load_video_tensor 함수 복사)
def load_video_tensor(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return None
    
    if total_frames <= num_frames: indices = np.arange(total_frames)
    else: indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    current_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if current_idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            if len(frames) == num_frames: break
        current_idx += 1
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224,224,3), dtype=np.float32))
    
    frames = np.array(frames)
    frames = np.transpose(frames, (3, 0, 1, 2)) 
    return torch.tensor(frames).unsqueeze(0)

# (기존 load_trained_models 함수 복사)
def load_trained_models():
    print(f"🔄 Loading models on {DEVICE}...")
    encoder = SLIPVideoEncoder(pretrained=False, embed_dim=EMBED_DIM).to(DEVICE)
    time_model = HybridTemporalModel(input_dim=EMBED_DIM, hidden_dim=EMBED_DIM).to(DEVICE)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    encoder.load_state_dict(checkpoint['encoder'])
    time_model.load_state_dict(checkpoint['temporal'])
    
    encoder.eval()
    time_model.eval()
    return encoder, time_model

def visualize_tsne():
    encoder, time_model = load_trained_models()
    classes = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    all_embs = []
    all_labels = []
    
    print("🚀 Extracting Embeddings for t-SNE...")
    
    # 모든 데이터를 다 뽑습니다 (Support/Query 구분 없이 전체 분포 확인)
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(DATA_ROOT, class_name)
        video_files = sorted(glob.glob(os.path.join(class_dir, "*.mp4")))
        
        print(f"   Processing {class_name}...")
        for v_path in video_files:
            tensor = load_video_tensor(v_path, NUM_FRAMES)
            if tensor is None: continue
            tensor = tensor.to(DEVICE)
            
            with torch.no_grad():
                f_feat = encoder(tensor)
                emb = time_model(f_feat) # [1, 512]
            
            all_embs.append(emb.cpu().numpy().flatten())
            all_labels.append(class_name) # 텍스트 라벨 그대로 사용

    # t-SNE 실행
    print("🎨 Running t-SNE...")
    X = np.array(all_embs)
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='random', learning_rate=200, method='exact')
    X_embedded = tsne.fit_transform(X)
    
    # 시각화
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=all_labels, 
                    palette='bright', s=100, alpha=0.8, edgecolor='k')
    
    plt.title("t-SNE Visualization of SignVLM Embeddings", fontsize=15)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = "tsne_result.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ t-SNE saved at {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_tsne()