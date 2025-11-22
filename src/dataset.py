import os
import json
import glob
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_frames=16):
        self.data_dir = data_dir
        self.transform = transform
        self.num_frames = num_frames
        
        # 폴더 내의 모든 JSON 파일을 찾습니다.
        self.json_paths = glob.glob(os.path.join(data_dir, "*.json"))
        
        if len(self.json_paths) == 0:
            print(f"Warning: No JSON files found in {data_dir}")
        else:
            print(f"Found {len(self.json_paths)} data samples in {data_dir}")

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        json_path = self.json_paths[idx]
        
        # 1. JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 라벨(텍스트) 추출
        # 구조: data -> [0] -> attributes -> [0] -> name ("가락로")
        try:
            label_text = data['data'][0]['attributes'][0]['name']
        except (KeyError, IndexError):
            # 예외 처리: 데이터가 비어있거나 형식이 다를 경우
            label_text = "Unknown"

        # 3. 비디오 파일 경로 추출
        # 구조: metaData -> name ("NIA_SL_FS0001_CROWD18_F.mp4")
        video_filename = data['metaData']['name']
        video_path = os.path.join(self.data_dir, video_filename)
        
        # 4. 비디오 로드 (파일이 존재하는지 확인)
        if not os.path.exists(video_path):
            # 혹시 영상이 JSON과 같은 폴더에 없다면 에러가 날 수 있음
            # 이럴 땐 로깅을 하고 0으로 채운 텐서를 반환하거나 스킵해야 함
            print(f"Video file missing: {video_path}")
            # 임시 방편: 빈 텐서 반환 (학습 시 에러 방지용, 실제론 데이터 확인 필요)
            # return torch.zeros((3, self.num_frames, 224, 224)), label_text
            raise FileNotFoundError(f"Video not found: {video_path}")

        frames = self._load_video(video_path)
        
        # 5. 전처리 및 텐서 변환
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
            
        # (Frames, Channels, H, W) -> (Channels, Frames, H, W) 로 차원 변경 (모델에 따라 다름)
        # SLIP/VideoMAE 등은 보통 (C, T, H, W)를 선호
        frames = frames.permute(1, 0, 2, 3) 

        return frames, label_text

    def _load_video(self, video_path):
        """비디오에서 균등한 간격으로 프레임 추출"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
             # 영상이 깨졌거나 못 읽을 때
            cap.release()
            # 검은 화면 등으로 대체하거나 에러 발생
            return [Image.new('RGB', (224, 224)) for _ in range(self.num_frames)]

        # 프레임 인덱스 결정 (균등 샘플링)
        if total_frames <= self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / self.num_frames
            frame_indices = [int(i * step) for i in range(self.num_frames)]
            
        frames = []
        current_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_idx in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames) == self.num_frames:
                    break
            current_idx += 1
        
        cap.release()
        
        # 프레임 모자라면 마지막 프레임 복사
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
            
        return frames

# === 테스트 실행 코드 ===
if __name__ == "__main__":
    # 경로 설정 (코랩용)
    # 주의: 영상 파일(.mp4)도 이 폴더 안에 JSON과 같이 들어있어야 합니다!
    data_path = "/content/drive/MyDrive/Capstone/수어영상2/18"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    try:
        dataset = SignLanguageDataset(data_dir=data_path, transform=transform)
        # 첫 번째 데이터 확인
        frames, label = dataset[0]
        print(f"Label: {label}")     # 기대값: "가락로"
        print(f"Shape: {frames.shape}") # 기대값: torch.Size([3, 16, 224, 224])
    except Exception as e:
        print(f"Error: {e}")