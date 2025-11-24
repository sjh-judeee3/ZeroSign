import os
import json
import glob
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SignLanguageDataset(Dataset):
    def __init__(self, label_dir, video_dir, transform=None, num_frames=16):
        """
        Args:
            label_dir (str): JSON 라벨 파일들이 있는 폴더
            video_dir (str): MP4 영상 파일들이 있는 폴더 (하위 폴더 포함 검색)
        """
        self.label_dir = label_dir
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames
        
        print(f"🔍 데이터셋 초기화 중... (Label: {label_dir}, Video: {video_dir})")

        # 1. 영상 파일 미리 찾아서 지도(Map) 만들기
        # (영상이 더 적으므로 영상을 기준으로 JSON을 찾는 게 안전합니다)
        self.video_map = {}
        mp4_files = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
        
        for path in mp4_files:
            filename = os.path.basename(path)
            self.video_map[filename] = path
            
        print(f"🎥 영상(MP4) 파일 {len(self.video_map)}개 위치 확보.")

        # 2. 모든 JSON 파일 경로 리스트업
        all_json_paths = glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True)
        print(f"📄 발견된 전체 라벨(JSON) 파일: {len(all_json_paths)}개")
        
        # 3. [핵심 수정] 영상이 존재하는 JSON만 리스트에 추가 (필터링)
        self.json_paths = []
        print("⚙️ 유효한 데이터 쌍 매칭 중... (잠시만 기다려주세요)")
        
        for json_path in all_json_paths:
            try:
                # JSON을 살짝 열어서 비디오 파일명을 확인
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                # JSON 안에 적힌 비디오 파일명
                target_video_name = meta['metaData']['name']
                
                # 그 파일명이 우리 비디오 지도(Map)에 있다면 합격!
                if target_video_name in self.video_map:
                    self.json_paths.append(json_path)
                    
            except Exception:
                continue # JSON이 깨졌거나 형식이 다르면 패스

        if len(self.json_paths) == 0:
            print(f"❌ 경고: 매칭되는 데이터가 하나도 없습니다! 경로를 확인하세요.")
        else:
            print(f"🎉 최종 학습 데이터셋 완성: 총 {len(self.json_paths)}개 쌍 (영상 O, 라벨 O)")

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        json_path = self.json_paths[idx]
        
        # 1. JSON 파일 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 라벨(텍스트) 추출
        try:
            label_text = data['data'][0]['attributes'][0]['name']
        except (KeyError, IndexError):
            label_text = "Unknown"

        # 3. 비디오 파일 경로 찾기
        video_filename = data['metaData']['name']
        
        # __init__에서 검증했으므로 여기선 무조건 존재한다고 가정 (안전)
        video_path = self.video_map[video_filename]

        # 4. 비디오 로드
        frames = self._load_video(video_path)
        
        # 5. 전처리 및 텐서 변환
        if self.transform:
            # frames가 빈 리스트가 아닐 때만 변환
            if len(frames) > 0:
                frames = torch.stack([self.transform(frame) for frame in frames])
            else:
                # 만약 영상 로드에 실패했다면 더미 데이터 반환 (에러 방지)
                frames = torch.zeros((self.num_frames, 3, 224, 224))

        # (Frames, Channels, H, W) -> (Channels, Frames, H, W)
        # SLIP 모델은 (C, T, H, W)를 좋아합니다.
        frames = frames.permute(1, 0, 2, 3) 

        return frames, label_text

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            # 영상이 깨졌을 경우를 대비해 검은 화면 반환
            return [Image.new('RGB', (224, 224)) for _ in range(self.num_frames)]

        if total_frames <= self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / self.num_frames
            frame_indices = [int(i * step) for i in range(self.num_frames)]
            
        frames = []
        current_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if current_idx in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                if len(frames) == self.num_frames: break
            current_idx += 1
        cap.release()
        
        # 프레임 모자라면 마지막 프레임 복사해서 채우기
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
            
        return frames

# === 테스트 실행 코드 ===
if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # 경로 설정 (코랩 환경)
    label_path = "/content/drive/MyDrive/Capstone/수어영상2/labels_01" 
    video_path = "/content/drive/MyDrive/Capstone/fin_videos_extracted"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    try:
        dataset = SignLanguageDataset(label_dir=label_path, video_dir=video_path, transform=transform)
        
        # 정상적으로 5031개가 나오는지 확인
        print(f"데이터셋 길이: {len(dataset)}") 
        
        if len(dataset) > 0:
            frames, label = dataset[0]
            print(f"첫 번째 데이터: {label}, {frames.shape}")
            
    except Exception as e:
        print(f"\n❌ 에러: {e}")