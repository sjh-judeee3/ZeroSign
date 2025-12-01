""""
데이터 전처리 첫단계
평가용 영상 화질, 크기 너무 커서 resize and crop
"""
import cv2
import os
import glob
from tqdm import tqdm

# --- [설정] ---
INPUT_ROOT = "eval_data"          
OUTPUT_ROOT = "eval_data_resized"
TARGET_SIZE = 224

# [추가] 영상을 얼마나 회전할지 설정
# cv2.ROTATE_180 : 180도 회전 (뒤집힌 경우)
# cv2.ROTATE_90_CLOCKWISE : 시계 방향 90도 (누워있는 경우)
# cv2.ROTATE_90_COUNTERCLOCKWISE : 반시계 방향 90도
ROTATE_CODE = cv2.ROTATE_180 

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # [중요] 회전 후의 가로/세로 길이는 그대로인지 확인
    # 180도 회전은 가로/세로가 유지되지만, 90도 회전은 서로 바뀝니다.
    # 지금은 180도 뒤집힘 문제이므로 그대로 진행합니다.
    
    min_dim = min(width, height)
    
    # Center Crop 좌표 계산
    start_x = (width - min_dim) // 2
    end_x = start_x + min_dim
    start_y = (height - min_dim) // 2
    # start_y = height - min_dim # (필요시 주석 해제: 바닥 기준 Crop)
    end_y = start_y + min_dim

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_SIZE, TARGET_SIZE))

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- [수정된 부분] 1. 먼저 회전 시키기 ---
        # 영상이 뒤집혀 있다면 여기서 돌려줍니다.
        frame = cv2.rotate(frame, ROTATE_CODE) 
        
        # 2. Crop (정사각형으로 자르기)
        crop_frame = frame[start_y:end_y, start_x:end_x]
        
        # 3. Resize (224x224로 줄이기)
        if crop_frame.shape[0] > 0 and crop_frame.shape[1] > 0:
            resized_frame = cv2.resize(crop_frame, (TARGET_SIZE, TARGET_SIZE))
            out.write(resized_frame)
        
    cap.release()
    out.release()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    input_dir_abs = os.path.join(project_root, INPUT_ROOT)
    output_dir_abs = os.path.join(project_root, OUTPUT_ROOT)

    print(f"Input Directory: {input_dir_abs}")
    print(f"Output Directory: {output_dir_abs}")

    if not os.path.exists(input_dir_abs):
        print("Error: 입력 폴더를 찾을 수 없습니다.")
        return

    video_files = []
    for ext in ["*.MOV", "*.mov", "*.mp4", "*.MP4"]:
        video_files.extend(glob.glob(os.path.join(input_dir_abs, "**", ext), recursive=True))

    print(f"총 {len(video_files)}개의 비디오 파일을 발견했습니다. (180도 회전 적용됨)")

    for file_path in tqdm(video_files):
        relative_path = os.path.relpath(file_path, input_dir_abs)
        output_path = os.path.join(output_dir_abs, relative_path)
        output_path = os.path.splitext(output_path)[0] + ".mp4"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        process_video(file_path, output_path)

    print("\n[완료] 회전 및 변환 완료!")

if __name__ == "__main__":
    main()