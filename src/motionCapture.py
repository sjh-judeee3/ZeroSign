import cv2
import numpy as np
import torch
import time
import os
import sounddevice as sd
from scipy.io.wavfile import write
from run_multimodal import main_run 

# --- 설정값 ---
FPS = 30
CAPTURE_DURATION = 3.0  # 3초 동안 수어 동작 캡처
OUTPUT_VIDEO_PATH = "captured_video.pt"  # PyTorch 텐서 파일
OUTPUT_AUDIO_PATH = "captured_audio.wav" # 오디오 파일

# ⚠️ 사용자의 최종 학습된 모델 및 프로토타입 경로로 변경하세요.
MODEL_PATH = "slip_protonet_final.pth" 
PROTO_PATH = "prototypes.pt"

def record_audio(filename, duration, samplerate=44100):
    """음성을 녹음하여 WAV 파일로 저장"""
    print(f"\n🎤 {duration}초 동안 음성 녹음 시작...")
    try:
        # 녹음 시작
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()  # 녹음이 끝날 때까지 대기
        write(filename, samplerate, recording)
        print(f"✅ 음성 녹음 완료: {filename}")
    except Exception as e:
        print(f"❌ 음성 녹음 실패 (마이크 설정 및 'sounddevice' 권한 확인 필요): {e}")



def main_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return None, None

    # ProtoNet 인풋 형식: T, 3, H, W (프레임 수, 채널, 높이, 너비)
    # SLIP은 보통 224x224를 사용합니다.
    TARGET_SIZE = (224, 224) 
    
    frames = []
    start_time = time.time()

    # 1. 오디오 녹음을 비동기 또는 병렬로 시작 (여기서는 간단히 순차 처리)
    # 실제 시연 시에는 별도 쓰레드로 처리해야 합니다.
    record_audio(OUTPUT_AUDIO_PATH, CAPTURE_DURATION) 
    
    print(f"🎬 {CAPTURE_DURATION}초 동안 수어 동작 캡처 시작...")

    while time.time() - start_time < CAPTURE_DURATION:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # 캡처된 프레임 처리 (ProtoNet 형식에 맞게)
        processed_frame = cv2.flip(frame, 1)  # 좌우 반전
        processed_frame = cv2.resize(processed_frame, TARGET_SIZE)
        # BGR -> RGB 및 정규화 (0-255 -> 0-1)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(processed_frame)

        cv2.putText(frame, f"Capturing: {len(frames)} frames", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)
        cv2.imshow("Sign Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 비디오 캡처 완료.")

    if not frames:
        print("캡처된 프레임이 없습니다.")
        return None, None

    # 2. PyTorch 텐서로 변환 (T, 3, H, W)
    video_np = np.stack(frames) # (T, H, W, 3)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2) # (T, 3, H, W)
    
    # 3. 텐서 저장
    torch.save(video_tensor, OUTPUT_VIDEO_PATH)
    print(f"✅ 비디오 텐서 저장 완료: {OUTPUT_VIDEO_PATH}")

    return OUTPUT_VIDEO_PATH, OUTPUT_AUDIO_PATH

if __name__ == '__main__':
    video_file, audio_file = main_capture()
    
    # -------------------------------------------------------------
    # 🌟🌟🌟 통합 실행: 캡처 완료 후 run_multimodal의 main_run 호출 🌟🌟🌟
    # -------------------------------------------------------------
    if video_file and audio_file:
        print("\n==================================================")
        print("          ✨ 캡처 완료! 멀티모달 추론 실행...      ")
        print("==================================================")
        
        # motionCapture.py가 run_multimodal.py의 main_run 함수를 직접 호출합니다.
        main_run(MODEL_PATH, PROTO_PATH, video_file, audio_file)
    else:
        print("❌ 캡처 오류로 인해 추론을 시작할 수 없습니다.")