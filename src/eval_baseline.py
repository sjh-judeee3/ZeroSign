import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- [설정] ---
DATA_ROOT = "eval_data_resized"

# MediaPipe는 기본적으로 7가지 제스처만 인식합니다.
# 우리가 원하는 동작과 가장 비슷한 것으로 매핑해야 채점이 가능합니다.
# None으로 설정된 것은 MediaPipe가 아예 모르는 동작이라는 뜻입니다.
MP_MAPPING = {
    # MediaPipe Output  : Our Class Label
    "Closed_Fist"       : "Grab",
    "Open_Palm"         : "Stop",
    "Pointing_Up"       : "Point",
    "Thumb_Up"          : None, # 우리 데이터엔 엄지척 없음 -> 오답 처리
    "Thumb_Down"        : None,
    "Victory"           : "Pinch", # 가끔 Pinch를 브이(Victory)로 착각함 (매핑해줘도 됨)
    "ILoveYou"          : None
}

# 우리 클래스 리스트 (정답지)
OUR_CLASSES = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])

def get_mediapipe_prediction(video_path):
    """
    영상 전체 프레임을 돌면서 가장 많이 나온 제스처를 최종 예측으로 선정 (Voting)
    """
    # 모델 로드 (가장 가벼운 기본 모델 사용)
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # MediaPipe용 이미지 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 추론
        recognition_result = recognizer.recognize(mp_image)

        if recognition_result.gestures:
            # 가장 확신하는 제스처 이름 가져오기
            top_gesture = recognition_result.gestures[0][0].category_name
            predictions.append(top_gesture)
        else:
            predictions.append("None")

    cap.release()

    if not predictions:
        return "Unknown"

    # 가장 많이 나온 예측값(최빈값) 선택
    from collections import Counter
    most_common = Counter(predictions).most_common(1)[0][0]
    
    # 우리의 라벨로 변환 (Mapping)
    final_pred = MP_MAPPING.get(most_common, "Unknown")
    
    # 매핑되지 않은(Unknown) 결과는 우리 클래스 중 아무거나 하나로 찍거나(Random),
    # 오답 처리를 위해 'Wrong'이라는 라벨을 둡니다.
    # 여기서는 정확한 비교를 위해 우리 클래스에 없으면 그냥 'Unknown'으로 둡니다.
    
    return final_pred

def run_baseline_eval():
    # 모델 파일이 없으면 다운로드
    if not os.path.exists('gesture_recognizer.task'):
        print("📥 Downloading MediaPipe Model...")
        os.system('wget -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task')

    print(f"📂 Evaluating Baseline (MediaPipe) on {DATA_ROOT}...")
    
    y_true = []
    y_pred = []
    
    # 쿼리/서포트 구분 없이 전체 데이터로 평가 (Baseline은 Few-shot이 아니므로)
    # 하지만 공정한 비교를 위해 test data 전체를 다 씁니다.
    
    for class_name in OUR_CLASSES:
        class_dir = os.path.join(DATA_ROOT, class_name)
        video_files = sorted(glob.glob(os.path.join(class_dir, "*.mp4")))
        
        print(f"   Processing {class_name} ({len(video_files)} videos)...")
        
        for v_path in video_files:
            # 정답
            y_true.append(class_name)
            
            # 예측
            pred = get_mediapipe_prediction(v_path)
            
            # 예측값이 우리 클래스 리스트에 없으면 (예: Pinch인데 Unknown이라고 함) -> 틀린 것으로 간주
            # 편의상 예측 리스트에 그대로 넣습니다. (나중에 Confusion Matrix에서 별도 컬럼으로 뜸)
            y_pred.append(pred)

    # --- 결과 계산 ---
    # 정확도 계산 시 Unknown이나 None은 무조건 오답 처리됨
    acc = accuracy_score(y_true, y_pred)
    print(f"\n📉 Baseline Accuracy: {acc * 100:.2f}%")
    
    # Confusion Matrix (Unknown 포함해서 그리기)
    # y_pred에 'Unknown'이나 'Grab' 등이 섞여 있음.
    # 시각화를 위해 라벨 유니온을 만듦
    all_labels = sorted(list(set(y_true + y_pred)))
    
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', # 일부러 빨간색 (경고 느낌)
                xticklabels=all_labels, yticklabels=all_labels)
    plt.title(f'Baseline Confusion Matrix (Acc: {acc*100:.1f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('baseline_result.png')
    plt.show()

if __name__ == "__main__":
    run_baseline_eval()