import torch
import whisper
import os
from openai import OpenAI # 혹은 로컬 LLM 라이브러리
from encoder import SLIPVideoEncoder 
from models import HybridTemporalModel

# API 키 설정 (OpenAI 사용하는 경우)
# os.environ["OPENAI_API_KEY"] = "sk-..." 

class MultimodalAgent:
    def __init__(self, model_path, proto_path, device="cuda"):
        self.device = device
        
        # 1. 수어 모델 로드
        self.encoder = SLIPVideoEncoder(pretrained=False, embed_dim=512).to(device)
        self.temporal = HybridTemporalModel(input_dim=512, hidden_dim=512).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.temporal.load_state_dict(checkpoint['temporal'])
        self.encoder.eval()
        self.temporal.eval()

        # 2. 프로토타입(기준점) 로드 - 여기가 핵심 변경점!
        print("📂 수어 기준점(Prototype) 로딩 중...")
        self.prototypes = torch.load(proto_path, map_location=device) 
        # self.prototypes는 {"안녕하세요": tensor, "배고파": tensor ...} 형태
        
        # 딕셔너리를 텐서 행렬로 변환 (계산 속도를 위해)
        self.class_names = list(self.prototypes.keys())
        self.proto_matrix = torch.stack([self.prototypes[k] for k in self.class_names]).to(device) 
        # (Class_Num, 512)

        # 3. Whisper 로드
        self.whisper = whisper.load_model("base").to(device)

        # 4. LLM 클라이언트 (OpenAI 예시, 로컬 모델이면 transformers pipeline 사용)
        self.client = OpenAI() 

    def predict_sign(self, video_tensor):
        """저장된 프로토타입과 비교하여 가장 가까운 수어 단어 찾기"""
        with torch.no_grad():
            video_tensor = video_tensor.to(self.device)
            features = self.encoder(video_tensor)
            query_emb = self.temporal(features) # (1, 512)

            # 유클리드 거리 계산 (Euclidean Distance)
            # (Query - Proto)^2
            dists = torch.cdist(query_emb, self.proto_matrix) # (1, Class_Num)
            
            # 가장 거리가 짧은 인덱스 찾기
            min_dist_idx = torch.argmin(dists, dim=1).item()
            predicted_word = self.class_names[min_dist_idx]
            
            return predicted_word

    def generate_response(self, video_tensor, audio_path):
        # 1. 인식 수행
        sign_word = self.predict_sign(video_tensor)
        audio_result = self.whisper.transcribe(audio_path)['text']
        
        print(f"👀 수어 인식: {sign_word}")
        print(f"👂 음성 인식: {audio_result}")

        # 2. LLM 프롬프트 (Prompt Engineering)
        system_prompt = "당신은 청각 장애인과 비장애인의 소통을 돕는 통역사입니다. 수어 단어와 음성 텍스트가 주어지면, 문맥을 고려하여 사용자의 의도를 완벽한 한국어 문장으로 만드세요."
        
        user_prompt = f"""
        [입력 정보]
        수어 단어: {sign_word}
        음성 텍스트: {audio_result}

        [지시 사항]
        1. 수어 단어는 핵심 키워드입니다.
        2. 음성 텍스트가 불완전하거나 짧으면 수어 단어를 사용하여 내용을 보완하세요.
        3. 반대로 수어 단어만으로 부족하면 음성을 참고하세요.
        4. 결과는 '해석된 문장' 딱 하나만 출력하세요.

        [예시 1]
        수어: 배고파 / 음성: 엄마 밥
        해석: 엄마, 저 배고파요. 밥 주세요.

        [예시 2]
        수어: 병원 / 음성: 머리가 너무 아파
        해석: 머리가 너무 아파서 병원에 가고 싶어요.

        [실제 문제]
        수어: {sign_word} / 음성: {audio_result}
        해석:
        """

        # 3. LLM 호출
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo", # or gpt-4o
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

# 실행 예시
# agent = MultimodalAgent("slip_protonet_final.pth", "prototypes.pt")
# print(agent.generate_response(dummy_video, "audio.mp3"))