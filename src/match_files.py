import os

def match_files():
    # ---------------------------------------------------------
    # [설정] 파일명 및 확장자 설정
    # ---------------------------------------------------------
    TARGET_LIST_FILE = 'filename.txt'  # 찾고 싶은 파일명 리스트 (예: train.csv의 내용)
    OUTPUT_FILE = 'matched.txt'        # 결과 저장 파일
    SEARCH_EXT = '.json'               # 로컬 폴더에서 찾을 파일 확장자 (실제 존재하는 파일)
    
    print(f"🚀 매칭 시작!")
    print(f"1. 로컬 탐색 확장자: {SEARCH_EXT}")
    print(f"2. 타겟 리스트 파일: {TARGET_LIST_FILE}")

    # ---------------------------------------------------------
    # 1. filename.txt 읽기 (타겟 리스트)
    # ---------------------------------------------------------
    if not os.path.exists(TARGET_LIST_FILE):
        print(f"❌ 오류: '{TARGET_LIST_FILE}' 파일이 없습니다.")
        return

    target_map = {} # { "파일명(확장자X)": "원래파일명(확장자O)" }
    
    with open(TARGET_LIST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            full_name = line.strip()
            if not full_name: continue
            
            # 확장자를 뗀 이름만 키(Key)로 사용 (예: "NIA_..._D")
            stem_name = os.path.splitext(full_name)[0]
            target_map[stem_name] = full_name
            
    print(f"📂 타겟 리스트 로드 완료: {len(target_map)}개 파일")

    # ---------------------------------------------------------
    # 2. 로컬 폴더 순회 (01~16 등 모든 하위 폴더)
    # ---------------------------------------------------------
    matched_results = []
    
    # 현재 디렉토리(.)부터 시작해서 모든 하위 폴더를 뒤집니다.
    for root, dirs, files in os.walk('.'):
        # root: 현재 탐색 중인 폴더 경로 (예: ./01)
        
        # 현재 폴더명 추출 (예: 01)
        folder_name = os.path.relpath(root, '.')
        if folder_name == '.': continue # 최상위 폴더에 있는 파일은 건너뜀 (필요 시 주석 처리)

        for file in files:
            # 우리가 찾는 확장자(.json)인지 확인
            if file.endswith(SEARCH_EXT):
                # 파일명에서 확장자 제거 (로컬 파일의 줄기 이름)
                local_stem = os.path.splitext(file)[0]
                
                # 3. 매칭 확인
                if local_stem in target_map:
                    # 매칭 성공!
                    # filename.txt에 적혀있던 원래 이름(확장자 포함)을 가져옴
                    original_target_name = target_map[local_stem]
                    
                    # 결과 포맷: 폴더명/파일명 (예: 03/NIA_..._D.mp4)
                    # 윈도우(\) 경로 구분자를 리눅스(/) 스타일로 통일
                    match_str = f"{folder_name}/{original_target_name}".replace('\\', '/')
                    matched_results.append(match_str)

    # ---------------------------------------------------------
    # 3. 결과 저장 (matched.txt)
    # ---------------------------------------------------------
    matched_results.sort() # 보기 좋게 정렬
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(matched_results))
        
    print(f"\n🎉 작업 완료!")
    print(f"✅ 총 {len(matched_results)}개의 파일이 매칭되었습니다.")
    print(f"💾 결과가 '{OUTPUT_FILE}'에 저장되었습니다.")

if __name__ == "__main__":
    match_files()