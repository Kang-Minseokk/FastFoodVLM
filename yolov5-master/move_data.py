import os
import re
from collections import Counter

def analyze_food_dataset(train_path='data/food_image/labels/train'):
    """
    음식 이미지 데이터셋 분석
    - 클래스 총 개수
    - 각 클래스별 파일 개수  
    - 총 파일 개수
    """
    
    # 경로 존재 확인
    if not os.path.exists(train_path):
        print(f"경로를 찾을 수 없습니다: {train_path}")
        return
    
    # 이미지 파일 확장자들
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    
    # 모든 이미지 파일 가져오기
    # files = [f for f in os.listdir(train_path) if f.lower().endswith(image_extensions)]
    files = [f for f in os.listdir(train_path) if f.lower().endswith('txt')]
    # 음식 이름 추출하여 카운트
    food_counts = Counter()
    for file in files:
        # 파일명에서 확장자 제거
        name_without_ext = os.path.splitext(file)[0]
        # 끝의 숫자들 제거하여 음식 이름만 추출  
        food_name = re.sub(r'\d+$', '', name_without_ext)
        if food_name:
            food_counts[food_name] += 1
    
    # 결과 출력
    print(f"클래스 총 개수: {len(food_counts)}")
    print(f"총 파일 개수: {len(files)}")
    print("\n각 클래스별 파일 개수:")
    for food, count in sorted(food_counts.items()):
        print(f"{food}: {count}")

if __name__ == "__main__":
    analyze_food_dataset()