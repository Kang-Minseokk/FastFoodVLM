import os

# 경로 설정
target_dir = "./food_image/images/val"

# 파일 개수 세기
file_count = sum(
    1 for fname in os.listdir(target_dir)
    if os.path.isfile(os.path.join(target_dir, fname))
)

print(f"총 파일 개수: {file_count}개")
