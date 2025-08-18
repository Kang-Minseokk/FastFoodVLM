import os 
import random
import shutil

# yolov5-master/raw_data 아래에 음식 이름으로 생성한 디렉토리 아래에 labels와 images 디렉토리를 
# 생성한 상태로 코드를 실행해야 합니다.

# 데이터 증강 또는 새롭게 추가하려는 음식의 이름을 입력하세요!
food_name = "blueberry"

os.chdir(f"./raw_data/{food_name}")

image_files = os.listdir('images')
label_files = os.listdir('labels')
image_path = 'images'
label_path = 'labels'

img_file_count = len([f for f in image_files ])
txt_file_count = len([f for f in label_files ])

print(f"이미지 파일 개수: {img_file_count}")
print(f"라벨 파일 개수: {txt_file_count}")

# 파일 이름을 변경하는 코드입니다.
# Create a copy of the image_files list before iterating
image_files_copy = image_files[:]

for idx, file_name in enumerate(image_files_copy, start=114):
    # 이미지 파일의 확장자를 추출합니다.
    img_ext = os.path.splitext(file_name)[1]

    # 이미지 파일과 동일한 이름의 라벨 파일이 있는지 확인합니다.
    label_file_name = os.path.splitext(file_name)[0] + '.txt'
    if label_file_name in label_files:
        # 새로운 파일 이름을 생성합니다. --> 바꾸고자하는 음식 클래스 이름으로 바꾸기
        new_name = f"{food_name}{idx}"

        # 이미지 파일의 전체 경로
        old_img_path = os.path.join(image_path, file_name)
        new_img_path = os.path.join(image_path, f"{new_name}{img_ext}")

        # 라벨 파일의 전체 경로
        old_label_path = os.path.join(label_path, label_file_name)
        new_label_path = os.path.join(label_path, f"{new_name}.txt")

        # 이미지 파일과 라벨 파일을 새 이름으로 이동합니다.
        shutil.move(old_img_path, new_img_path)
        shutil.move(old_label_path, new_label_path)

print("파일 이름 변경 완료!")