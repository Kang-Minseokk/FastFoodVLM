# 이미지와 라벨의 쌍이 일치하는지 여부를 확인하기 위한 코드입니다.

import os

image_path = 'food_image/images/train'

image_names = os.listdir(image_path)
print(len(image_names))

label_path = 'food_image/labels/train'

label_names = os.listdir(label_path)
print(len(label_names))

image_stems = {os.path.splitext(f)[0] for f in image_names}
label_stems = {os.path.splitext(f)[0] for f in label_names}

extra_labels = label_stems - image_stems
print(f"라벨만 있고 이미지에는 없는 훈련 데이터 개수 : {len(extra_labels)}")

image_path = 'food_image/images/val'

image_names = os.listdir(image_path)
print(len(image_names))

label_path = 'food_image/labels/val'

label_names = os.listdir(label_path)
print(len(label_names))

image_stems = {os.path.splitext(f)[0] for f in image_names}
label_stems = {os.path.splitext(f)[0] for f in label_names}

extra_labels = label_stems - image_stems
print(f"라벨만 있고 이미지에는 없는 검증 데이터 개수 : {len(extra_labels)}")

# for lbl in extra_labels:
#     label_file = os.path.join(label_path, lbl + '.txt')
#     if os.path.exists(label_file):
#         os.remove(label_file)
#         print(f"Removed: {label_file}")
