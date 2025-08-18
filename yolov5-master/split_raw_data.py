import random
import shutil
import os

base_output_dir = './data/food_image'
train_image_dir = os.path.join(base_output_dir, 'images/train')
val_image_dir = os.path.join(base_output_dir, 'images/val')
train_label_dir = os.path.join(base_output_dir, 'labels/train')
val_label_dir = os.path.join(base_output_dir, 'labels/val')

for d in [train_image_dir, val_image_dir, train_label_dir, val_label_dir]:
    os.makedirs(d, exist_ok=True)

# 분할할 음식의 이름을 작성해주세요
dir_names = ["blueberry"]

# 이미지 및 라벨 파일 수집
all_image_label_pairs = []
parent_directory_path = './raw_data'

for dir_name in dir_names:
    image_dir = os.path.join(parent_directory_path, dir_name, 'images')
    label_dir = os.path.join(parent_directory_path, dir_name, 'labels')

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        img_path = os.path.join(image_dir, img_file)
        lbl_path = os.path.join(label_dir, label_file)
        if os.path.exists(lbl_path):
            all_image_label_pairs.append((img_path, lbl_path))

# 데이터 셔플 및 분할
random.seed(42)
random.shuffle(all_image_label_pairs)
split_idx = int(len(all_image_label_pairs) * 0.8)
train_pairs = all_image_label_pairs[:split_idx]
val_pairs   = all_image_label_pairs[split_idx:]

# 복사 함수
def copy_pairs(pairs, image_dst_dir, label_dst_dir):
    for img_src, lbl_src in pairs:
        img_dst = os.path.join(image_dst_dir, os.path.basename(img_src))
        lbl_dst = os.path.join(label_dst_dir, os.path.basename(lbl_src))
        shutil.copy(img_src, img_dst)
        shutil.copy(lbl_src, lbl_dst)

# 복사 실행
copy_pairs(train_pairs, train_image_dir, train_label_dir)
copy_pairs(val_pairs, val_image_dir, val_label_dir)

print(f"총 {len(all_image_label_pairs)}개 중 {len(train_pairs)}개는 훈련용, {len(val_pairs)}개는 검증용으로 복사되었습니다.")