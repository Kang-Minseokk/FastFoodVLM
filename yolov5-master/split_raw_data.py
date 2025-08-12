import os, glob, yaml
from collections import defaultdict

# ==== 경로 설정 ====
DATA_YAML = "data/food.yaml"          # data.yaml 경로
DATA_ROOT = "data/food_image"         # images/, labels/ 가 들어있는 루트
SPLIT = "train"                            # 'train' 또는 'val'

# ==== 유틸 ====
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

with open(DATA_YAML, 'r') as f:
    y = yaml.safe_load(f)
names = y.get('names', [])
nc = y.get('nc', len(names))

labels_dir = os.path.join(DATA_ROOT, "labels", SPLIT)
images_dir = os.path.join(DATA_ROOT, "images", SPLIT)

# 이미지 목록
img_files = [p for p in glob.glob(os.path.join(images_dir, "*"))
             if os.path.splitext(p)[1].lower() in IMG_EXTS]

cls_instance_cnt = defaultdict(int)   # 클래스별 인스턴스(박스) 수
cls_image_set    = defaultdict(set)   # 클래스별 등장한 이미지 집합
missing_labels = []
bad_label_files = []

for img_path in img_files:
    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(labels_dir, base + ".txt")
    if not os.path.exists(lbl_path):
        missing_labels.append(img_path)
        continue

    try:
        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            continue

        seen_classes_in_this_image = set()
        for line in lines:
            # YOLO: <class_id> <xc> <yc> <w> <h> [optional extra]
            parts = line.split()
            cid = int(parts[0])
            if nc and (cid < 0 or cid >= nc):
                raise ValueError(f"class_id {cid} out of range [0,{nc-1}] in {lbl_path}")
            cls_instance_cnt[cid] += 1
            seen_classes_in_this_image.add(cid)

        for cid in seen_classes_in_this_image:
            cls_image_set[cid].add(img_path)

    except Exception as e:
        bad_label_files.append((lbl_path, str(e)))

# ==== 출력 ====
total_images = len(img_files)
labeled_images = total_images - len(missing_labels)
total_instances = sum(cls_instance_cnt.values())

print(f"[요약] split: {SPLIT}")
print(f"- 총 이미지 수: {total_images}")
print(f"- 라벨 있는 이미지 수: {labeled_images}")
print(f"- 라벨 누락 이미지 수: {len(missing_labels)}")
print(f"- 총 인스턴스(박스) 수: {total_instances}")
print()

print("클래스별 통계 (class_id | name | 인스턴스 수 | 등장 이미지 수):")
for cid in sorted(set(cls_instance_cnt.keys()) | set(cls_image_set.keys())):
    name = names[cid] if cid < len(names) else f"class_{cid}"
    inst = cls_instance_cnt[cid]
    imgc = len(cls_image_set[cid])
    print(f"{cid:3d} | {name:20s} | {inst:6d} | {imgc:6d}")

if missing_labels:
    print("\n[라벨 누락 예시 5개]")
    for p in missing_labels[:5]:
        print(" -", p)

if bad_label_files:
    print("\n[비정상 라벨 파일 예시]")
    for p, err in bad_label_files[:5]:
        print(f" - {p}: {err}")
