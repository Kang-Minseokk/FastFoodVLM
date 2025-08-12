import torch

ckpt = torch.load('yolov5m.pt', map_location='cpu')

# 1. 클래스 개수
if 'nc' in ckpt['model'].yaml:
    print("클래스 개수:", ckpt['model'].yaml['nc'])
else:
    print("클래스 개수:", len(ckpt['model'].names))

# 2. 클래스 이름
if hasattr(ckpt['model'], 'names'):
    print("클래스 이름:", ckpt['model'].names)
elif 'names' in ckpt['model'].yaml:
    print("클래스 이름:", ckpt['model'].yaml['names'])
else:
    print("클래스 이름 정보를 찾을 수 없습니다.")
