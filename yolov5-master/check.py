import torch

ckpt = torch.load('yolov5s.pt', map_location='cpu')
print(ckpt['model'].yaml['nc'])