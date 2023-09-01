import torch
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5m, yolov5l, yolov5x, custom
freeze = 10
freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
total_params = 0
for k, v in model.named_parameters():
    v.requires_grad = True  # train all layers
    if any(x in k for x in freeze) and False:
        print(f'freezing {k}')
        v.requires_grad = False
    else: 
        tup_shape = list(v.detach().cpu().numpy().shape)
        res = 1
        for elem in tup_shape:
            res *= elem
        if res != 1:
            total_params += res

print(total_params)