### CREDIT TO Kai Zhang: https://github.com/cszn
### https://github.com/cszn/KAIR/blob/master/main_test_dncnn.py

import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from models.dncnn import DnCNN

if __name__ == '__main__':
    # Runs pretrained dnCNN on test img

    model_path = 'dncnn_color_blind.pth'
    # From Kai Zhang, dnnn_color_blind is nb=20
    # act_mode determines what the actiavation is, for example: BR == batch norm + ReLU
    # R == ReLU, we can ignore batch norm since utils_bnorm.merge_bn was ran, see https://github.com/cszn/KAIR/blob/master/utils/utils_bnorm.py 
    model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    params = {}
    for name, param in model.named_parameters():
        print(name)
        params[name] = param

    import pdb
    pdb.set_trace()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    seed = 17
    torch.manual_seed(seed)
    model.to(device)

    img_path = 'test.png'
    img = torchvision.io.read_image(img_path) # reads as C,H,W 0-255

    plt.imshow(img.permute(1,2,0))
    plt.show()

    img = (img / 255.).unsqueeze(dim=0) # 0-255 -> 0-1 and add batch dim

    out_img = model(img.to(device))

    plt.imshow(out_img.squeeze().permute(1,2,0).cpu())
    plt.show()

