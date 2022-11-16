### CREDIT TO Kai Zhang: https://github.com/cszn
### https://github.com/cszn/KAIR/blob/master/main_test_dncnn.py

import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from models.dncnn import DnCNN


if __name__ == '__main__':
    # Runs pretrained dnCNN on test img

    # sigma = torch.full((1,1,1,1), 15/255.)
    # m = sigma.repeat(1, 1, 8, 12)
    # temp = torch.arange(0, 96).view(8, 12)
    # temp = torch.stack([temp, temp + 200, temp - 200], dim=0)
    # temp = temp.unsqueeze(dim=0)
    # input_view = temp.contiguous().view(1, 3, 4, 2, 6, 2)
    # unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    # out = unshuffle_out.view(1, 12, 4, 6)

    # restored = F.pixel_shuffle(out, 2)

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
    # if not os.path.exists('weights/'):
    #     os.mkdir('weights/')
    # for name, param in model.named_parameters():
    #     print(name)
    #     params[name] = param
    #     name_list = name.split('.')
    #     if name_list[-1] == 'weight':
    #         # NOTE Might need to do some reshaping or permuting here?
    #         param = torch.flatten(param, start_dim=0, end_dim=1) 
    #         param = torch.flatten(param, start_dim=1, end_dim=2)
    #     np.savetxt('weights/' + name_list[1] + '_' + name_list[2] + '.csv', param.numpy(), delimiter=',')

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

