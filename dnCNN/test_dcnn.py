### CREDIT TO Kai Zhang: https://github.com/cszn
### https://github.com/cszn/KAIR/blob/master/main_test_dncnn.py
### https://github.com/cszn/KAIR/blob/master/main_test_ffdnet.py

import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from models.dncnn import DnCNN
from models.ffdnet import FFDNet


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

    model_path = 'original_dncnn.pth'
    # From Kai Zhang, dnnn_color_blind is nb=20
    # act_mode determines what the actiavation is, for example: BR == batch norm + ReLU
    # R == ReLU, we can ignore batch norm since utils_bnorm.merge_bn was ran, see https://github.com/cszn/KAIR/blob/master/utils/utils_bnorm.py 
    model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
    #model = FFDNet(in_nc=3, out_nc=3, nc=96, nb=12, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    params = {}

    if not os.path.exists('weights/'):
        os.mkdir('weights/')
    for name, param in model.named_parameters():
        print(name)
        params[name] = param
        name_list = name.split('.')
        if name_list[-1] == 'weight':
            # NOTE Might need to do some reshaping or permuting here?
            param = torch.flatten(param, start_dim=0, end_dim=1) 
            param = torch.flatten(param, start_dim=1, end_dim=2)
        #np.savetxt('weights/' + name_list[1] + '_' + name_list[2] + '.csv', param.numpy(), delimiter=',')

    img_path = 'test.png'
    img = torchvision.io.read_image(img_path) # reads as C,H,W 0-255
    img = (img / 255.)#[:3]# 0-255 -> 0-1
    plt.imshow(img.permute(1,2,0))
    plt.show()

    # img_r = np.genfromtxt('img_log/orig_img_chan0.txt', delimiter=',').reshape(481, 321)
    # img_g = np.genfromtxt('img_log/orig_img_chan1.txt', delimiter=',').reshape(481, 321)
    # img_b = np.genfromtxt('img_log/orig_img_chan2.txt', delimiter=',').reshape(481, 321)

    # kernel_0 = params['model.0.weight']
    # bias_0 = params['model.0.bias']
    # kernel_1 = params['model.2.weight']
    # bias_1 = params['model.2.bias']
    # arr = np.genfromtxt('out3.txt', delimiter=',')
    
    # a_conv0 = np.genfromtxt('img_log/conv2_chan0.txt', delimiter=',').reshape(481, 321)
    # a_convbias0 = np.genfromtxt('img_log/conv2bias_chan0.txt', delimiter=',').reshape(481, 321)
    # a_convrelu0 = np.genfromtxt('img_log/conv2relu_chan0.txt', delimiter=',').reshape(481, 321)
    # a_conv64 = np.genfromtxt('img_log/conv2_chan63.txt', delimiter=',').reshape(481, 321)
    # a_convbias64 = np.genfromtxt('img_log/conv2bias_chan63.txt', delimiter=',').reshape(481, 321)
    # a_convrelu64 = np.genfromtxt('img_log/conv2relu_chan63.txt', delimiter=',').reshape(481, 321)

    # convbias = F.conv2d(img.unsqueeze(dim=0), kernel_0, padding=(1,1), bias=bias_0).squeeze()
    # conv = F.conv2d(img.unsqueeze(dim=0), kernel_0, padding=(1,1)).squeeze()
    # convfull = F.relu(F.conv2d(img.unsqueeze(dim=0), kernel_0, padding=(1,1), bias=bias_0).squeeze())

    # convbias_1 = F.conv2d(convfull.unsqueeze(dim=0), kernel_1, padding=(1,1), bias=bias_1).squeeze()
    # conv_1 = F.conv2d(convfull.unsqueeze(dim=0), kernel_1, padding=(1,1)).squeeze()
    # convfull_1 = F.relu(F.conv2d(convfull.unsqueeze(dim=0), kernel_1, padding=(1,1), bias=bias_1).squeeze())
    # in_tensor = img.unsqueeze(dim=0)

    # for i in range(20):
    #     kernel = params[f"model.{i*2}.weight"]
    #     bias = params[f"model.{i*2}.bias"]
    #     convbias = F.conv2d(in_tensor, kernel, padding=(1,1), bias=bias)
    #     conv = F.conv2d(in_tensor, kernel, padding=(1,1))
    #     convfull = F.relu(F.conv2d(in_tensor, kernel, padding=(1,1), bias=bias))
    #     if i == 10:
    #         import pdb
    #         pdb.set_trace()
    #     in_tensor = convfull if i != 19 else convbias

    #arr = arr.reshape(578, 549)
    # diff = torch.abs(conv[0,0] - torch.from_numpy(arr))
    # plt.imshow(conv[0,0])
    # plt.show()
    # plt.imshow(torch.from_numpy(arr))
    # plt.show()

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    seed = 17
    torch.manual_seed(seed)
    model.to(device)

    #img_path = 'test2.png'
    #img = torchvision.io.read_image(img_path) # reads as C,H,W 0-255
    # plt.imshow(img.permute(1,2,0))
    # plt.show()

    #img = (img / 255.).unsqueeze(dim=0) # 0-255 -> 0-1 and add batch dim
    #noise_level_model = 120
    #sigma = torch.full((1,1,1,1), noise_level_model/255.).float().to(device)
    #out_img = model(img.to(device), sigma)
    out_img = model(img.to(device))

    plt.imshow(out_img.squeeze().permute(1,2,0).cpu())
    plt.show()

    plt.imshow((img - out_img.detach().cpu()).squeeze().permute(1,2,0).cpu())
    plt.show()

    out_img_r = np.genfromtxt('img_log/out_img_chan0.txt', delimiter=',').reshape(481, 321)
    out_img_g = np.genfromtxt('img_log/out_img_chan1.txt', delimiter=',').reshape(481, 321)
    out_img_b = np.genfromtxt('img_log/out_img_chan2.txt', delimiter=',').reshape(481, 321)

    import pdb
    pdb.set_trace()

