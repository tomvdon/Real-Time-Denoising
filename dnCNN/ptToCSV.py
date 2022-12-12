import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from models.dncnn import DnCNN
from models.ffdnet import FFDNet
from models.utils_bnorm import merge_bn, tidy_sequential, add_bn
import argparse

# add arguments to the parser for model weight dir and model checkpoint dir
parser = argparse.ArgumentParser()
parser.add_argument('--weight_dir', type=str, default='weights/', help='Weight output directory (CSV)')
parser.add_argument('--cktpt_dir', type=str, default='weights/checkpoint.pt', help='Model Checkpoint Directory (pth)')
parser.add_argument('--num_layers', type=int, default=20, help='Number of layers in the model')
parser.add_argument('--batch_norm', type=bool, default=False, help='Boolean variable that indicates batch norm or not')
parser.add_argument('--gbuff', type=bool, default=False, help='Boolean variable that indicates gbuffer or not (9 or 3 input channels)')

def main():
    args = parser.parse_args()
    model_path = args.cktpt_dir
    output_path = args.weight_dir
    num_layers = args.num_layers
    batch_norm = args.batch_norm
    gbuff = args.gbuff

    model = DnCNN(in_nc=9 if gbuff else 3, out_nc=3, nc=64, nb=num_layers, act_mode='BR' if batch_norm else 'R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    if batch_norm:
        merge_bn(model)
        tidy_sequential(model)

    for k, v in model.named_parameters():
        v.requires_grad = False
    params = {}

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    i = 0
    for name, param in model.named_parameters():
        print(name)
        params[name] = param
        name_list = name.split('.')
        if name_list[-1] == 'weight':
            param = torch.flatten(param, start_dim=0, end_dim=1) 
            param = torch.flatten(param, start_dim=1, end_dim=2)
        np.savetxt(output_path + str(int(i/2)) + '_' + name_list[2] + '.csv', param.numpy(), delimiter=',')
        i += 1

if __name__ == '__main__':
    main()
