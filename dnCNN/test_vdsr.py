import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
from models.vdsr import VDSR
from models.utils_bnorm import merge_bn, tidy_sequential, add_bn

# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def rgb2ycbcr(image: np.ndarray, use_y_channel: bool = False) -> np.ndarray:
    """Implementation of rgb2ycbcr function in Matlab under Python language.
    Args:
        image (np.ndarray): Image input in RGB format.
        use_y_channel (bool): Extract Y channel separately. Default: ``False``.
    Returns:
        ndarray: YCbCr image array data.
    """

    if use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    seed = 17
    torch.manual_seed(seed)

    model = VDSR().to(device)
    checkpoint = torch.load("vdsr-weights.pth.tar", map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    img_path = 'denoised.png'
    img = torchvision.io.read_image(img_path) # reads as C,H,W 0-255
    img = (img / 255.)#[:3]# 0-255 -> 0-1
    plt.imshow(img.permute(1,2,0))
    plt.show()

    img_upscale = F.interpolate(img, scale_factor=2, mode='bicubic', align_corners=False)
    plt.imshow(img_upscale.permute(1,2,0))
    plt.show()

    img_y = rgb2ycbcr(img.permute(1,2,0).numpy(), use_y_channel=False)


    with torch.no_grad():
        model_out = model(torch.tensor(img_y[..., -1]).unsqueeze(dim=0).to(device)).clamp(0.0, 1.0)

    import pdb
    pdb.set_trace()