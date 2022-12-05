### Pytorch code for dnCNN
Credit to Kai Zhang: https://github.com/cszn/KAIR
Specifically for dnCNN: https://github.com/cszn/KAIR/blob/master/models/network_dncnn.py

### Instructions
1. Use `!pip install requirements.txt` to install neccesary requirements. Note that this installs pytorch for CUDA which requires that you have a CUDA capable device and CUDA 11.7 installed. See https://pytorch.org/get-started/locally/. 
2. Download model weights from https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D, specifically dncnn_color_blind.pth
3. Use `python test_dcnn.py` to run the script, it will show you the original image and then the denoised image
