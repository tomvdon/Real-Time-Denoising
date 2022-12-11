[Pitch](https://docs.google.com/presentation/d/1y1yV0J7CyVD8jc_lO0PWT8My7wSFG6huEi9DV5qEQxY/edit#slide=id.p)
# Real Time Denoising and Upscaling for CUDA Pathtracer
This project aims to create a real time denoiser and upscaler based on [DNCNN](https://arxiv.org/abs/1608.03981) and [VDSR](https://arxiv.org/abs/1511.04587) deep learing. In addition, it includes data generation pipeline, scene creation, and utility programs.

## Denoising
![](img/low_spp.png)  ![](img/denoised_spp.png)  
Currently the pathtracer iterates to a specific spp level and then denoises the image, this will be improved in the coming days to have multiple denoising operations.

## Data Generation
![](img/ms2gif.gif)  
Based in the data-gen branch, this program saves GBuffer data as well as path traced images at different sample rates for use in training. 
Use:  
`..\run_datagen.py --path ..\scenes\ --a NUM_CAMERA_ANGLE --loop (0 or 1)`
We were able to generate a dataset of _ images acros _ scenes which is then used to fine tune the denoising model.

## Denoising with DNCNN
We do denoising in the path tracer using DNCNN built with cuDNN. Firstly we run a python script to load a pytorch check point with trained model weights. We write these model weights to .txt files to be read by the path tracer. Note that we use condense batch normaization layers into the convolutional layers after training to reduce computation (done in pytorch). On the path tracer start up, these .txt files are read and loaded into cuDNN kernels and biases. Then an image generated from the path tracer can be ran through the weights and biases and outputted back into the PBO to be dispayed.

To use: Make sure denoise_on is true in pathtrace.cu.

## Performance and Analysis  

![](img/denoising_graph.png)  
Current performance is unoptimized and poor, taking ~12 seconds to denoise an image. This will be improved in the coming days.

![cd007bb848a6e9c4e12cf626d2add57](https://user-images.githubusercontent.com/54868517/205754761-267ce7a5-7e76-404d-8fbb-4dd6abfb63d9.png)

performance analysis for Bounding Volumn Hierarchy
