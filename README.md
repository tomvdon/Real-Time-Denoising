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

## Performance and Analysis  

![](img/denoising_graph.png)  
Current performance is unoptimized and poor, taking ~12 seconds to denoise an image. This will be improved in the coming days.