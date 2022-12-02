# Proj 3 CUDA Path Tracer - Instructions

To Save GBuffers
Install Boost:
https://www.boost.org/doc/libs/1_80_0/more/getting_started/windows.html  

Build with serilziation, once you get to the b2 stage run:  
`b2 --build-type=complete stage`
In your installer:
Add Boost_INCLUDE_DIR_, Boost_LIBRARY_DIR, Boost_ROOT to system path and set in CMAKE.

Build CMAKE  
Run VS  
If library is still missing link in VS Code properties ![](./img/prop.png)   
Build in Relase mode

## To Run
From the build directory run   
~python ..\run_datagen.py --path ..\scenese\ --a NUM_CAMERA_ANGLE --loop (0 or 1)~

Path is the path to the directory of scene files, --a is the number of camera angles to save per scene, -loop is whether to get saving training data