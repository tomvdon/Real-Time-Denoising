#pragma once

#include <vector>
#include "scene.h"
#include "main.h"
#include "cudnn.h"

//circular dependecies
struct tensor;
struct layer;

extern int ui_iterations;

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, cudnnHandle_t handle, std::vector<layer>& model, int frame, int iteration, float* workspace);
