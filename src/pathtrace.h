#pragma once

#include <vector>
#include "scene.h"
#include "main.h"

struct tensor;

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, std::vector<tensor>& filters, std::vector<tensor>& biases);
