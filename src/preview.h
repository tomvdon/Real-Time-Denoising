#pragma once

extern GLuint pbo;

std::string currentTimeString();
bool init();
void mainLoop(int max_angles_per_scene);

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);