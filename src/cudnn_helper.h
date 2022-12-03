#pragma once

#include <cstring>
#include "utilities.h"
#include <cudnn.h>
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "main.h"

// Credit http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

struct tensor {
	int n;
	int c;
	int h;
	int w;
	float* dev;
	float* host;
	// For kernels, use n as output channel and c as input channel
	tensor(int size, int chan, int height, int width) :
		n(size), c(chan), h(height), w(width) {}

	tensor() :
		n(0), c(0), h(0), w(0) {}
};

cv::Mat load_image(const char* image_path);
void opencv_saveimage(const char* output_filename, float* buffer, int height, int width);
void read_filter(tensor& filter, std::string f_path);
void createTensorDescriptor(cudnnTensorDescriptor_t& descriptor, cudnnTensorFormat_t format, int n, int c, int h, int w);
void readBias(tensor& input, tensor& output, std::string bias_path);
void convolutionalForward(cudnnHandle_t handle, tensor& input, tensor& kernel, tensor& output);
void addBias(cudnnHandle_t handle, tensor& input, tensor& bias);
void reshapeTensor(cudnnHandle_t handle, tensor& t, cudnnTensorFormat_t in_format, cudnnTensorFormat_t out_format);
void logTensor(tensor& t, std::string out_path, std::string name);