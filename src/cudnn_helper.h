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

struct layer {
	tensor filter;
	tensor bias;
	cudnnTensorDescriptor_t input_desc;
	cudnnTensorDescriptor_t output_desc;
	cudnnTensorDescriptor_t bias_desc;
	cudnnFilterDescriptor_t filter_desc;
	cudnnConvolutionDescriptor_t convolution;
	cudnnConvolutionFwdAlgoPerf_t conv_alg;
	//cudnnConvolutionFwdAlgo_t conv_alg;
};

struct fusionTensor {
	cudnnBackendDescriptor_t desc;
	cudnnDataType_t dtype;
	int64_t* dim;
	int64_t* stride;
	int64_t id;
	int64_t alignment;
};
struct fusionConv {
	cudnnBackendDescriptor_t conv_desc;
	cudnnBackendDescriptor_t conv_op;
	fusionTensor in;
	fusionTensor out;
	fusionTensor bias;
	fusionTensor filter;
	cudnnDataType_t dtype;
	cudnnConvolutionMode_t convMode;
	int64_t dims;
	int64_t* pad;
	int64_t* stride;
	int64_t* dilation;
};
struct fusionLayer {
	fusionTensor input;
	fusionTensor output;
	fusionTensor bias;
	fusionConv conv;
};

cv::Mat load_image(const char* image_path);
void opencv_saveimage(const char* output_filename, float* buffer, int height, int width);
void read_filter(tensor& filter, std::string f_path);
void createTensorDescriptor(cudnnTensorDescriptor_t& descriptor, cudnnTensorFormat_t format, int n, int c, int h, int w);
void readBias(int channels, tensor& output, std::string bias_path);
void convolutionalForward(cudnnHandle_t handle, layer& l, tensor& input, tensor& output);
void addBias(cudnnHandle_t handle, layer& l, tensor& bias, bool subtract);
void reshapeTensor(cudnnHandle_t handle, tensor& t, cudnnTensorFormat_t in_format, cudnnTensorFormat_t out_format);
void logTensor(tensor& t, std::string out_path, std::string name);
void loadDncnn(cudnnHandle_t handle, std::vector<layer>& model, int height, int width, std::string model_path);
void createFusionTensorDescriptor(cudnnBackendDescriptor_t& desc, int n, int c, int h, int w, uint64_t id, bool vir);
void createFusionConvDescriptor(fusionConv conv);
void createFusionConvOp(fusionConv conv);
void createFusionBiasDescriptor(cudnnBackendDescriptor_t& desc, bool subtract);
void createFusionBiasOp(cudnnBackendDescriptor_t& desc, cudnnBackendDescriptor_t& in_desc,
	cudnnBackendDescriptor_t& bias_desc, cudnnBackendDescriptor_t& out_desc, cudnnBackendDescriptor_t& pointwise, bool subtract);
void generateStrides(vector<int64_t>& dimA, vector<int64_t>& strideA, int nbDims, cudnnTensorFormat_t filterFormat);