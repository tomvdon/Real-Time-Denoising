#include "main.h"
#include "preview.h"
#include "utilities.h"
#include <cstring>

#include <chrono>

#include <cudnn.h>
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"

static std::string startTimeString;

static double time_duration = 0.0;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

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

cv::Mat load_image(const char* image_path) {
	// Credit http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
	cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);

	if (image.empty()) {
		std::cout << "Could not read the image: " << image_path << std::endl;
		return image;
	}

	image.convertTo(image, CV_32FC3);
	//cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);

	return image;
}

void opencv_saveimage(const char* output_filename,
	float* buffer,
	int height,
	int width) {
	// Credit http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
	cv::Mat output_image(height, width, CV_32FC3, buffer);
	cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR, 3 );
	//// Make negative values zero.
	cv::threshold(output_image,
		output_image,
		/*threshold=*/0,
		/*maxval=*/0,
		cv::THRESH_TOZERO);
	//cv::normalize(output_image, output_image, 0, 1, cv::NORM_MINMAX);

	cv::imshow("Display window", output_image);
	int k = cv::waitKey(0); // Wait for a keystroke in the 

	cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
	output_image.convertTo(output_image, CV_8UC3);
	cv::imwrite(output_filename, output_image);
}

//struct filter {
//	int in_channels;
//	int out_channels;
//	int height;
//	int width;
//	int num_els;
//
//	filter(int in_c, int out_c, int h, int w) :
//		in_channels(in_c), out_channels(out_c), height(h), width(w), num_els(in_c * out_c * h * w) {}
//};

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

void read_filter(tensor& filter, std::string f_path) {
	// The way filters are in pytorch: [out_channels, in_channels, height, width]
	// we write these filters to the csv file by flattening the filter to each row (size height*width) structured as row1,row2,row3,etc
	// if we think of the filters as an array of size (out_channels, in_channels) then the order is row1,row2,row3
	// read and store into filter tensor

	std::ifstream fp_in;
	fp_in.open(f_path);
	if (!fp_in.is_open()) {
		std::cout << "Error reading from file - aborting!" << std::endl;
		throw;
	}

	int size = sizeof(float) * filter.n * filter.c * filter.h * filter.w;
	float* h_arr = (float*) malloc(size);
	float* d_arr;
	cudaMalloc(&d_arr, size);
	cudaMemset(d_arr, 0, size);

	for (int i = 0; i < filter.n; ++i) {
		for (int j = 0; j < filter.c; ++j) {
			// Credit https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
			std::string line;
			utilityCore::safeGetline(fp_in, line);
			size_t pos = 0;
			std::string token;
			std::string delimiter = ",";
			for (int k = 0; k < filter.h; ++k) {
				for (int l = 0; l < filter.w; ++l) {
					pos = line.find(delimiter);
					token = line.substr(0, pos);
					float t = std::stof(token);
					int index = i * filter.c * filter.h * filter.w + j * filter.h * filter.w + k * filter.w + l;
					//std::cout << "setting index " << index << " to " << t << std::endl;
					h_arr[index] = t;
					line.erase(0, pos + delimiter.length());
				}
			}
		}
	}

	cudaMemcpy(d_arr, h_arr, sizeof(float) * filter.n * filter.c * filter.h * filter.w, cudaMemcpyHostToDevice);

	filter.dev = d_arr;
	filter.host = h_arr;

	fp_in.close();
}

void createTensorDescriptor(cudnnTensorDescriptor_t& descriptor, cudnnTensorFormat_t format, int n, int c, int h, int w) {
	checkCUDNN(cudnnCreateTensorDescriptor(&descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(descriptor,
		/*format=*/format,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/n,
		/*channels=*/c,
		/*image_height=*/h,
		/*image_width=*/w));
}

void readBias(tensor& input, tensor& output, std::string bias_path) {
	// This function reads in a csv file where each row contains a single value that is the bias term for that channel
	// Expects the csv file to have number of lines == input.c
	// Creates tensor that is size input.n, input.c, input.h, input.w by repeating the bias term for each channel to form a h,w arr
	// Outputs to output tensor

	std::ifstream fp_in;
	fp_in.open(bias_path);
	if (!fp_in.is_open()) {
		std::cout << "Error reading from file - aborting!" << std::endl;
		throw;
	}

	int out_size = sizeof(float) * input.n * input.c * input.h * input.w;
	float* h_out = (float*)malloc(out_size);
	float* d_out;
	cudaMalloc(&d_out, out_size);
	cudaMemset(d_out, 0.f, out_size);

	for (int i = 0; i < input.c; ++i) {
		std::string line;
		utilityCore::safeGetline(fp_in, line);
		float bias = std::stof(line);
		//std::cout << "Bias " << bias << " in channel " << i << std::endl;
		for (int j = 0; j < input.n; ++j) {
			for (int k = 0; k < input.h; ++k) {
				for (int l = 0; l < input.w; ++l) {
					int index = j * input.c * input.h * input.w + i * input.h * input.w + k * input.w + l;
					h_out[index] = bias;
				}
			}
		}
	}

	cudaMemcpy(d_out, h_out, out_size, cudaMemcpyHostToDevice);

	output.n = input.n;
	output.c = input.c;
	output.h = input.h;
	output.w = input.w;

	output.host = h_out;
	output.dev = d_out;

	fp_in.close();
}

void convolutionalForward(cudnnHandle_t handle, tensor& input, tensor& kernel, tensor& output) {
	// Assumes input and out are always NHWC
	// Outputs tensor struct

	// Define input
	cudnnTensorDescriptor_t input_descriptor;
	createTensorDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, input.n, input.c, input.h, input.w);

	// Define kernel
	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*out_channels=*/kernel.n,
		/*in_channels=*/kernel.c,
		/*kernel_height=*/kernel.h,
		/*kernel_width=*/kernel.w));

	// Define convolution
	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
		/*pad_height=*/1,
		/*pad_width=*/1,
		/*vertical_stride=*/1,
		/*horizontal_stride=*/1,
		/*dilation_height=*/1,
		/*dilation_width=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION,
		/*computeType=*/CUDNN_DATA_FLOAT));
	// Calc and define output dimensions
	int out_n, out_c, out_h, out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
		convolution_descriptor,
		input_descriptor,
		kernel_descriptor,
		&out_n, &out_c, &out_h, &out_w));
	cudnnTensorDescriptor_t output_descriptor;
	createTensorDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, out_n, out_c, out_h, out_w);

	// Allocate output
	output.n = out_n;
	output.c = out_c;
	output.h = out_h;
	output.w = out_w;
	//std::cout << "Out size " << out_n << ", " << out_c << ", " << out_h << ", " << out_w << std::endl;
	int output_bytes = out_n * out_c * out_h * out_w * sizeof(float);
	float* d_out { nullptr };
	float* h_out = (float*)malloc(output_bytes);
	cudaMalloc(&d_out, output_bytes);
	cudaMemset(d_out, 0, output_bytes);

	// Find fastest conv algorithim
	cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;
	int num_algs = 0;
	checkCUDNN(
		cudnnFindConvolutionForwardAlgorithm(handle,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			/*RequestedNumAlgs*/1,
			/*ReturnedNumAlgs*/&num_algs,
			&convolution_algorithm));

	// Find workspace size needed
	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
		input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		convolution_algorithm.algo,
		&workspace_bytes));

	void* d_workspace{ nullptr };
	cudaMalloc(&d_workspace, workspace_bytes);

	// Do convolution forward
	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnConvolutionForward(handle,
		&alpha,
		input_descriptor,
		input.dev,
		kernel_descriptor,
		kernel.dev,
		convolution_descriptor,
		convolution_algorithm.algo,
		d_workspace,
		workspace_bytes,
		&beta,
		output_descriptor,
		d_out));
	// Copy back to host
	cudaMemcpy(h_out, d_out, output_bytes, cudaMemcpyDeviceToHost);

	output.host = h_out;
	output.dev = d_out;

	//for (int i = 0; i < 100; ++i) {
	//	std::cout << input.host[i] << ", " << output.host[i] << std::endl;
	//}

	// Free up stuff
	cudaFree(d_workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}

void addBias(cudnnHandle_t handle, tensor& input, tensor& bias) {
	// Similar to convolutional forward but adds the bias term
	// cudnnAddTensor is done in place

	// Define input
	cudnnTensorDescriptor_t input_descriptor;
	createTensorDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, input.n, input.c, input.h, input.w);

	// Define bias
	cudnnTensorDescriptor_t bias_descriptor;
	createTensorDescriptor(bias_descriptor, CUDNN_TENSOR_NCHW, bias.n, bias.c, bias.h, bias.w);

	// 'C' tensor is the output tensor, 'A' tensor is the bias tensor
	const float alpha = 1, beta = 1;
	checkCUDNN(cudnnAddTensor(handle, &alpha, bias_descriptor, bias.dev, &beta, input_descriptor, input.dev));

	// Copy to host tensor
	cudaMemcpy(input.host, input.dev, sizeof(float) * input.n * input.c * input.h * input.w, cudaMemcpyDeviceToHost);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(bias_descriptor);
}

void reshapeTensor(cudnnHandle_t handle, tensor& t, cudnnTensorFormat_t in_format, cudnnTensorFormat_t out_format) {
	//Reshape from in format to out format
	//Assumes tensor has malloced dev and host arrays

	cudnnTensorDescriptor_t in_desc;
	checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,
		/*format=*/in_format,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/t.n,
		/*channels=*/t.c,
		/*image_height=*/t.h,
		/*image_width=*/t.w));
	cudnnTensorDescriptor_t reshaped_desc;
	checkCUDNN(cudnnCreateTensorDescriptor(&reshaped_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(reshaped_desc,
		/*format=*/out_format,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/t.n,
		/*channels=*/t.c,
		/*image_height=*/t.h,
		/*image_width=*/t.w));

	int size = sizeof(float) * t.n * t.c * t.h * t.w;
	float* d_reshaped;
	float* h_reshaped = (float*)malloc(size);
	cudaMalloc(&d_reshaped, size);
	
	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnTransformTensor(handle,
		&alpha,
		in_desc,
		t.dev,
		&beta,
		reshaped_desc,
		d_reshaped));

	cudaMemcpy(h_reshaped, d_reshaped, size, cudaMemcpyDeviceToHost);
	
	cudaFree(t.dev);
	free(t.host);

	t.dev = d_reshaped;
	t.host = h_reshaped;
}

void tryCUDNN() {
	// Credit http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
	std::cout << "Running" << std::endl;
	cudnnHandle_t handle;
	cudnnCreate(&handle);

	//Load input
	cv::Mat image = load_image("C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\img\\tensorflow.png");
	std::cout << "Image has shape " << image.rows << ", " << image.cols << std::endl;
	int image_bytes = 1 * 3 * image.rows * image.cols * sizeof(float); 
	// Make 0-255 -> 0-1
	float* img = image.ptr<float>(0);
	for (int i = 0; i < 3 * 578 * 549; ++i) {
		img[i] /= 255.f;
	}

	float* d_input{ nullptr };
	cudaMalloc(&d_input, image_bytes);
	cudaMemcpy(d_input, img, image_bytes, cudaMemcpyHostToDevice);

	//Reshape output to NCHW instead of NHWC
	//int out_size = output.n * output.c * output.h * output.w * sizeof(float);
	float* d_reshaped_in;
	cudaMalloc(&d_reshaped_in, image_bytes);
	cudaMemset(d_reshaped_in, 0, image_bytes);

	cudnnTensorDescriptor_t reshaped_desc;
	checkCUDNN(cudnnCreateTensorDescriptor(&reshaped_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(reshaped_desc,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/3,
		/*image_height=*/image.rows,
		/*image_width=*/image.cols));
	cudnnTensorDescriptor_t in_desc;
	checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
	checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/3,
		/*image_height=*/image.rows,
		/*image_width=*/image.cols));

	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnTransformTensor(handle,
		&alpha,
		in_desc,
		d_input,
		&beta,
		reshaped_desc,
		d_reshaped_in));

	float* h_reshaped_in = (float*)malloc(image_bytes);
	cudaMemcpy(h_reshaped_in, d_reshaped_in, image_bytes, cudaMemcpyDeviceToHost);

	tensor input = tensor(1, 3, image.rows, image.cols);
	input.dev = d_reshaped_in;
	input.host = h_reshaped_in;

	// Load filter
	std::cout << "Loading filter ..." << std::endl;
	tensor kernel = tensor(64, 3, 3, 3);
	std::string f_path = "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\weights\\0_weight.csv";
	read_filter(kernel, f_path);
	std::cout << "Succesful load" << std::endl;

	// Conv forward
	tensor output = tensor();
	convolutionalForward(handle, input, kernel, output);

	// Add bias
	std::string bias_path = "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\weights\\0_bias.csv";
	tensor bias = tensor();
	readBias(output, bias, bias_path);
	addBias(handle, output, bias);

	// write to txt file to double check stuff
	std::string out_path = "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\out3.txt";
	std::ofstream fp_out;
	fp_out.open(out_path);
	
	if (!fp_out.is_open()) {
		std::cout << "Error writing to file - aborting!" << std::endl;
		throw;
	}

	for (int i = 0; i < output.w * output.h; ++i) {
		fp_out << output.host[i];
		if (i != output.h * output.w - 1) {
			fp_out << ", ";
		}
	}
	fp_out.close();
	// Save img
	reshapeTensor(handle, output, CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NHWC);
	opencv_saveimage("C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\img\\out.png", output.host, image.rows, image.cols);

	// Free up stuff
	cudaFree(d_input);
	cudaFree(input.dev);

	cudaFree(kernel.dev);
	free(kernel.host);

	cudaFree(output.dev);
	free(output.host);

	cudaFree(bias.dev);
	free(bias.host);
}

void dnCNN() {
	cudnnHandle_t handle;
	cudnnCreate(&handle);

	//Load input
	cv::Mat image = load_image("C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\test.png");
	std::cout << "Image has shape " << image.rows << ", " << image.cols << std::endl;
	int image_bytes = 1 * 3 * image.rows * image.cols * sizeof(float);
	// Make 0-255 -> 0-1
	float* img = image.ptr<float>(0);
	for (int i = 0; i <= 3 * image.rows * image.cols; ++i) {
		img[i] /= 255.f;
	}
	
	// For some reason the pointer from opencv doesnt liked to be freed?
	float* h_input = (float*)malloc(image_bytes);
	memcpy(h_input, img, image_bytes); 

	cv::imshow("Display window", image);
	int k = cv::waitKey(0); // Wait for a keystroke in the window

	float* d_input{ nullptr };
	cudaMalloc(&d_input, image_bytes);
	cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice);

	tensor input = tensor(1, 3, image.rows, image.cols);
	input.dev = d_input;
	input.host = h_input;

	reshapeTensor(handle, input, CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW);

	//Reshape output to NCHW instead of NHWC
	//float* d_reshaped_in;
	//cudaMalloc(&d_reshaped_in, image_bytes);
	//cudaMemset(d_reshaped_in, 0, image_bytes);
	//cudnnTensorDescriptor_t reshaped_desc;
	//checkCUDNN(cudnnCreateTensorDescriptor(&reshaped_desc));
	//checkCUDNN(cudnnSetTensor4dDescriptor(reshaped_desc,
	//	/*format=*/CUDNN_TENSOR_NCHW,
	//	/*dataType=*/CUDNN_DATA_FLOAT,
	//	/*batch_size=*/1,
	//	/*channels=*/3,
	//	/*image_height=*/image.rows,
	//	/*image_width=*/image.cols));
	//cudnnTensorDescriptor_t in_desc;
	//checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
	//checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,
	//	/*format=*/CUDNN_TENSOR_NHWC,
	//	/*dataType=*/CUDNN_DATA_FLOAT,
	//	/*batch_size=*/1,
	//	/*channels=*/3,
	//	/*image_height=*/image.rows,
	//	/*image_width=*/image.cols));
	//const float alpha = 1, beta = 0;
	//checkCUDNN(cudnnTransformTensor(handle,
	//	&alpha,
	//	in_desc,
	//	d_input,
	//	&beta,
	//	reshaped_desc,
	//	d_reshaped_in));
	//float* h_reshaped_in = (float*)malloc(image_bytes);
	//cudaMemcpy(h_reshaped_in, d_reshaped_in, image_bytes, cudaMemcpyDeviceToHost);

	//tensor input = tensor(1, 3, image.rows, image.cols);
	//input.dev = d_reshaped_in;
	//input.host = h_reshaped_in;

	for (int i = 0; i < 19; ++i) {
		std::cout << "Layer " << i+1 << std::endl;

		// Load filter
		// TODO Figure out filter sizes and somehow read that or hard code it
		// kernel should not always be this shape
		int in_chan = 64;
		int out_chan = 64;
		if (i == 0) {
			in_chan = 3;
		}
		if (i == 18) {
			out_chan = 3;
		}
		tensor kernel = tensor(out_chan, in_chan, 3, 3);
		std::ostringstream path_stream;
		path_stream << "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\weights\\" << i * 2 << "_weight.csv";
		std::string f_path = path_stream.str();
		read_filter(kernel, f_path);
		std::cout << "Read filter" << std::endl;

		// Conv forward
		tensor output = tensor();
		convolutionalForward(handle, input, kernel, output);
		std::cout << "Conv forward" << std::endl;

		// Add bias
		std::ostringstream bias_stream;
		bias_stream << "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\weights\\" << i * 2 << "_bias.csv";
		std::string bias_path = bias_stream.str();
		tensor bias = tensor();
		readBias(output, bias, bias_path);
		addBias(handle, output, bias);
		std::cout << "Add Bias" << std::endl;

		// ReLU
		// TODO closer look at documentation, it says HW-packed is faster. Does this mean NHWC is faster than NCHW?
		// Can be done in place
		const float alpha = 1, beta = 0;
		cudnnActivationDescriptor_t activation;
		checkCUDNN(cudnnCreateActivationDescriptor(&activation));
		checkCUDNN(cudnnSetActivationDescriptor(activation,
			CUDNN_ACTIVATION_RELU,
			CUDNN_PROPAGATE_NAN,
			0.0));
		cudnnTensorDescriptor_t output_descriptor;
		createTensorDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, output.n, output.c, output.h, output.w);
		if (i != 18) {
			std::cout << "ReLU" << std::endl;
			checkCUDNN(cudnnActivationForward(handle,
				activation,
				&alpha,
				output_descriptor,
				output.dev,
				&beta,
				output_descriptor,
				output.dev));
		}

		std::cout << "Output_before: " << output.n << ", " << output.c << ", " << output.h << ", " << output.w << std::endl;
		// Free input stuff, set input to output
		cudaFree(input.dev);
		free(input.host);
		input.n = output.n;
		input.c = output.c;
		input.h = output.h;
		input.w = output.w;
		input.dev = output.dev;
		input.host = output.host;

		cudnnDestroyTensorDescriptor(output_descriptor);
		cudnnDestroyActivationDescriptor(activation);
		cudaFree(kernel.dev);
		cudaFree(bias.dev);
		free(kernel.host);
		free(bias.host);
	}
	cudaMemcpy(input.host, input.dev, sizeof(float) * input.n * input.c * input.h * input.w, cudaMemcpyDeviceToHost);
	std::cout << "Final output shape is " << input.n << ", " << input.c << ", " << input.h << ", " << input.w << std::endl;

	// Save image as 3 txt channels, one for RGB
	std::string red_path = "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\red.txt";
	std::string green_path = "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\green.txt";
	std::string blue_path = "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\blue.txt";
	
	std::ofstream fp_out;
	fp_out.open(red_path);
	if (!fp_out.is_open()) {
		std::cout << "Error writing to file - aborting!" << std::endl;
		throw;
	}
	for (int i = 0; i < input.w * input.h; ++i) {
		fp_out << input.host[i];
		if (i != input.h * input.w - 1) {
			fp_out << ", ";
		}
	}
	fp_out.close();

	fp_out.open(green_path);
	if (!fp_out.is_open()) {
		std::cout << "Error writing to file - aborting!" << std::endl;
		throw;
	}
	for (int i = 0; i < input.w * input.h; ++i) {
		fp_out << input.host[i + input.w * input.h];
		if (i != input.h * input.w - 1) {
			fp_out << ", ";
		}
	}
	fp_out.close();

	fp_out.open(blue_path);
	if (!fp_out.is_open()) {
		std::cout << "Error writing to file - aborting!" << std::endl;
		throw;
	}
	for (int i = 0; i < input.w * input.h; ++i) {
		fp_out << input.host[i + input.w * input.h * 2];
		if (i != input.h * input.w - 1) {
			fp_out << ", ";
		}
	}
	fp_out.close();

	reshapeTensor(handle, input, CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NHWC);
	opencv_saveimage("C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\img\\noise.png", input.host, input.h, input.w);

	for (int i = 0; i < 3 * image.rows * image.cols; ++i) {
		img[i] -= input.host[i];
	}
	opencv_saveimage("C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\img\\denoised.png", img, input.h, input.w);

	cudaFree(input.dev);
	free(input.host);
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	//tryCUDNN();
	dnCNN();
	std::cout << "Success!" << std::endl;
}

/*
int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];

	// Load scene file
	scene = new Scene(sceneFile);

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;

	//renderState->iterations = 100;

	Camera& cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	// Initialize CUDA and GL components
	init();

	// Initialize ImGui Data
	InitImguiData(guiData);
	InitDataContainer(guiData);

	// GLFW main loop
	mainLoop();

	return 0;
}*/

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
			img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
	if (camchanged) {
		iteration = 0;
		Camera& cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		pathtraceFree();
		pathtraceInit(scene);
	}

	if (iteration < renderState->iterations) {

		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		auto start = std::chrono::steady_clock::now();


		int frame = 0;
		pathtrace(pbo_dptr, frame, iteration);


		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		time_duration += elapsed_seconds.count();
		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	else {
		std::cout << "elapsed time to compute: " << time_duration << "s\n";
		saveImage();
		pathtraceFree();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			saveImage();
			break;
		case GLFW_KEY_SPACE:
			camchanged = true;
			renderState = &scene->state;
			Camera& cam = renderState->camera;
			cam.lookAt = ogLookAt;
			break;
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow())
	{
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
		cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}


