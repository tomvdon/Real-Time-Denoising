#include "cudnn_helper.h"

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
	float* temp = (float*)malloc(sizeof(float) * 3 * height * width);
	memcpy(temp, buffer, sizeof(float) * 3 * height * width);

	cv::Mat output_image(height, width, CV_32FC3, temp);
	cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR, 3);
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

	free(temp);
}

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
	float* h_arr = (float*)malloc(size);
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
					//int index = i * filter.c * filter.h * filter.w + j * filter.h * filter.w + k * filter.w + l; //NCHW
					int index = i * filter.h * filter.w * filter.c + k * filter.w * filter.c + l * filter.c + j; //NHWC
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

void readBias(int channels, tensor& output, std::string bias_path) {
	// This function reads in a csv file where each row contains a single value that is the bias term for that channel
	// Expects the csv file to have number of lines == channels
	// cudnn accepts tensors where:
	// "Each dimension of the bias tensor A must match the corresponding dimension of the destination tensor C or must be equal to 1."

	std::ifstream fp_in;
	fp_in.open(bias_path);
	if (!fp_in.is_open()) {
		std::cout << "Error reading from file - aborting!" << std::endl;
		throw;
	}

	//int out_size = sizeof(float) * input.n * input.c * input.h * input.w;
	int out_size = sizeof(float) * channels;
	float* h_out = (float*)malloc(out_size);
	float* d_out;
	cudaMalloc(&d_out, out_size);
	cudaMemset(d_out, 0.f, out_size);

	for (int i = 0; i < channels; ++i) {
		std::string line;
		utilityCore::safeGetline(fp_in, line);
		float bias = std::stof(line);
		//std::cout << "Bias " << bias << " in channel " << i << std::endl;
		//for (int j = 0; j < input.n; ++j) {
		//	for (int k = 0; k < input.h; ++k) {
		//		for (int l = 0; l < input.w; ++l) {
		//			int index = j * input.c * input.h * input.w + i * input.h * input.w + k * input.w + l;
		//			h_out[index] = bias;
		//		}
		//	}
		//}
		h_out[i] = bias;
	}

	cudaMemcpy(d_out, h_out, out_size, cudaMemcpyHostToDevice);

	//output.n = input.n;
	//output.c = input.c;
	//output.h = input.h;
	//output.w = input.w;
	output.n = 1;
	output.c = channels;
	output.h = 1;
	output.w = 1;

	output.host = h_out;
	output.dev = d_out;

	fp_in.close();
}

void convolutionalForward(cudnnHandle_t handle, layer& l, tensor& input, tensor& output) {
	auto setup_start = chrono::high_resolution_clock::now();

	// Find workspace size needed
	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
		l.input_desc,
		l.filter_desc,
		l.convolution,
		l.output_desc,
		l.conv_alg.algo,
		&workspace_bytes));
	void* d_workspace{ nullptr };
	cudaMalloc(&d_workspace, workspace_bytes);
	auto setup_end = chrono::high_resolution_clock::now();
	auto setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start);
	std::cout << "workspace setup: " << setup_duration.count() << std::endl;

	setup_start = chrono::high_resolution_clock::now();
	// Do convolution forward
	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnConvolutionForward(handle,
		&alpha,
		l.input_desc,
		input.dev,
		l.filter_desc,
		l.filter.dev,
		l.convolution,
		l.conv_alg.algo,
		d_workspace,
		workspace_bytes,
		&beta,
		l.output_desc,
		output.dev));

	// Free up stuff
	cudaFree(d_workspace);

	setup_end = chrono::high_resolution_clock::now();
	setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start);
	std::cout << "conv forward: " << setup_duration.count() << std::endl;
}

void addBias(cudnnHandle_t handle, layer& l, tensor& input, bool subtract) {
	// Similar to convolutional forward but adds the bias term
	// cudnnAddTensor is done in place

	// 'C' tensor is the output tensor, 'A' tensor is the bias tensor
	const float beta = 1;
	const float alpha = subtract ? -1 : 1;
	checkCUDNN(cudnnAddTensor(handle, &alpha, l.bias_desc, l.bias.dev, &beta, l.output_desc, input.dev));
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
	//float* h_reshaped = (float*)malloc(size);
	cudaMalloc(&d_reshaped, size);

	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnTransformTensor(handle,
		&alpha,
		in_desc,
		t.dev,
		&beta,
		reshaped_desc,
		d_reshaped));

	//cudaMemcpy(h_reshaped, d_reshaped, size, cudaMemcpyDeviceToHost);

	cudaFree(t.dev);
	//free(t.host);

	t.dev = d_reshaped;
	//t.host = h_reshaped;
}

void logTensor(tensor& t, std::string out_path, std::string name) {
	// Logs tensors that are NCHW to txt files, one per channel
	std::cout << t.h << ", " << t.w << std::endl;
	for (int i = 0; i < t.c; ++i) {
		std::ofstream fp_out;
		std::ostringstream path_stream;
		path_stream << out_path + name + "_chan" << i << ".txt";
		std::string f_path = path_stream.str();
		fp_out.open(f_path);
		if (!fp_out.is_open()) {
			std::cout << "Error writing to file - aborting!" << std::endl;
			throw;
		}

		for (int j = 0; j < t.h * t.w; ++j) {
			fp_out << t.host[i * t.w * t.h + j];
			if (j != t.w * t.h - 1) {
				fp_out << ", ";
			}
		}
		fp_out.close();
	}
}

void loadDncnn(cudnnHandle_t handle, std::vector<layer>& model, int height, int width, std::string model_path) {
	for (int i = 0; i < 20; ++i) {
		// Load filter
		int in_chan = 64;
		int out_chan = 64;
		if (i == 0) {
			in_chan = 3;
		}
		if (i == 19) {
			out_chan = 3;
		}

		layer l;

		// Read weights and biases and define descriptors
		l.filter = tensor(out_chan, in_chan, 3, 3);
		std::ostringstream path_stream;
		path_stream << model_path << i * 2 << "_weight.csv";
		std::string f_path = path_stream.str();
		read_filter(l.filter, f_path);
		checkCUDNN(cudnnCreateFilterDescriptor(&l.filter_desc));
		checkCUDNN(cudnnSetFilter4dDescriptor(l.filter_desc,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*format=*/CUDNN_TENSOR_NHWC,
			/*out_channels=*/out_chan,
			/*in_channels=*/in_chan,
			/*kernel_height=*/3,
			/*kernel_width=*/3));

		std::ostringstream bias_stream;
		bias_stream << model_path << i * 2 << "_bias.csv";
		std::string bias_path = bias_stream.str();
		l.bias = tensor();
		readBias(out_chan, l.bias, bias_path);
		createTensorDescriptor(l.bias_desc, CUDNN_TENSOR_NHWC, l.bias.n, l.bias.c, l.bias.h, l.bias.w);

		//Define input out descriptors
		// TODO probably need a better way to do this if the stride/padding make it so that resolution changes
		// Would probably involve using:
			// Calc and define output dimensions
			//int out_n, out_c, out_h, out_w;
			//checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
			//	convolution_descriptor,
			//	input_descriptor,
			//	kernel_descriptor,
			//	&out_n, &out_c, &out_h, &out_w));

		createTensorDescriptor(l.input_desc, CUDNN_TENSOR_NHWC, 1, in_chan, height, width);
		createTensorDescriptor(l.output_desc, CUDNN_TENSOR_NHWC, 1, out_chan, height, width);

		//Define convolution and convolution alg
		checkCUDNN(cudnnCreateConvolutionDescriptor(&l.convolution));
		checkCUDNN(cudnnSetConvolution2dDescriptor(l.convolution,
			/*pad_height=*/1,
			/*pad_width=*/1,
			/*vertical_stride=*/1,
			/*horizontal_stride=*/1,
			/*dilation_height=*/1,
			/*dilation_width=*/1,
			/*mode=*/CUDNN_CROSS_CORRELATION,
			/*computeType=*/CUDNN_DATA_FLOAT));
		int num_algs = 0;
		checkCUDNN(
			cudnnFindConvolutionForwardAlgorithm(handle,
				l.input_desc,
				l.filter_desc,
				l.convolution,
				l.output_desc,
				/*RequestedNumAlgs*/1,
				/*ReturnedNumAlgs*/&num_algs,
				&l.conv_alg));
		// TODO maybe add workspace loading ?
		std::cout << l.conv_alg.algo << std::endl;
		model.push_back(l);
	}
}