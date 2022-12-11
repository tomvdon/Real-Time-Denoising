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
#include <chrono>

//cuDNN stuff
// TODO maybe have a global activation and convolution descriptor?
// or at least in layer ?
static cudnnHandle_t handle;
static std::vector<layer> model;
static float* conv_workspace;

bool ui_denoise = false;
int ui_iterations = 1;
bool use_gbuff = false;
int num_layers = 20;

//void tryCUDNN() {
//	// Credit http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
//	std::cout << "Running" << std::endl;
//	cudnnHandle_t handle;
//	cudnnCreate(&handle);
//
//	//Load input
//	cv::Mat image = load_image("C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\img\\tensorflow.png");
//	std::cout << "Image has shape " << image.rows << ", " << image.cols << std::endl;
//	int image_bytes = 1 * 3 * image.rows * image.cols * sizeof(float); 
//	// Make 0-255 -> 0-1
//	float* img = image.ptr<float>(0);
//	for (int i = 0; i < 3 * 578 * 549; ++i) {
//		img[i] /= 255.f;
//	}
//
//	float* d_input{ nullptr };
//	cudaMalloc(&d_input, image_bytes);
//	cudaMemcpy(d_input, img, image_bytes, cudaMemcpyHostToDevice);
//
//	//Reshape output to NCHW instead of NHWC
//	//int out_size = output.n * output.c * output.h * output.w * sizeof(float);
//	float* d_reshaped_in;
//	cudaMalloc(&d_reshaped_in, image_bytes);
//	cudaMemset(d_reshaped_in, 0, image_bytes);
//
//	cudnnTensorDescriptor_t reshaped_desc;
//	checkCUDNN(cudnnCreateTensorDescriptor(&reshaped_desc));
//	checkCUDNN(cudnnSetTensor4dDescriptor(reshaped_desc,
//		/*format=*/CUDNN_TENSOR_NCHW,
//		/*dataType=*/CUDNN_DATA_FLOAT,
//		/*batch_size=*/1,
//		/*channels=*/3,
//		/*image_height=*/image.rows,
//		/*image_width=*/image.cols));
//	cudnnTensorDescriptor_t in_desc;
//	checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
//	checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,
//		/*format=*/CUDNN_TENSOR_NHWC,
//		/*dataType=*/CUDNN_DATA_FLOAT,
//		/*batch_size=*/1,
//		/*channels=*/3,
//		/*image_height=*/image.rows,
//		/*image_width=*/image.cols));
//
//	const float alpha = 1, beta = 0;
//	checkCUDNN(cudnnTransformTensor(handle,
//		&alpha,
//		in_desc,
//		d_input,
//		&beta,
//		reshaped_desc,
//		d_reshaped_in));
//
//	float* h_reshaped_in = (float*)malloc(image_bytes);
//	cudaMemcpy(h_reshaped_in, d_reshaped_in, image_bytes, cudaMemcpyDeviceToHost);
//
//	tensor input = tensor(1, 3, image.rows, image.cols);
//	input.dev = d_reshaped_in;
//	input.host = h_reshaped_in;
//
//	// Load filter
//	std::cout << "Loading filter ..." << std::endl;
//	tensor kernel = tensor(64, 3, 3, 3);
//	std::string f_path = "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\weights\\0_weight.csv";
//	read_filter(kernel, f_path);
//	std::cout << "Succesful load" << std::endl;
//
//	// Conv forward
//	tensor output = tensor();
//	convolutionalForward(handle, input, kernel, output);
//
//	// Add bias
//	std::string bias_path = "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\weights\\0_bias.csv";
//	tensor bias = tensor();
//	readBias(output, bias, bias_path);
//	addBias(handle, output, bias);
//
//	// write to txt file to double check stuff
//	std::string out_path = "C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\dnCNN\\out3.txt";
//	std::ofstream fp_out;
//	fp_out.open(out_path);
//	
//	if (!fp_out.is_open()) {
//		std::cout << "Error writing to file - aborting!" << std::endl;
//		throw;
//	}
//
//	for (int i = 0; i < output.w * output.h; ++i) {
//		fp_out << output.host[i];
//		if (i != output.h * output.w - 1) {
//			fp_out << ", ";
//		}
//	}
//	fp_out.close();
//	// Save img
//	reshapeTensor(handle, output, CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NHWC);
//	opencv_saveimage("C:\\Users\\Tom\\CIS5650\\Real-Time-Denoising-And-Upscaling\\img\\out.png", output.host, image.rows, image.cols);
//
//	// Free up stuff
//	cudaFree(d_input);
//	cudaFree(input.dev);
//
//	cudaFree(kernel.dev);
//	free(kernel.host);
//
//	cudaFree(output.dev);
//	free(output.host);
//
//	cudaFree(bias.dev);
//	free(bias.host);
//}
//
//void dnCNN_old() {
//	cudnnHandle_t handle;
//	cudnnCreate(&handle);
//	std::cout << sizeof(float) << std::endl;
//	std::string img_path = "C:\\Users\\ryanr\\Desktop\\Penn\\22-23\\CIS565\\Real-Time-Denoising-And-Upscaling\\dnCNN\\";
//
//	//Load input
//	cv::Mat image = load_image((img_path + "test.png").c_str());
//	std::cout << "Image has shape " << image.rows << ", " << image.cols << std::endl;
//	int image_bytes = 1 * 3 * image.rows * image.cols * sizeof(float);
//	// Make 0-255 -> 0-1
//	float* img = image.ptr<float>(0);
//	for (int i = 0; i < 3 * image.rows * image.cols; ++i) {
//		img[i] /= 255.f;
//	}
//	
//	// For some reason the pointer from opencv doesnt liked to be freed?
//	float* h_input = (float*)malloc(image_bytes);
//	memcpy(h_input, img, image_bytes); 
//
//	cv::imshow("Display window", image);
//	int k = cv::waitKey(0); // Wait for a keystroke in the window
//
//	float* d_input{ nullptr };
//	cudaMalloc(&d_input, image_bytes);
//	cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice);
//
//	tensor input = tensor(1, 3, image.rows, image.cols);
//	input.dev = d_input;
//	input.host = h_input;
//
//	reshapeTensor(handle, input, CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW);
//
//	logTensor(input, img_path + "img_log\\", "orig_img");
//
//	for (int i = 0; i < 20; ++i) {
//		// Load filter
//		// TODO Figure out way to describe the filter sizes across a network, for now its hardcoded since dncnn is simple
//		int in_chan = 64;
//		int out_chan = 64;
//		if (i == 0) {
//			in_chan = 3;
//		}
//		if (i == 19) {
//			out_chan = 3;
//		}
//		tensor kernel = tensor(out_chan, in_chan, 3, 3);
//		std::ostringstream path_stream;
//		path_stream << img_path + "weights\\" << i * 2 << "_weight.csv";
//		std::string f_path = path_stream.str();
//		read_filter(kernel, f_path);
//		std::cout << "Read filter" << std::endl;
//
//		// Conv forward
//		tensor output = tensor();
//		convolutionalForward(handle, input, kernel, output);
//		std::cout << "Conv forward" << std::endl;
//
//		// Add bias, done in place
//		std::ostringstream bias_stream;
//		bias_stream << img_path + "weights\\" << i * 2 << "_bias.csv";
//		std::string bias_path = bias_stream.str();
//		tensor bias = tensor();
//		readBias(out_chan, bias, bias_path);
//		addBias(handle, output, bias);
//		std::cout << "Add Bias" << std::endl;
//
//
//		// ReLU
//		// TODO closer look at documentation, it says HW-packed is faster. Does this mean NHWC is faster than NCHW?
//		// Can be done in place
//		const float alpha = 1, beta = 0;
//		cudnnActivationDescriptor_t activation;
//		checkCUDNN(cudnnCreateActivationDescriptor(&activation));
//		checkCUDNN(cudnnSetActivationDescriptor(activation,
//			CUDNN_ACTIVATION_RELU,
//			CUDNN_PROPAGATE_NAN,
//			0.0));
//		cudnnTensorDescriptor_t output_descriptor;
//		createTensorDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, output.n, output.c, output.h, output.w);
//		if (i != 19) {
//			std::cout << "ReLU" << std::endl;
//			checkCUDNN(cudnnActivationForward(handle,
//				activation,
//				&alpha,
//				output_descriptor,
//				output.dev,
//				&beta,
//				output_descriptor,
//				output.dev));
//		}
//
//		std::cout << "Output_before: " << output.n << ", " << output.c << ", " << output.h << ", " << output.w << std::endl;
//		// Free input stuff, set input to output
//		cudaFree(input.dev);
//		free(input.host);
//		input.n = output.n;
//		input.c = output.c;
//		input.h = output.h;
//		input.w = output.w;
//		input.dev = output.dev;
//		input.host = output.host;
//
//		cudnnDestroyTensorDescriptor(output_descriptor);
//		cudnnDestroyActivationDescriptor(activation);
//		cudaFree(kernel.dev);
//		cudaFree(bias.dev);
//		free(kernel.host);
//		free(bias.host);
//	}
//	cudaMemcpy(input.host, input.dev, sizeof(float) * input.n * input.c * input.h * input.w, cudaMemcpyDeviceToHost);
//	std::cout << "Final output shape is " << input.n << ", " << input.c << ", " << input.h << ", " << input.w << std::endl;
//
//	//Make sure the folder exists
//	logTensor(input, img_path + "img_log\\", "out_img");
//
//	//Reshape tensor back to NHWC since img is in that format
//	reshapeTensor(handle, input, CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NHWC);
//	opencv_saveimage((img_path + "noise.png").c_str(), input.host, input.h, input.w);
//
//	for (int i = 0; i < 3 * image.rows * image.cols; ++i) {
//		if (i < 20) {
//			std::cout << img[i] << " - " << input.host[i] << std::endl;
//		}
//		img[i] -= input.host[i];
//	}
//
//	opencv_saveimage((img_path + "denoised.png").c_str(), img, input.h, input.w);
//
//	tensor temp = tensor(1, 3, image.rows, image.cols);
//	float* h_temp = (float*) malloc(image_bytes);
//	float* d_temp;
//	cudaMalloc(&d_temp, image_bytes);
//
//	memcpy(h_temp, img, image_bytes);
//	cudaMemcpy(d_temp, h_temp, image_bytes, cudaMemcpyHostToDevice);
//
//	temp.host = h_temp;
//	temp.dev = d_temp;
//
//	reshapeTensor(handle, temp, CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW);
//	logTensor(temp, img_path + "img_log\\", "final");
//	
//	free(temp.host);
//	cudaFree(temp.dev);
//	cudaFree(input.dev);
//	free(input.host);
//	cudnnDestroy(handle);
//}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

//int main(int argc, char** argv) {
//	//tryCUDNN();
//	dnCNN();
//	std::cout << "Success!" << std::endl;
//}

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

	//dnCNN init
	cudnnCreate(&handle);

	loadDncnn(handle, model, cam.resolution.y, cam.resolution.x, "C:\\Users\\ryanr\\Desktop\\Penn\\22-23\\CIS565\\Real-Time-Denoising-And-Upscaling\\dnCNN\\weights_renamed\\");
	cudaMalloc(&conv_workspace, 4000000);

	// Initialize ImGui Data
	InitImguiData(guiData);
	InitDataContainer(guiData);

	// GLFW main loop
	mainLoop();

	//dnCNN cleanup
	for (layer& l : model) {
		cudaFree(l.filter.dev);
		cudaFree(l.bias.dev);
		free(l.filter.host);
		free(l.bias.host);
		cudnnDestroyTensorDescriptor(l.input_desc);
		cudnnDestroyTensorDescriptor(l.output_desc);
		cudnnDestroyFilterDescriptor(l.filter_desc);
		cudnnDestroyConvolutionDescriptor(l.convolution);
		cudnnDestroyActivationDescriptor(l.relu);
	}
	cudnnDestroy(handle);
	cudaFree(conv_workspace);

	return 0;
}

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
	auto start = chrono::high_resolution_clock::now();
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
		pathtrace(pbo_dptr, handle, model, frame, iteration, conv_workspace);

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
	auto end = chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//std::cout << "One Iteration: " << duration.count() << std::endl;
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


