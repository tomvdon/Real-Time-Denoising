#include "main.h"
#include "preview.h"
#include "utilities.h"
#include <cstring>

#include <chrono>

#include <cudnn.h>
#include "opencv2\opencv.hpp"

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

	if (image.empty()) {
		std::cout << "Could not read the image: " << image_path << std::endl;
		return image;
	}

	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);

	cv::imshow("Display window", image);
	int k = cv::waitKey(0); // Wait for a keystroke in the window

	return image;
}

void save_image(const char* output_filename,
	float* buffer,
	int height,
	int width) {
	// Credit http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
	cv::Mat output_image(height, width, CV_32FC3, buffer);
	// Make negative values zero.
	cv::threshold(output_image,
		output_image,
		/*threshold=*/0,
		/*maxval=*/0,
		cv::THRESH_TOZERO);
	cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
	output_image.convertTo(output_image, CV_8UC3);
	cv::imwrite(output_filename, output_image);
}

struct filter {
	int in_channels;
	int out_channels;
	int height;
	int width;
};

float* read_filter(const filter& f, std::string f_path) {
	float* arr = (float*) malloc(sizeof(float) * f.in_channels * f.out_channels * f.height * f.width);
	std::ifstream fp_in;
	fp_in.open(f_path);
	if (!fp_in.is_open()) {
		std::cout << "Error reading from file - aborting!" << std::endl;
		throw;
	}
	for (int i = 0; i < f.out_channels; ++i) {
		for (int j = 0; j < f.in_channels; ++j) {
			// Credit https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
			std::string line;
			utilityCore::safeGetline(fp_in, line);
			size_t pos = 0;
			std::string token;
			std::string delimiter = ",";
			// TODO Fix this
			for (int k = 0; k < f.width; ++k) {
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				float t = std::stof(token);
				std::cout << t << std::endl;
				arr[i * f.height * f.width + j * f.width + k] = t;
				line.erase(0, pos + delimiter.length());
			}
		}
	}
	return arr;
}

void tryCUDNN() {
	// Credit http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
	std::cout << "Running" << std::endl;
	cudnnHandle_t handle;
	cudnnCreate(&handle);

	cv::Mat image = load_image("C:\\Users\\ryanr\\Desktop\\Penn\\22-23\\CIS565\\Real-Time-Denoising-And-Upscaling\\img\\tensorflow.png");
	std::cout << "Image has shape " << image.rows << ", " << image.cols << std::endl;
	// Describe input tensor
	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/3,
		/*image_height=*/image.rows,
		/*image_width=*/image.cols));

	// Describe output tensor
	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/1,
		/*channels=*/3,
		/*image_height=*/image.rows,
		/*image_width=*/image.cols));

	// Describe convolutional filter shape
	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*format=*/CUDNN_TENSOR_NCHW,
		/*out_channels=*/3,
		/*in_channels=*/3,
		/*kernel_height=*/3,
		/*kernel_width=*/3));

	// Describe convolution properties
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

	// Find fastest conv algorithim
	cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;
	checkCUDNN(
		cudnnFindConvolutionForwardAlgorithm(handle,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			/*RequestedNumAlgs*/1,
			/*ReturnedNumAlgs*/0,
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
	std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
		<< std::endl;

	// Allocatate buffers on device and copy/memset
	void* d_workspace{ nullptr };
	cudaMalloc(&d_workspace, workspace_bytes);

	int image_bytes = 1 * 3 * image.rows * image.cols * sizeof(float); 

	float* d_input{ nullptr };
	cudaMalloc(&d_input, image_bytes);
	cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

	float* d_output{ nullptr };
	cudaMalloc(&d_output, image_bytes); // Normally you would use cudnnGetConvolution2dForwardOutputDim to know the output size but we know it will be the same for this example
	cudaMemset(d_output, 0, image_bytes);

	// Mystery kernel
	const float kernel_template[3][3] = {
	  {1,  1, 1},
	  {1, -8, 1},
	  {1,  1, 1}
	};

	float h_kernel[3][3][3][3];
	for (int kernel = 0; kernel < 3; ++kernel) {
		for (int channel = 0; channel < 3; ++channel) {
			for (int row = 0; row < 3; ++row) {
				for (int column = 0; column < 3; ++column) {
					h_kernel[kernel][channel][row][column] = kernel_template[row][column];
				}
			}
		}
	}

	float* d_kernel{ nullptr };
	cudaMalloc(&d_kernel, sizeof(h_kernel));
	cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

	// Do convolution forward
	const float alpha = 1, beta = 0;
	checkCUDNN(cudnnConvolutionForward(handle,
		&alpha,
		input_descriptor,
		d_input,
		kernel_descriptor,
		d_kernel,
		convolution_descriptor,
		convolution_algorithm.algo,
		d_workspace,
		workspace_bytes,
		&beta,
		output_descriptor,
		d_output));

	// Retrive output from device
	float* h_output = new float[image_bytes];
	cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
	// Save img
	save_image("../img/cudnn-out.png", h_output, height, width);

	// Free up stuff
	delete[] h_output;
	cudaFree(d_kernel);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	tryCUDNN();
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


