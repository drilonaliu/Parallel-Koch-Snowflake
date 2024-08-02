// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <device_launch_parameters.h>

#include "SnowflakeVariables.cuh"
#include "SnowflakeRendering.cuh"
#include "UserInteraction.cuh";
#include "KochSnowflake.cuh";

using namespace std;

#define DIM 512 ;

GLuint bufferObj;
cudaGraphicsResource* resource;
int dim = 512;
int iteration = 0;
int numVertices = 3 * pow(4, iteration);
int iterations =  5;
int max_iteration = 4;


void initializeWindow(int argc, char** argv);
void bindFunctionsToWindow();
void setUpCudaOpenGLInterop();

void startKochSnowflake(int argc, char** argv) {
	initializeWindow(argc, argv);
	printControls();
	createMenu();
	setUpCudaOpenGLInterop();
	bindFunctionsToWindow();
	glutMainLoop();
}

void initializeWindow(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(512, 512);
	glutCreateWindow("Koch Snwoflake");
	glewInit();
}

void bindFunctionsToWindow() {
	glutKeyboardFunc(keyboardHandler);
	glutSpecialFunc(specialKeyHandler);
	glutDisplayFunc(draw_func);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);
	glutMouseWheelFunc(mouseWheel);
}

void setUpCudaOpenGLInterop() {
	//Choose the most suitable CUDA device based on the specified properties (in prop). It assigns the device ID to dev.
	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaError_t error = cudaChooseDevice(&dev, &prop);
	if (error != cudaSuccess) {
		printf("Error choosing CUDA device: %s\n", cudaGetErrorString(error));
	}
	cudaGLSetGLDevice(dev);

	//Buffer Size
	float numVertices = 3.0 * pow(4, 12);
	size_t bufferSize = sizeof(float) * numVertices * 2;

	//Generate openGL buffer
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_ARRAY_BUFFER, bufferObj); 
	glBufferData(GL_ARRAY_BUFFER, bufferSize, NULL, GL_DYNAMIC_COPY);

	//Notify CUDA runtime that we intend to share the OpenGL buffer named bufferObj with CUDA.//FlagsNone, ReadOnly, WriteOnly
	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
}


