#include "SnowflakeRendering.cuh";
#include "KernelSnowflake.cuh";
#include "SnowflakeVariables.cuh"
#include "CPUSnowflake.cuh";

using namespace std;
#include <chrono>
using namespace std::chrono;

Segment* d_segments = 0;
float numberOfPoints;
bool onlyOnce = true;
int inverted = 1;

void clearBackground();
void generatePointsUsingGPU();
void renderSnowflakeFromBuffer();
void generatePointsUsing100GPU();

bool CPUImplementation = false;
bool GPUImplementationNormal = false;
bool GPUImplementationFast = true;
bool generatePoints = true;

void draw_func(void) {
	clearBackground();
	if (GPUImplementationNormal) {
		if (generatePoints) {
			generatePointsUsingGPU();
		}
		renderSnowflakeFromBuffer();
	}
	else if (GPUImplementationFast) {
		if (generatePoints) {
			generatePointsUsing100GPU();
		}
		renderSnowflakeFromBuffer();
	}
	else if (CPUImplementation) {
		drawSnowflakeWithCPU();
	}
	glutSwapBuffers();
}

void clearBackground() {
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
}

void generatePointsUsingGPU() {
	int blocks = 1;
	int threads = 1024;

	if (onlyOnce) {
		cudaMalloc((void**)&d_segments, pow(4, 12) * sizeof(Segment));
		onlyOnce = false;
	}

	//Map resource to openGL
	float* devPtr;
	size_t size;
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	//Number of points 
	numberOfPoints = 3.0 * pow(4, iterations);

	//Launching kernels
	int kernelCalls = (pow(4, iterations + 1) / (blocks * threads));
	int start_iteration = 1;
	for (int k = 0; k <= kernelCalls; k++) {
		int threadShiftIndex = k * (blocks * threads);
		start_iteration = (0.5 * log2(threadShiftIndex+1));
		if (threadShiftIndex == 0) {
			start_iteration = 1;
		}
		kernel << <1, 1024 >> > (devPtr, d_segments, start_iteration, iterations, inverted, threadShiftIndex);
		cudaDeviceSynchronize();
	}
	cudaGraphicsUnmapResources(1, &resource, NULL);
}

void generatePointsUsing100GPU() {
	//Maximum number of threads that support grid sync

	//Find the number of threads and blocks needed for cooperative groups
	int threads;
	int blocks;
	int maxNumberOfThreads;
	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel, 0, 0);
	maxNumberOfThreads = blocks * threads;

	//Allocate large memory only once for branches
	if (onlyOnce) {
		cudaMalloc((void**)&d_segments, pow(4, 13) * sizeof(Segment));
		onlyOnce = false;
	}

	//Map resources to OpenGL
	float* devPtr;
	size_t size;
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);
	numberOfPoints = pow(4, iterations );

	//Launching Kernels
	int kernelCalls = (pow(4, iterations + 1) / (blocks * threads));
	int start_iteration = 1;

	//Keep calling the kernel until we generate all data 
	for (int k = 0; k <= kernelCalls; k++) {
		int threadShiftIndex = k * (blocks * threads);
		int previuos_iteration = start_iteration;
		start_iteration = (0.5 * log2(threadShiftIndex));
		if (threadShiftIndex == 0) {
			start_iteration = 1;
		}
		void* kernelArgs[] = { &devPtr, &d_segments, &start_iteration, &iterations, &inverted, &threadShiftIndex };
		cudaLaunchCooperativeKernel((void*)kernel, blocks, threads, kernelArgs, 0, 0);
	}
	cudaGraphicsUnmapResources(1, &resource, NULL);
}

void renderSnowflakeFromBuffer() {
	glColor3f(0.29f, 0.44f, 0.55f);
	glLineWidth(2.0f);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(0);
	numberOfPoints = 3*round(pow(4, iterations));
	glDrawArrays(GL_LINE_LOOP, 0, numberOfPoints);
}