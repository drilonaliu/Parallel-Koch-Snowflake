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

__device__ struct Point {
	float x;
	float y;
};


__device__ struct Segment {
	Point A;
	Point B;
};

__device__ void triangleOnSegment(Point A, Point B, Point* A1, Point* B1, Point* C1, int inverted);
__global__ void kernel(float* points, Segment* segments, int start_iteration, int max_iteration, int inverted, int threadShiftIndex);