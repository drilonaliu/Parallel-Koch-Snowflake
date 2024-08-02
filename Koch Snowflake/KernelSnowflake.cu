#include "KernelSnowflake.cuh";
#include <cooperative_groups.h>;

namespace cg = cooperative_groups;

using namespace std;
#include <chrono>
using namespace std::chrono;

__global__ void kernel(float* points, Segment* segments, int start_iteration, int max_iteration	, int inverted, int threadShiftIndex) {
	Point A;
	Point B;
	Point C;
	Point A1;
	Point B1;
	Point C1;
	Segment segment_1;
	Segment segment_2;
	Segment segment_3;
	Segment segment_4;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	idx += threadShiftIndex;

	auto g = cg::this_grid();
	
	if (idx == 0) {
		A.x = 0.0f;
		A.y = 1.29 - 0.4f;
		B.x = 0.75f;
		B.y = 0.0f - 0.4f;
		C.x = -0.75f;
		C.y = 0.0f - 0.4f;

		points[0] = A.x;  
		points[1] = A.y;
		points[2] = B.x; 
		points[3] = B.y;  
		points[4] = C.x; 
		points[5] = C.y;  

		segment_1.A = A;
		segment_1.B = B;
		segment_2.A = B;
		segment_2.B = C;
		segment_3.A = C;
		segment_3.B = A;

		segments[1] = segment_1;
		segments[2] = segment_2;
		segments[3] = segment_3;
	}

	g.sync();
	//__syncthreads();

	for (int iteration = start_iteration; iteration <= max_iteration; iteration++) {

		int start_at = round(pow(4.0, iteration - 1));
		int end_at =round( pow(4.0, iteration));

		if (idx >= start_at && idx <= end_at) {

			Segment segment = segments[idx];
			A = segment.A;
			B = segment.B;
			triangleOnSegment(A, B, &A1, &B1, &C1, inverted);

			//Koch curve
			segment_1.A = A;
			segment_1.B = B1;
			segment_2.A = B1;
			segment_2.B = C1;
			segment_3.A = C1;
			segment_3.B = A1;
			segment_4.A = A1;
			segment_4.B = B;

			//Insert the generated koch curvers into segments array
			int offset = end_at + 4 * (idx - start_at);
			segments[offset] = segment_1;
			segments[offset + 1] = segment_2;
			segments[offset + 2] = segment_3;
			segments[offset + 3] = segment_4;

			//Insert vertices to points array
			offset = 2 * 4 * (idx - start_at);
			points[offset] = A.x;
			points[offset + 1] = A.y;
			points[offset + 2] = B1.x;
			points[offset + 3] = B1.y;
			points[offset + 4] = C1.x;
			points[offset + 5] = C1.y;
			points[offset + 6] = A1.x;
			points[offset + 7] = A1.y;
			points[offset + 8] = B.x;
			points[offset + 9] = B.y;
		}
		g.sync();
		//__syncthreads();
	}
}

/**
* Finds points A1 B1 C1 such that triangle A1B1C1 is equilitrial with sides 1/3AB of  the segment AB.
*/
__device__ void triangleOnSegment(Point A, Point B, Point* A1, Point* B1, Point* C1, int inverted) {

	float n1 = 1.0f;
	float m1 = 2.0f;
	float n2 = 2.0f;
	float m2 = 1.0f;

	(*A1).x = (m1 * B.x + n1 * A.x) / (m1 + n1);
	(*A1).y = (m1 * B.y + n1 * A.y) / (m1 + n1);

	(*B1).x = (m2 * B.x + n2 * A.x) / (m2 + n2);
	(*B1).y = (m2 * B.y + n2 * A.y) / (m2 + n2);

	float angle = -3.14159265358f / 3.0f;
	angle = angle * inverted;

	(*C1).x = (*B1).x * cos(angle) - (*B1).y * sin(angle) - (*A1).x * cos(angle) + (*A1).y * sin(angle) + (*A1).x;
	(*C1).y = (*B1).x * sin(angle) + (*B1).y * cos(angle) - (*A1).x * sin(angle) - (*A1).y * cos(angle) + (*A1).y;
}

