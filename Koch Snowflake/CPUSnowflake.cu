#include "CPUSnowflake.cuh";
#include "KernelSnowflake.cuh";

using namespace std;
#include <chrono>
using namespace std::chrono;

void triangleOnSegmentCPU(Point A, Point B, Point* A1, Point* B1, Point* C1);

const int size_segments = pow(4, 14);
Segment* segments = new Segment[size_segments]; //4^12




void drawSnowflakeWithCPU() {
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

	A.x = 0.0f;
	A.y = 1.29 - 0.4f;
	B.x = 0.75f;
	B.y = 0.0f - 0.4f;
	C.x = -0.75f;
	C.y = 0.0f - 0.4f;

	glBegin(GL_LINE_LOOP);
	glVertex2f(A.x, A.y);
	glVertex2f(B.x, B.y);
	glVertex2f(C.x, C.y);
	glEnd();

	segment_1.A = A;
	segment_1.B = B;
	segment_2.A = B;
	segment_2.B = C;
	segment_3.A = C;
	segment_3.B = A;

	segments[1] = segment_1;
	segments[2] = segment_2;
	segments[3] = segment_3;

	glClear(GL_COLOR_BUFFER_BIT);
	glPointSize(1.0f);
	glColor3f(0.0f, 0.31f, 0.45f);

	glBegin(GL_LINE_LOOP);
	int start_iteration = 1;
	int max_iteration = iterations;

	auto start = high_resolution_clock::now();
	for (int iteration = start_iteration; iteration <= max_iteration; iteration++) {
		int start_at = pow(4.0, iteration - 1);
		int end_at = pow(4.0, iteration);
		for (int idx = start_at; idx < end_at; idx++) {
			Segment segment = segments[idx];
			A = segment.A;
			B = segment.B;
			triangleOnSegmentCPU(A, B, &A1, &B1, &C1);

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

			if (iteration == max_iteration) {
				glVertex2f(A.x, A.y);
				glVertex2f(B1.x, B1.y);
				glVertex2f(C1.x, C1.y);
				glVertex2f(A1.x, A1.y);
				glVertex2f(B.x, B.y);
			}
		}
	}
	glEnd();

	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);

	cout << "\n The time is in microseconds ";
	cout << duration.count()
		<< endl;
}


void triangleOnSegmentCPU(Point A, Point B, Point* A1, Point* B1, Point* C1) {

	float n1 = 1.0f;
	float m1 = 2.0f;
	float n2 = 2.0f;
	float m2 = 1.0f;

	(*A1).x = (m1 * B.x + n1 * A.x) / (m1 + n1);
	(*A1).y = (m1 * B.y + n1 * A.y) / (m1 + n1);

	(*B1).x = (m2 * B.x + n2 * A.x) / (m2 + n2);
	(*B1).y = (m2 * B.y + n2 * A.y) / (m2 + n2);

	float angle = -3.14159265358f / 3.0f;
	angle = inverted * angle;

	(*C1).x = (*B1).x * cos(angle) - (*B1).y * sin(angle) - (*A1).x * cos(angle) + (*A1).y * sin(angle) + (*A1).x;
	(*C1).y = (*B1).x * sin(angle) + (*B1).y * cos(angle) - (*A1).x * sin(angle) - (*A1).y * cos(angle) + (*A1).y;
}