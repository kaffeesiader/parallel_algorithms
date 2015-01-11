
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

#include "cl_utils.h"
#include "time_ms.h"

#ifndef CL_DEVICE
	#define CL_DEVICE 0
#endif

// the problem size - the number of particles
#ifndef N
  #define N 200
#endif

#define SPACE_SIZE 1000
#define SMOOTHING_LENGTH 0.5
#define KERNEL_FILE_NAME "./n_body.cl"

// structure for keeping track of opencl management data
typedef struct {
	cl_context		 ctx;
	cl_command_queue queue;
	cl_device_id	 id;
	cl_program		 prog;
	cl_kernel		 kernel_step;
	cl_mem			 mem_bodies;
} ocl_mgmt;

// the type used to model one body
typedef struct {
	cl_float  m;		// the mass of the body
	cl_float3 pos;		// the position in space
	cl_float3 v;		// the velocity of the body
} body;

// ocl stuff
ocl_mgmt ocl;
// global array of bodies
body B[N];
// count simulation steps
int steps;
// sum step times
double totalTime;

// ----- utility functions ------

float rand_val(float min, float max) {
	return (rand() / (float) RAND_MAX) * (max - min) + min;
}

cl_float3 vec_rand() {
	cl_float3 rnd;
	rnd.s[0] = rand_val(-SPACE_SIZE, SPACE_SIZE);
	rnd.s[1] = rand_val(-SPACE_SIZE, SPACE_SIZE);
	rnd.s[2] = rand_val(-SPACE_SIZE, SPACE_SIZE);

	return rnd;
}

cl_float3 vec_zero() {
	cl_float3 zero;
	zero.s[0] = 0.0f;
	zero.s[1] = 0.0f;
	zero.s[2] = 0.0f;

	return zero;
}

void simulation_step() {
	cl_event ev_kernel; cl_int err;

	size_t glb_sz[1] = { N };
	err = clEnqueueNDRangeKernel(ocl.queue, ocl.kernel_step, 1, NULL, glb_sz, NULL, 0, NULL, &ev_kernel);
	CLU_ERRCHECK(err, "Failed to enqueue kernel");
	// read result...
	err = clEnqueueReadBuffer(ocl.queue, ocl.mem_bodies, CL_TRUE, 0, N * sizeof(body), &B, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to read result");
	// and evaluate time
	totalTime += cluGetDurationMS(ev_kernel);
	steps++;
}

void check() {
	// check result (impulse has to be zero)
//	cl_float3 sum = vec_zero();
//	for(int i=0; i<N; i++) {
//		// impulse = m * v
//		sum = add(sum, mul_s(B[i].v,B[i].m));
//	}
//	int success = norm(sub(sum, triple_zero())) < eps*100.0f;
//	printf("Verification: %s\n", ((success)?"OK":"ERR"));
//	if(!success) {
//		triple_print(sum); printf(" should be (0,0,0)\n");
//	}
}

// ----------- opencl -------------
void init_ocl() {
	cl_int err;

	ocl.id = cluInitDevice(CL_DEVICE, &ocl.ctx, &ocl.queue);
	printf("OCL Device: %s\n", cluGetDeviceDescription(ocl.id, CL_DEVICE));

	// create kernel from source
	ocl.prog = cluBuildProgramFromFile(ocl.ctx, ocl.id, KERNEL_FILE_NAME, NULL);
	ocl.kernel_step = clCreateKernel(ocl.prog, "simulation_step", &err);
	CLU_ERRCHECK(err, "Failed to create 'simulation_step' kernel from program");

	// create memory buffer
	ocl.mem_bodies = clCreateBuffer(ocl.ctx, CL_MEM_READ_WRITE, N * sizeof(body), NULL, &err);
	CLU_ERRCHECK(err, "Failed to create memory buffer");

	// fill memory buffer
	err = clEnqueueWriteBuffer(ocl.queue, ocl.mem_bodies, CL_FALSE, 0, N * sizeof(body), B, 0, NULL, NULL);
	CLU_ERRCHECK(err, "Failed to write data to device");

	// set arguments
	cluSetKernelArguments(ocl.kernel_step, 1, sizeof(cl_mem), (void *)&ocl.mem_bodies);
}

void cleanup_ocl() {
	cl_int err;

	// finalization
	err = clFinish(ocl.queue);
	err |= clReleaseKernel(ocl.kernel_step);
	err |= clReleaseProgram(ocl.prog);
	err |= clReleaseMemObject(ocl.mem_bodies);
	err |= clReleaseCommandQueue(ocl.queue);
	err |= clReleaseContext(ocl.ctx);
	CLU_ERRCHECK(err, "Failed during ocl cleanup");
}

// ----- opengl -----
void init_gl(int *argc, char** argv) {

	glutInit(argc, argv);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(20, 20);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow("N-Body-OCL");

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(90.0, 4.0/3.0, 1.0, 10000.0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	// light
	glEnable(GL_LIGHTING);
	glShadeModel(GL_SMOOTH);

	{
		GLfloat ambient[] = { 0.4f, 0.4f, 0.4f, 1.0f };
		GLfloat diffuse[] = { 0.8f, 0.8f, 0.8, 1.0f };
		GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		GLfloat position[] = { 0.0f, 0.0f, 0.0f, 1.0f };
		glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
		glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
		glLightfv(GL_LIGHT0, GL_POSITION, position);
		glEnable(GL_LIGHT0);
	}
	{
		GLfloat ambient[] = { 0.4f, 0.0f, 0.0f, 1.0f };
		GLfloat diffuse[] = { 0.8f, 0.0f, 0.0, 1.0f };
		GLfloat specular[] = { 1.0f, 0.0f, 0.0f, 1.0f };
		GLfloat position[] = { 0.0f, 0.0f, -2000.0f, 1.0f };
		glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);
		glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
		glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
		glLightfv(GL_LIGHT1, GL_POSITION, position);
		glEnable(GL_LIGHT1);
	}
	printf("OpenGL-Version: %s\n", glGetString(GL_VERSION));
}

void display() {
	simulation_step(&ocl);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GLUquadricObj *quadric = gluNewQuadric();
	gluQuadricNormals(quadric, GLU_SMOOTH);
	gluQuadricOrientation(quadric, GLU_OUTSIDE);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.0, 0.0, -2000.0);

	// render particles
	for(int j=0; j<N; j++) {
		glPushMatrix();
		glTranslatef(B[j].pos.s[0], B[j].pos.s[1], B[j].pos.s[2]);
		gluSphere(quadric, B[j].m, 16, 16);
		glPopMatrix();
	}

	gluDeleteQuadric(quadric);
	glutSwapBuffers();
	glutPostRedisplay();
}

void exit_cb() {
	//do verification
	check();

	// print result and clean up
	printf("Average step time: %.3f ms\n", totalTime / steps);
	printf("Calculated steps : %d\n", steps);
	cleanup_ocl();
}

int main(int argc, char **argv)
{	
	printf("N=%d, CL_DEVICE=%d\n", N, CL_DEVICE);
	steps = 0;
	totalTime = 0;

	// distribute bodies in space (randomly)
	for(int i=0; i<N; i++) {
		B[i].m = rand_val(8,20); // (8,20)
		B[i].pos = vec_rand();
		B[i].v   = vec_zero();
	}

	// ocl initialization
	init_ocl();

	// ogl stuff
	init_gl(&argc, argv);
	glutDisplayFunc(&display);
	atexit(&exit_cb);
	glutMainLoop();
	
	return EXIT_SUCCESS;
}

