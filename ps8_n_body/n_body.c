
#ifdef WIN32
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "time_ms.h"

// the problem size - the number of particles
#ifndef N
  #define N 200
#endif

#define SPACE_SIZE 1000

#define SMOOTHING_LENGTH 0.5

// the type used to represent a triple of floats
typedef struct {
	float x, y, z;
} triple;

// the types used to model position, speed and forces
typedef triple position;
typedef triple velocity;
typedef triple force;
typedef triple impulse;

// the type used to model one body
typedef struct {
	float m;			// the mass of the body
	position pos;		// the position in space
	velocity v;			// the velocity of the body
} body;

// ----- necessary globals due to GLUT limitations -----

// the list of bodies
body B[N];
// the forces effecting the particles
force F[N];

// count simulation steps
int steps;
// sum step times
unsigned long long totalTime;

// ----- utility functions ------

float rand_val(float min, float max) {
	return (rand() / (float) RAND_MAX) * (max - min) + min;
}

triple triple_zero() {
	triple zero = {0.0, 0.0, 0.0};
	return zero;
}

triple triple_rand() {
	triple rand_triple = {
		rand_val(-SPACE_SIZE,SPACE_SIZE),
		rand_val(-SPACE_SIZE,SPACE_SIZE),
		rand_val(-SPACE_SIZE,SPACE_SIZE)
	};
	return rand_triple;
}

void triple_print(triple t) {
	printf("(%f,%f,%f)", t.x, t.y, t.z);
}

// ----- some operators -----

#define eps 0.0001
#define abs(V) (((V)<0)?-(V):(V))
#define min(A,B) (((A)<(B))?(A):(B))

triple add(triple a, triple b) { 
	triple ret = { a.x + b.x, a.y + b.y, a.z + b.z };
	return ret;
}
triple sub(triple a, triple b) { 
	triple ret = { a.x - b.x, a.y - b.y, a.z - b.z };
	return ret;
}
triple div_(triple a, triple b) { 
	triple ret = { a.x / b.x, a.y / b.y, a.z / b.z };
	return ret;
}

triple mul_s(triple a, float s) { 
	triple ret = { a.x * s, a.y * s, a.z * s };
	return ret;
}
triple div_s(triple a, float s) { 
	triple ret = { a.x / s, a.y / s, a.z / s };
	return ret;
}

bool eq(triple a, triple b) {
	return (abs(a.x-b.x) < eps && abs(a.y-b.y) < eps && abs(a.z-b.z) < eps);
}
float norm(triple t) {
	return sqrt(t.x*t.x + t.y*t.y + t.z*t.z);
}
triple normalized(triple t) {
	return mul_s(t, (1/norm(t)));
}

// ----- simulation -----

void simulation_step() {
	unsigned long long startTime = time_ms();

	// set forces to zero
	for(int j=0; j<N; j++) {
		F[j] = triple_zero();
	}
	
	// compute forces for each body (very naive)
	for(int j=0; j<N; j++) {
		for(int k=0; k<N; k++) {
			if(j!=k) {
				// compute distance vector
				triple dist = sub(B[k].pos, B[j].pos);

				// compute absolute distance
				float r = norm(dist) + SMOOTHING_LENGTH;
				
				// compute strength of force (G = 1 (who cares))
				//			F = G * (m1 * m2) / r^2
				float f = (B[j].m * B[k].m) / (r*r);

				// compute current contribution to force
				float s = f / r;
				force cur = mul_s(dist,s);

				// accumulate force
				F[j] = add(F[j], cur);
			}
		}
	}
		
	// apply forces
	for(int j=0; j<N; j++) {
		// update speed
		//		F = m * a
		//		a = F / m		// m=1
		//		v' = v + a
		B[j].v = add(B[j].v, div_s(F[j], B[j].m));

		// update position
		//		pos = pos + v * dt		// dt = 1
		B[j].pos = add(B[j].pos, B[j].v);
	}

	unsigned long long stepTime = time_ms() - startTime;
	totalTime += stepTime;
	steps++;
}

void check() {
	// check result (impulse has to be zero)
	impulse sum = triple_zero();
	for(int i=0; i<N; i++) {
		// impulse = m * v
		sum = add(sum, mul_s(B[i].v,B[i].m));
	}
	int success = norm(sub(sum, triple_zero())) < eps*100.0f;
	printf("Verification: %s\n", ((success)?"OK":"ERR"));
	if(!success) {
		triple_print(sum); printf(" should be (0,0,0)\n");
	}
}

void exit_cb() {
	check();
	printf("Average step time: %llu ms\n", totalTime / steps);
}

// ----- opengl -----

void init_gl(int *argc, char** argv) {
	glutInit(argc, argv);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(20, 20);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow("N-Body");

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
}

void display() {
	simulation_step();
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
		glTranslatef(B[j].pos.x, B[j].pos.y, B[j].pos.z);
		gluSphere(quadric, B[j].m, 16, 16);
		glPopMatrix();
	}

	gluDeleteQuadric(quadric); 
	glutSwapBuffers();
	glutPostRedisplay();
}

// ----- main -----
int main(int argc, char** argv) {

	printf("N=%d\n", N);
	steps = 0;
	totalTime = 0;

	// distribute bodies in space (randomly)
	for(int i=0; i<N; i++) {
		B[i].m = rand_val(8,20);
		B[i].pos = triple_rand();
		B[i].v   = triple_zero();
	}

	init_gl(&argc, argv);
	glutDisplayFunc(&display);
	atexit(&exit_cb);
	glutMainLoop();
}
