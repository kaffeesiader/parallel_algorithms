#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define SMOOTHING_LENGTH 0.5

// the type used to model one body
typedef struct {
	float m;		// the mass of the body
	float3 pos;		// the position in space
	float3 vel;		// the velocity of the body
} body;

__kernel void simulation_step(__global body *B)
{
	int gId = get_global_id(0);
	int g_size = get_global_size(0);

	// init force with zero
	float3 force = (float3) (.0f, .0f, .0f);

	// copy body data into private memory
	body this = B[gId];

	for(int i = 0; i < g_size; ++i) {
	    if(i != gId) {
		// compute distance vector
		float3 dist = B[i].pos - this.pos;

		// compute absolute distance
		float r = length(dist) + SMOOTHING_LENGTH;

		// compute strength of force (G = 1 (who cares))
		// F = G * (m1 * m2) / r^2
		float f = (this.m * B[i].m) / (r*r);

		// compute current contribution to force
		float s = f / r;
		float3 cur = dist * s;

		// accumulate force
		force += cur;
	    }
	}
	// apply force
	// update speed
	// F = m * a
	// a = F / m	// m=1
	// v' = v + a
	B[gId].vel += (force / B[gId].m);

	// update position
	// pos = pos + v * dt	// dt = 1
	B[gId].pos += B[gId].vel;
}
