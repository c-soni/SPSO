/*
Standard Particle Swarm Optimization
Author: Chintan Soni
Date: 19/03/2017

Objective/Fitness function:
Rastrigin Function:
y = summation (i = 1 to d) {x[i] ^ 2 - 10 * cos(2 * pi * x[i]) + 10}

Constraints:
Domain:
Hypercube x[i]: [-5.12,5.12] for all i < number of dimensions

Global minimum:
y = 0 at x[i] = 0 for all i < d
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define dims 128                     //number of dimensions in fitness function
#define inf 9999.99f                //infinity
#define x_min -5.12f                //minimum x
#define x_max 5.12f                 //maximum x
#define max_iters 2048              //number of iterations
#define max_particles 2048          //number of particles
#define chi 0.72984f                //chi (constriction factor)
#define pi 3.14159265f              //value of pi

#define cudaCheckError()\
{\
	cudaError_t e = cudaGetLastError();\
	if(e != cudaSuccess)\
				{\
			printf("CUDA failure: %s%d: %s", __FILE__, __LINE__, cudaGetErrorString(e));\
			exit(EXIT_FAILURE);\
				}\
}

//Kernel to initialize particles
//Uses cuRAND to generate pseudorandom numbers on the GPU
__global__
void Initialize(float *pos, float *velocity, float *p_best_y, int *l_best_index, int *best_index, curandState *states)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int t_index = threadIdx.x;

	//Adjust pos between -5.12 and 5.12
	pos[index] = x_max * (2.0f * pos[index] - 1.0f);

	//Adjust velocity
	velocity[index] = 0.5f * velocity[index] * (x_max - x_min) / 2.0f;

	//Set PBest to infinity and LBest to self
	//Initialize array of best indices
	if (t_index == 0)
	{
		p_best_y[blockIdx.x] = inf;
		l_best_index[blockIdx.x] = blockIdx.x;
		best_index[blockIdx.x] = blockIdx.x;
	}

	//Initializing up cuRAND
	//Each thread gets a different seed, different sequence number and no offset
	curand_init(index, index, 0, &states[index]);
}

//Kernel for each iteration
__global__
void Iterate(float *pos, float *velocity, float *p_best_x, float *p_best_y, int *l_best_index, curandState *states)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	float personal_best;
	int local_best;
	curandState local_state = states[index];
	float v_max = 0.5f * (x_max - x_min) / 2.0f;
	float c1 = 2.05f, c2 = 2.05f;
	float r1, r2;

	//Set left and right neighbours
	int left = (max_particles + index - 1) % max_particles;
	int right = (1 + index) % max_particles;

	//Calculate fitness of particle
	float fitness = 0.0f;
	for (int i = 0; i < dims; i++)
		fitness += (pos[index * dims + i] * pos[index * dims + i] - 10.0f * cos(2.0f * pi * pos[index * dims + i]));
	fitness += 10.0f * dims;

	//Set PBest if fitness is better
	if (p_best_y[index] > fitness)
	{
		p_best_y[index] = fitness;
		for (int i = 0; i < dims; i++)
			p_best_x[index * dims + i] = pos[index * dims + i];
	}
	personal_best = p_best_y[index];

	//Set the local best index
	if (p_best_y[left] < personal_best)
		l_best_index[index] = left;
	if (p_best_y[right] < personal_best)
		l_best_index[index] = right;
	local_best = l_best_index[index];

	//Update the particle velocity and position
	for (int i = 0; i < dims; i++)
	{
		int id = index * dims + i;
		r1 = curand_uniform(&local_state);
		r2 = curand_uniform(&local_state);

		//Update the velocity
		velocity[id] = chi * (velocity[id] + (c1 * r1 * (p_best_x[id] - pos[id])) + (c2 * r2 * (p_best_x[local_best] - pos[id])));

		//Ensure velocity values are within range
		if (velocity[id] > v_max)
			velocity[id] = v_max;
		if (velocity[id] < -v_max)
			velocity[id] = -v_max;

		//Update the position
		pos[id] = pos[id] + velocity[id];

		//Ensure position values are within range
		if (pos[id] > x_max)
			pos[id] = x_max;
		if (pos[id] < -x_max)
			pos[id] = -x_max;
	}

	//Set the current state of the PRNG
	states[index] = local_state;
}

//Kernel to find the global best
//Parallel reduce to find minimum
//Uses shared memory
//Over-writes PBestY
__global__
void Reduce(float *p_best_x, float *p_best_y, int *best_index, int step)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int tx = threadIdx.x;

	//Declare shared memory for staging the reduce phase
	__shared__ float stage[512];
	__shared__ int best[512];

	//Copy PBestY to shared memory
	best[tx] = best_index[index];
	stage[tx] = p_best_y[index];
	__syncthreads();

	//Perform the actual reduce
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tx < s)
		{
			if (stage[tx] > stage[tx + s])
			{
				stage[tx] = stage[tx + s];
				best[tx] = best[tx + s];
			}
		}
		__syncthreads();
	}

	//Copy results back into global memory
	if (tx == 0)
	{
		p_best_y[blockIdx.x] = stage[0];
		best_index[blockIdx.x] = best[0];
	}

	//Copy particle co-ordinates to first location for step 2
	if (step == 2)
	{
		for (int i = 0; i < dims; i++)
			p_best_x[i] = p_best_x[best[0] * dims + i];
	}
}

int main()
{
	cout << endl;

	float *g_best;
	float *g_best_pos;
	float *pos, *velocity;
	float *p_best_x, *p_best_y;
	int *l_best_index, *best_index;
	float ms = 0;

	//Dynamically allocating memory for results
	g_best = new float;
	g_best_pos = new float[dims];

	curandState *states;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//GPU memory allocations and error checking
	//Position AoS
	cudaMalloc((void**)&pos, max_particles * dims * sizeof(float));
	cudaCheckError();

	//Velocity AoS
	cudaMalloc((void**)&velocity, max_particles * dims * sizeof(float));
	cudaCheckError();

	//PBestX AoS
	cudaMalloc((void**)&p_best_x, max_particles * dims * sizeof(float));
	cudaCheckError();

	//PBestY Array
	cudaMalloc((void**)&p_best_y, max_particles * sizeof(float));
	cudaCheckError();

	//LBestIndex
	cudaMalloc((void**)&l_best_index, max_particles * sizeof(int));
	cudaCheckError();

	//GBestIndex
	cudaMalloc((void**)&best_index, max_particles * sizeof(int));
	cudaCheckError();

	//cuRAND States
	cudaMalloc((void**)&states, max_particles * dims * sizeof(curandState));
	cudaCheckError();

	//Create PRNG
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

	//Initialize pos
	curandGenerateUniform(gen, pos, max_particles * dims);
	cudaCheckError();

	//Initialize velocity
	curandGenerateUniform(gen, velocity, max_particles * dims);
	cudaCheckError();

	//Adjust the pos and velocity values and initialize PBest and LBestIndex
	Initialize << <max_particles, dims >> >(pos, velocity, p_best_y, l_best_index, best_index, states);
	cudaCheckError();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//Print Initialization time taken
	cudaEventElapsedTime(&ms, start, stop);
	cout << "Initialization: time taken: " << ms << " millisec" << endl;

	cout << endl;

	cudaEventRecord(start);

	for (int i = 0; i < max_iters; i++)
		Iterate << <max_particles / 32, 32 >> >(pos, velocity, p_best_x, p_best_y, l_best_index, states);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//Print Iterations time taken
	cudaEventElapsedTime(&ms, start, stop);
	cout << "Iterations: time taken: " << ms << " millisec" << endl;

	cout << endl;

	cudaEventRecord(start);

	//Perform a 2-step global reduce to determine the minimum
	//Step 1
	Reduce << <max_particles / 32, 32 >> >(p_best_x, p_best_y, best_index, 1);
	cudaCheckError();

	//Step 2
	Reduce << <1, max_particles / 32 >> >(p_best_x, p_best_y, best_index, 2);
	cudaCheckError();

	//Copy Results back to host
	//Copy global minimum
	cudaMemcpy((void*)g_best, (void*)p_best_y, sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheckError();

	//Copy co-ordinates of global minimum
	cudaMemcpy((void*)g_best_pos, (void*)p_best_x, dims * sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheckError();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//Print Reduce time taken
	cudaEventElapsedTime(&ms, start, stop);
	cout << "Reduce: time taken: " << ms << " millisec" << endl;

	cout << endl;

	//Print results
	cout << "Global minimum is: " << *g_best << endl;
	cout << "At:" << endl;

	for (int i = 0; i < dims; i++)
		cout << "x[" << i << "] = " << g_best_pos[i] << endl;

	//Clean up section
	//Freeing device memory
	cudaFree(pos);
	cudaFree(velocity);
	cudaFree(p_best_x);
	cudaFree(p_best_y);
	cudaFree(l_best_index);
	cudaFree(best_index);
	cudaFree(states);
	cudaCheckError();

	//Destroying cuRAND generator
	curandDestroyGenerator(gen);
	cudaCheckError();

	cout << endl;

	return 0;
}
