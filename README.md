# Standard Particle Swarm Optimization #

This repo contains a GPU based Parallel Implementation of the Standard Particle Swarm Optimization.

Fitness Function: Rastrigin Function.

Make use of the directives to make changes as follows:
1. dims: change the number of dimensions of the search space.
2. x_min and x_max: minimum and maximum values of the independent variable x in each dimension. Note: it is assumed here that the x variable in each dimension has the same domain. For most benchmarking functions, this works. If your function needs different domains for different x variables, you will need to change the code accordingly.
3. max_iters: the number of iterations of the 'Iterations' kernel/step.
4. max_particles: the number of particles in the swarm.
5. fitness function: change lines in the 'Iterate' kernel.

To run this code, you need:
1. A CUDA capable NVIDIA GPU.
2. The CUDA Toolkit (CUDA 8.0 is used for this, but you should be fine with lower versions in most cases).

NOTE: You will need to pass the -lcurand flag to nvcc in order to run this.
If you are working in Visual Studio (Windows), follow these steps:
https://devtalk.nvidia.com/default/topic/509679/curand-linking-error-lnk2019-unresolved-external-symbol/
