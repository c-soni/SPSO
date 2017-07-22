# Standard Particle Swarm Optimization #

SPSO is a swarm based approach to find the global minimum of an objective function over a given problem space.

This implementation includes a serial and a parallel version.

The serial version is written in C++ and works for an objective function in multiple variables with small tweaks.

The parallel version has been implemented using CUDA C++ and is similar to the serial version in terms of flexibility of objective function.

The simulations directory includes Python versions of the algorithm which plots the function and simulates the behaviour of the particles using matplotlib.

* SPSO implementations on CPU and GPU
* Simulation of SPSO
