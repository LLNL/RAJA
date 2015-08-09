/// \file
/// Simple random number generators for uniform and gaussian
/// distributions.

#ifndef __RANDOM_H
#define __RANDOM_H

#include "mytype.h"
#include <stdint.h>

/// Return a random number from a Gaussian distribution.
real_t gasdev(uint64_t* seed);

/// Return a random number from a uniform distribution.
double lcg61(uint64_t* seed);

/// Return a seed suitable for calling lcg61 or gasdev.
uint64_t mkSeed(uint32_t id, uint32_t callSite);

#endif
