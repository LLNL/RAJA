/// \file
/// Leapfrog time integrator

#ifndef __LEAPFROG_H
#define __LEAPFROG_H

#include "CoMDTypes.h"

double timestep(SimFlat* s, int n, real_t dt);
void computeForce(SimFlat* s);
void kineticEnergy(SimFlat* s);
void updateIndexSets(SimFlat *s) ;

/// Update local and remote link cells after atoms have moved.
void redistributeAtoms(struct SimFlatSt* sim);

#endif
