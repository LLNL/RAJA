/// \file
/// Leapfrog time integrator

#include "timestep.h"

#include <omp.h>

#include "CoMDTypes.h"
#include "linkCells.h"
#include "parallel.h"
#include "performanceTimers.h"

static void advanceVelocity(SimFlat* s, RAJA::IndexSet *extent, real_t dt);
static void advancePosition(SimFlat* s, RAJA::IndexSet *extent, real_t dt);


/// Advance the simulation time to t+dt using a leap frog method
/// (equivalent to velocity verlet).
///
/// Forces must be computed before calling the integrator the first time.
///
///  - Advance velocities half time step using forces
///  - Advance positions full time step using velocities
///  - Update link cells and exchange remote particles
///  - Compute forces
///  - Update velocities half time step using forces
///
/// This leaves positions, velocities, and forces at t+dt, with the
/// forces ready to perform the half step velocity update at the top of
/// the next call.
///
/// After nSteps the kinetic energy is computed for diagnostic output.
double timestep(SimFlat* s, int nSteps, real_t dt)
{
   for (int ii=0; ii<nSteps; ++ii)
   {
      startTimer(velocityTimer);
      advanceVelocity(s, s->isLocal, 0.5*dt); 
      stopTimer(velocityTimer);

      startTimer(positionTimer);
      advancePosition(s, s->isLocal, dt);
      stopTimer(positionTimer);

      startTimer(redistributeTimer);
      redistributeAtoms(s);
      stopTimer(redistributeTimer);

      startTimer(computeForceTimer);
      computeForce(s);
      stopTimer(computeForceTimer);

      startTimer(velocityTimer);
      advanceVelocity(s, s->isLocal, 0.5*dt); 
      stopTimer(velocityTimer);
   }

   kineticEnergy(s);

   return s->ePotential;
}

void computeForce(SimFlat* s)
{
   s->pot->force(s);
}


void advanceVelocity(SimFlat* s, RAJA::IndexSet *extent, real_t dt)
{
   RAJA::forall<atomWork>(*extent, [=] (int iOff) {
      s->atoms->p[iOff][0] += dt*s->atoms->f[iOff][0];
      s->atoms->p[iOff][1] += dt*s->atoms->f[iOff][1];
      s->atoms->p[iOff][2] += dt*s->atoms->f[iOff][2];
   } ) ;
}

void advancePosition(SimFlat* s, RAJA::IndexSet *extent, real_t dt)
{
   RAJA::forall<atomWork>(*extent, [=] (int iOff) {
      int iSpecies = s->atoms->iSpecies[iOff];
      real_t invMass = 1.0/s->species[iSpecies].mass;
      s->atoms->r[iOff][0] += dt*s->atoms->p[iOff][0]*invMass;
      s->atoms->r[iOff][1] += dt*s->atoms->p[iOff][1]*invMass;
      s->atoms->r[iOff][2] += dt*s->atoms->p[iOff][2]*invMass;
   } ) ;
}

/// Calculates total kinetic and potential energy across all tasks.  The
/// local potential energy is a by-product of the force routine.
void kineticEnergy(SimFlat* s)
{
   real_t eLocal[2];
   RAJA::ReduceSum<RAJA::omp_reduce, real_t> kenergy(0.0) ;
   eLocal[0] = s->ePotential;
   eLocal[1] = 0;

   {
   RAJA::forall<atomWork>(*s->isLocal, [&] (int iOff) {
      int iSpecies = s->atoms->iSpecies[iOff];
      real_t invMass = 0.5/s->species[iSpecies].mass;
      kenergy += ( s->atoms->p[iOff][0] * s->atoms->p[iOff][0] +
                   s->atoms->p[iOff][1] * s->atoms->p[iOff][1] +
                   s->atoms->p[iOff][2] * s->atoms->p[iOff][2] )*
                  invMass ;
   } ) ;
   }

   eLocal[1] = kenergy;

   real_t eSum[2];
   startTimer(commReduceTimer);
   addRealParallel(eLocal, eSum, 2);
   stopTimer(commReduceTimer);

   s->ePotential = eSum[0];
   s->eKinetic = eSum[1];
}

/// \details
/// This function provides one-stop shopping for the sequence of events
/// that must occur for a proper exchange of halo atoms after the atom
/// positions have been updated by the integrator.
///
/// - updateLinkCells: Since atoms have moved, some may be in the wrong
///   link cells.
/// - haloExchange (atom version): Sends atom data to remote tasks. 
/// - sort: Sort the atoms.
///
/// \see updateLinkCells
/// \see initAtomHaloExchange
/// \see sortAtomsInCell
void redistributeAtoms(SimFlat* sim)
{
   updateLinkCells(sim->boxes, sim->atoms);

   startTimer(atomHaloTimer);
   haloExchange(sim->atomExchange, sim);
   stopTimer(atomHaloTimer);

   updateIndexSets(sim) ;

   #pragma omp parallel for
   for (int ii=0; ii<sim->boxes->nTotalBoxes; ++ii)
      sortAtomsInCell(sim->atoms, sim->boxes, ii);
}

void updateIndexSets(SimFlat *s)
{
#pragma omp parallel for
   for (int i=0; i<s->boxes->nTotalBoxes; ++i) {
      RAJA::RangeSegment *seg =
         static_cast<RAJA::RangeSegment *>(s->isTotal->getSegment(i)) ;

      seg->setEnd(i*MAXATOMS + s->boxes->nAtoms[i]) ;
   }
}

