/// \file
/// Parallel domain decomposition.

#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include "mytype.h"

/// Domain decomposition information.
typedef struct DomainSt
{
   // process-layout data
   int procGrid[3];        //!< number of processors in each dimension
   int procCoord[3];       //!< i,j,k for this processor

   // global bounds data
   real3 globalMin;        //!< minimum global coordinate (angstroms)
   real3 globalMax;        //!< maximum global coordinate (angstroms)
   real3 globalExtent;     //!< global size: globalMax - globalMin

   // local bounds data
   real3 localMin;         //!< minimum coordinate on local processor
   real3 localMax;         //!< maximum coordinate on local processor
   real3 localExtent;      //!< localMax - localMin
} Domain;

struct DomainSt* initDecomposition(int xproc, int yproc, int zproc,
                                   real3 globalExtent);

/// Find the MPI rank of a neighbor domain from a relative coordinate.
int processorNum(Domain* domain, int dix, int diy, int dik);

#endif
