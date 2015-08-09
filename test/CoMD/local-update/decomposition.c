/// \file
/// Parallel domain decomposition.  This version of CoMD uses a simple
/// spatial Cartesian domain decomposition.  The simulation box is
/// divided into equal size bricks by a grid that is xproc by yproc by
/// zproc in size.

#include "decomposition.h"

#include <assert.h>

#include "memUtils.h"
#include "parallel.h"

/// \param [in] xproc x-size of domain decomposition grid.
/// \param [in] yproc y-size of domain decomposition grid.
/// \param [in] zproc z-size of domain decomposition grid.
/// \param [in] globalExtent Size of the simulation domain (in Angstroms).
Domain* initDecomposition(int xproc, int yproc, int zproc, real3 globalExtent)
{
   assert( xproc * yproc * zproc == getNRanks());

   Domain* dd = (Domain*)comdMalloc(sizeof(Domain));
   dd->procGrid[0] = xproc;
   dd->procGrid[1] = yproc;
   dd->procGrid[2] = zproc;
   // calculate grid coordinates i,j,k for this processor
   int myRank = getMyRank();
   dd->procCoord[0] = myRank % dd->procGrid[0];
   myRank /= dd->procGrid[0];
   dd->procCoord[1] = myRank % dd->procGrid[1];
   dd->procCoord[2] = myRank / dd->procGrid[1];

   // initialialize global bounds
   for (int i = 0; i < 3; i++)
   {
      dd->globalMin[i] = 0;
      dd->globalMax[i] = globalExtent[i];
      dd->globalExtent[i] = dd->globalMax[i] - dd->globalMin[i];
   }
   
   // initialize local bounds on this processor
   for (int i = 0; i < 3; i++)
   {
      dd->localExtent[i] = dd->globalExtent[i] / dd->procGrid[i];
      dd->localMin[i] = dd->globalMin[i] +  dd->procCoord[i]    * dd->localExtent[i];
      dd->localMax[i] = dd->globalMin[i] + (dd->procCoord[i]+1) * dd->localExtent[i];
   }

   return dd;
}

/// \details
/// Calculates the rank of the processor with grid coordinates
/// (ix+dix, iy+diy, iz+diz) where (ix, iy, iz) are the grid coordinates
/// of the local rank.  Assumes periodic boundary conditions.  The
/// deltas cannot be smaller than -procGrid[ii].
int processorNum(Domain* domain, int dix, int diy, int diz)
{
   const int* procCoord = domain->procCoord; // alias
   const int* procGrid  = domain->procGrid;  // alias
   int ix = (procCoord[0] + dix + procGrid[0]) % procGrid[0];
   int iy = (procCoord[1] + diy + procGrid[1]) % procGrid[1];
   int iz = (procCoord[2] + diz + procGrid[2]) % procGrid[2];

   return ix + procGrid[0] *(iy + procGrid[1]*iz);
}
