/*
  This is a Version 2.0 MPI + OpenMP implementation of LULESH

                 Copyright (c) 2010-2013.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 2.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
* Addition of regions to make work more representative of multi-material codes
* Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
  more representative of our actual working set sizes
* Single source distribution supports pure serial, pure OpenMP, MPI-only, 
  and MPI+OpenMP
* Addition of ability to visualize the mesh using VisIt 
  https://wci.llnl.gov/codes/visit/download.html
* Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
      printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
      printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
      printf(" -p              : Print out progress\n");
      printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
      printf(" -h              : This message\n");
      printf("\n\n");

*Notable changes in LULESH 2.0

* Split functionality into different files
lulesh.cc - where most (all?) of the timed functionality lies
lulesh-comm.cc - MPI functionality
lulesh-init.cc - Setup code
lulesh-viz.cc  - Support for visualization option
lulesh-util.cc - Non-timed functions
*
* The concept of "regions" was added, although every region is the same ideal
*    gas material, and the same sedov blast wave problem is still the only
*    problem its hardcoded to solve.
* Regions allow two things important to making this proxy app more representative:
*   Four of the LULESH routines are now performed on a region-by-region basis,
*     making the memory access patterns non-unit stride
*   Artificial load imbalances can be easily introduced that could impact
*     parallelization strategies.  
* The load balance flag changes region assignment.  Region number is raised to
*   the power entered for assignment probability.  Most likely regions changes
*   with MPI process id.
* The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
*   entered multiple. The cost of 5% is 10x the entered multiple.
* MPI and OpenMP were added, and coalesced into a single version of the source
*   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked
*   with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which
*   results in an additional reduction.
* Default domain (mesh) size reduced from 45^3 to 30^3
* Command line options to allow numerous test cases without needing to recompile
* Performance optimizations and code cleanup beyond LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and
*   output in support of using LULESH 2.0 for the 2017 CORAL procurement
*
* Possible Differences in Final Release (other changes possible)
*
* High Level mesh structure to allow data structure transformations
* Different default parameters
* Minor code performance changes and cleanup

TODO in future versions
* Add reader for (truly) unstructured meshes, probably serial only
* CMake based build system

//////////////

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include <climits>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>

#include "lulesh.h"
#include "Timer.hxx"

#define RAJA_STORAGE static inline
//#define RAJA_STORAGE 

/* Manage temporary allocations with a pool */
RAJA::MemoryPool< Real_t > elemMemPool ;

/******************************************/

/* Work Routines */

RAJA_STORAGE
void TimeIncrement(Domain& domain)
{
   Real_t targetdt = domain.stoptime() - domain.time() ;

   if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
      Real_t ratio ;
      Real_t olddt = domain.deltatime() ;

      /* This will require a reduction in parallel */
      Real_t gnewdt = Real_t(1.0e+20) ;
      Real_t newdt ;
      if (domain.dtcourant() < gnewdt) {
         gnewdt = domain.dtcourant() / Real_t(2.0) ;
      }
      if (domain.dthydro() < gnewdt) {
         gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0) ;
      }

#if USE_MPI      
      MPI_Allreduce(&gnewdt, &newdt, 1,
                    ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_MIN, MPI_COMM_WORLD) ;
#else
      newdt = gnewdt;
#endif
      
      ratio = newdt / olddt ;
      if (ratio >= Real_t(1.0)) {
         if (ratio < domain.deltatimemultlb()) {
            newdt = olddt ;
         }
         else if (ratio > domain.deltatimemultub()) {
            newdt = olddt*domain.deltatimemultub() ;
         }
      }

      if (newdt > domain.dtmax()) {
         newdt = domain.dtmax() ;
      }
      domain.deltatime() = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > domain.deltatime()) &&
       (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0))) ) {
      targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0) ;
   }

   if (targetdt < domain.deltatime()) {
      domain.deltatime() = targetdt ;
   }

   domain.time() += domain.deltatime() ;

   ++domain.cycle() ;
}

/******************************************/

RAJA_STORAGE
void CollectDomainNodesToElemNodes(Domain* domain,
                                   const Index_t* elemToNode,
                                   Real_t elemX[8],
                                   Real_t elemY[8],
                                   Real_t elemZ[8])
{
   Index_t nd0i = elemToNode[0] ;
   Index_t nd1i = elemToNode[1] ;
   Index_t nd2i = elemToNode[2] ;
   Index_t nd3i = elemToNode[3] ;
   Index_t nd4i = elemToNode[4] ;
   Index_t nd5i = elemToNode[5] ;
   Index_t nd6i = elemToNode[6] ;
   Index_t nd7i = elemToNode[7] ;

   elemX[0] = domain->x(nd0i);
   elemX[1] = domain->x(nd1i);
   elemX[2] = domain->x(nd2i);
   elemX[3] = domain->x(nd3i);
   elemX[4] = domain->x(nd4i);
   elemX[5] = domain->x(nd5i);
   elemX[6] = domain->x(nd6i);
   elemX[7] = domain->x(nd7i);

   elemY[0] = domain->y(nd0i);
   elemY[1] = domain->y(nd1i);
   elemY[2] = domain->y(nd2i);
   elemY[3] = domain->y(nd3i);
   elemY[4] = domain->y(nd4i);
   elemY[5] = domain->y(nd5i);
   elemY[6] = domain->y(nd6i);
   elemY[7] = domain->y(nd7i);

   elemZ[0] = domain->z(nd0i);
   elemZ[1] = domain->z(nd1i);
   elemZ[2] = domain->z(nd2i);
   elemZ[3] = domain->z(nd3i);
   elemZ[4] = domain->z(nd4i);
   elemZ[5] = domain->z(nd5i);
   elemZ[6] = domain->z(nd6i);
   elemZ[7] = domain->z(nd7i);

}

/******************************************/

RAJA_STORAGE
void InitStressTermsForElems(Domain* domain,
                             Real_t *sigxx, Real_t *sigyy, Real_t *sigzz)
{
   //
   // pull in the stresses appropriate to the hydro integration
   //

   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
      [=] RAJA_DEVICE (int i) {
      sigxx[i] = sigyy[i] = sigzz[i] =  - domain->p(i) - domain->q(i) ;
   } );
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
void CalcElemShapeFunctionDerivatives( Real_t const x[],
                                       Real_t const y[],
                                       Real_t const z[],
                                       Real_t b[][8],
                                       Real_t* const volume )
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = Real_t(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = Real_t(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = Real_t(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = Real_t(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = Real_t(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = Real_t(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = Real_t(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = Real_t(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = Real_t(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                       Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                       Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                       const Real_t x0, const Real_t y0, const Real_t z0,
                       const Real_t x1, const Real_t y1, const Real_t z1,
                       const Real_t x2, const Real_t y2, const Real_t z2,
                       const Real_t x3, const Real_t y3, const Real_t z3)
{
   Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
void CalcElemNodeNormals(Real_t pfx[8],
                         Real_t pfy[8],
                         Real_t pfz[8],
                         const Real_t x[8],
                         const Real_t y[8],
                         const Real_t z[8])
{
   for (Index_t i = 0 ; i < 8 ; ++i) {
      pfx[i] = Real_t(0.0);
      pfy[i] = Real_t(0.0);
      pfz[i] = Real_t(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
void SumElemStressesToNodeForces( const Real_t B[][8],
                                  const Real_t stress_xx,
                                  const Real_t stress_yy,
                                  const Real_t stress_zz,
                                  Real_t* fx, Real_t* fy, Real_t* fz )
{
   for(Index_t i = 0; i < 8; i++) {
      fx[i] = -( stress_xx * B[0][i] );
      fy[i] = -( stress_yy * B[1][i]  );
      fz[i] = -( stress_zz * B[2][i] );
   }
}

/******************************************/

RAJA_STORAGE
void IntegrateStressForElems( Domain* domain,
                              Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                              Real_t *determ, Index_t numElem)
{
#if defined(OMP_FINE_SYNC)
  Real_t *fx_elem = elemMemPool.allocate(numElem*8) ;
  Real_t *fy_elem = elemMemPool.allocate(numElem*8) ;
  Real_t *fz_elem = elemMemPool.allocate(numElem*8) ;
#endif

  // loop over all elements
  RAJA::forall<elem_exec_policy>(domain->getElemISet(),
     [=] RAJA_DEVICE (int k) {
    const Index_t* const elemToNode = domain->nodelist(k);
    Real_t B[3][8] __attribute__((aligned(32))) ;// shape function derivatives
    Real_t x_local[8] __attribute__((aligned(32))) ;
    Real_t y_local[8] __attribute__((aligned(32))) ;
    Real_t z_local[8] __attribute__((aligned(32))) ;
#if !defined(OMP_FINE_SYNC)
    Real_t fx_local[8] __attribute__((aligned(32))) ;
    Real_t fy_local[8] __attribute__((aligned(32))) ;
    Real_t fz_local[8] __attribute__((aligned(32))) ;
#endif

    // get nodal coordinates from global arrays and copy into local arrays.
    CollectDomainNodesToElemNodes(domain, elemToNode,
                                  x_local, y_local, z_local);

    // Volume calculation involves extra work for numerical consistency
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                         B, &determ[k]);

    CalcElemNodeNormals( B[0] , B[1], B[2],
                          x_local, y_local, z_local );

    SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
#if !defined(OMP_FINE_SYNC)
                                 fx_local, fy_local, fz_local
#else
                                 &fx_elem[k*8], &fy_elem[k*8], &fz_elem[k*8]
#endif
                       ) ;

#if !defined(OMP_FINE_SYNC)
    // copy nodal force contributions to global force arrray.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode ) {
       Index_t gnode = elemToNode[lnode];
       domain->fx(gnode) += fx_local[lnode];
       domain->fy(gnode) += fy_local[lnode];
       domain->fz(gnode) += fz_local[lnode];
    }
#endif
  } );

#if defined(OMP_FINE_SYNC)
  RAJA::forall<node_exec_policy>(domain->getNodeISet(),
                                 [=] RAJA_DEVICE (int gnode) {
     Index_t count = domain->nodeElemCount(gnode) ;
     Index_t *cornerList = domain->nodeElemCornerList(gnode) ;
     Real_t fx_sum = Real_t(0.0) ;
     Real_t fy_sum = Real_t(0.0) ;
     Real_t fz_sum = Real_t(0.0) ;
     for (Index_t i=0 ; i < count ; ++i) {
        Index_t ielem = cornerList[i] ;
        fx_sum += fx_elem[ielem] ;
        fy_sum += fy_elem[ielem] ;
        fz_sum += fz_elem[ielem] ;
     }
     domain->fx(gnode) = fx_sum ;
     domain->fy(gnode) = fy_sum ;
     domain->fz(gnode) = fz_sum ;
  } );

  elemMemPool.release(&fz_elem) ;
  elemMemPool.release(&fy_elem) ;
  elemMemPool.release(&fx_elem) ;
#endif
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8])
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t hourgam[][4],
                              Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
   Real_t hxx[4];
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
               hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
               hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
               hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfx[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
               hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
               hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
               hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfy[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for(Index_t i = 0; i < 4; i++) {
      hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
               hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
               hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
               hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
   }
   for(Index_t i = 0; i < 8; i++) {
      hgfz[i] = coefficient *
                (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                 hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
}

/******************************************/

RAJA_STORAGE
void CalcFBHourglassForceForElems( Domain* domain,
                                   Real_t *determ,
                                   Real_t *x8n, Real_t *y8n, Real_t *z8n,
                                   Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                                   Real_t hourg, Index_t numElem)
{
  /*************************************************
   *
   *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
   *               force.
   *
   *************************************************/
  
#if defined(OMP_FINE_SYNC)
   Real_t *fx_elem = elemMemPool.allocate(numElem*8) ;
   Real_t *fy_elem = elemMemPool.allocate(numElem*8) ;
   Real_t *fz_elem = elemMemPool.allocate(numElem*8) ;
#endif

/*************************************************/
/*    compute the hourglass modes */


   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
      [=] RAJA_DEVICE (int i2) {

#if !defined(OMP_FINE_SYNC)
      Real_t hgfx[8], hgfy[8], hgfz[8] ;
#endif

      Real_t coefficient;

      Real_t hourgam[8][4];
      Real_t xd1[8], yd1[8], zd1[8] ;

      // Define this here so code works on both host and device
      const Real_t gamma[4][8] =
      {
        { Real_t( 1.), Real_t( 1.), Real_t(-1.), Real_t(-1.),
          Real_t(-1.), Real_t(-1.), Real_t( 1.), Real_t( 1.) },

        { Real_t( 1.), Real_t(-1.), Real_t(-1.), Real_t( 1.),
          Real_t(-1.), Real_t( 1.), Real_t( 1.), Real_t(-1.) },

        { Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.),
          Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.) },

        { Real_t(-1.), Real_t( 1.), Real_t(-1.), Real_t( 1.),
          Real_t( 1.), Real_t(-1.), Real_t( 1.), Real_t(-1.) }
      } ;

      const Index_t *elemToNode = domain->nodelist(i2);
      Index_t i3=8*i2;
      Real_t volinv=Real_t(1.0)/determ[i2];
      Real_t ss1, mass1, volume13 ;
      for(Index_t i1=0;i1<4;++i1){

         Real_t hourmodx =
            x8n[i3] * gamma[i1][0] + x8n[i3+1] * gamma[i1][1] +
            x8n[i3+2] * gamma[i1][2] + x8n[i3+3] * gamma[i1][3] +
            x8n[i3+4] * gamma[i1][4] + x8n[i3+5] * gamma[i1][5] +
            x8n[i3+6] * gamma[i1][6] + x8n[i3+7] * gamma[i1][7];

         Real_t hourmody =
            y8n[i3] * gamma[i1][0] + y8n[i3+1] * gamma[i1][1] +
            y8n[i3+2] * gamma[i1][2] + y8n[i3+3] * gamma[i1][3] +
            y8n[i3+4] * gamma[i1][4] + y8n[i3+5] * gamma[i1][5] +
            y8n[i3+6] * gamma[i1][6] + y8n[i3+7] * gamma[i1][7];

         Real_t hourmodz =
            z8n[i3] * gamma[i1][0] + z8n[i3+1] * gamma[i1][1] +
            z8n[i3+2] * gamma[i1][2] + z8n[i3+3] * gamma[i1][3] +
            z8n[i3+4] * gamma[i1][4] + z8n[i3+5] * gamma[i1][5] +
            z8n[i3+6] * gamma[i1][6] + z8n[i3+7] * gamma[i1][7];

         hourgam[0][i1] = gamma[i1][0] -  volinv*(dvdx[i3  ] * hourmodx +
                                                  dvdy[i3  ] * hourmody +
                                                  dvdz[i3  ] * hourmodz );

         hourgam[1][i1] = gamma[i1][1] -  volinv*(dvdx[i3+1] * hourmodx +
                                                  dvdy[i3+1] * hourmody +
                                                  dvdz[i3+1] * hourmodz );

         hourgam[2][i1] = gamma[i1][2] -  volinv*(dvdx[i3+2] * hourmodx +
                                                  dvdy[i3+2] * hourmody +
                                                  dvdz[i3+2] * hourmodz );

         hourgam[3][i1] = gamma[i1][3] -  volinv*(dvdx[i3+3] * hourmodx +
                                                  dvdy[i3+3] * hourmody +
                                                  dvdz[i3+3] * hourmodz );

         hourgam[4][i1] = gamma[i1][4] -  volinv*(dvdx[i3+4] * hourmodx +
                                                  dvdy[i3+4] * hourmody +
                                                  dvdz[i3+4] * hourmodz );

         hourgam[5][i1] = gamma[i1][5] -  volinv*(dvdx[i3+5] * hourmodx +
                                                  dvdy[i3+5] * hourmody +
                                                  dvdz[i3+5] * hourmodz );

         hourgam[6][i1] = gamma[i1][6] -  volinv*(dvdx[i3+6] * hourmodx +
                                                  dvdy[i3+6] * hourmody +
                                                  dvdz[i3+6] * hourmodz );

         hourgam[7][i1] = gamma[i1][7] -  volinv*(dvdx[i3+7] * hourmodx +
                                                  dvdy[i3+7] * hourmody +
                                                  dvdz[i3+7] * hourmodz );

      }

      /* compute forces */
      /* store forces into h arrays (force arrays) */

      ss1=domain->ss(i2);
      mass1=domain->elemMass(i2);
      volume13=CBRT(determ[i2]);

      Index_t n0si2 = elemToNode[0];
      Index_t n1si2 = elemToNode[1];
      Index_t n2si2 = elemToNode[2];
      Index_t n3si2 = elemToNode[3];
      Index_t n4si2 = elemToNode[4];
      Index_t n5si2 = elemToNode[5];
      Index_t n6si2 = elemToNode[6];
      Index_t n7si2 = elemToNode[7];

      xd1[0] = domain->xd(n0si2);
      xd1[1] = domain->xd(n1si2);
      xd1[2] = domain->xd(n2si2);
      xd1[3] = domain->xd(n3si2);
      xd1[4] = domain->xd(n4si2);
      xd1[5] = domain->xd(n5si2);
      xd1[6] = domain->xd(n6si2);
      xd1[7] = domain->xd(n7si2);

      yd1[0] = domain->yd(n0si2);
      yd1[1] = domain->yd(n1si2);
      yd1[2] = domain->yd(n2si2);
      yd1[3] = domain->yd(n3si2);
      yd1[4] = domain->yd(n4si2);
      yd1[5] = domain->yd(n5si2);
      yd1[6] = domain->yd(n6si2);
      yd1[7] = domain->yd(n7si2);

      zd1[0] = domain->zd(n0si2);
      zd1[1] = domain->zd(n1si2);
      zd1[2] = domain->zd(n2si2);
      zd1[3] = domain->zd(n3si2);
      zd1[4] = domain->zd(n4si2);
      zd1[5] = domain->zd(n5si2);
      zd1[6] = domain->zd(n6si2);
      zd1[7] = domain->zd(n7si2);

      coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

      CalcElemFBHourglassForce(xd1,yd1,zd1, hourgam, coefficient,
#if !defined(OMP_FINE_SYNC)
                               hgfx, hgfy, hgfz
#else
                               &fx_elem[i3], &fy_elem[i3], &fz_elem[i3]
#endif
                              );

#if !defined(OMP_FINE_SYNC)
      domain->fx(n0si2) += hgfx[0];
      domain->fy(n0si2) += hgfy[0];
      domain->fz(n0si2) += hgfz[0];

      domain->fx(n1si2) += hgfx[1];
      domain->fy(n1si2) += hgfy[1];
      domain->fz(n1si2) += hgfz[1];

      domain->fx(n2si2) += hgfx[2];
      domain->fy(n2si2) += hgfy[2];
      domain->fz(n2si2) += hgfz[2];

      domain->fx(n3si2) += hgfx[3];
      domain->fy(n3si2) += hgfy[3];
      domain->fz(n3si2) += hgfz[3];

      domain->fx(n4si2) += hgfx[4];
      domain->fy(n4si2) += hgfy[4];
      domain->fz(n4si2) += hgfz[4];

      domain->fx(n5si2) += hgfx[5];
      domain->fy(n5si2) += hgfy[5];
      domain->fz(n5si2) += hgfz[5];

      domain->fx(n6si2) += hgfx[6];
      domain->fy(n6si2) += hgfy[6];
      domain->fz(n6si2) += hgfz[6];

      domain->fx(n7si2) += hgfx[7];
      domain->fy(n7si2) += hgfy[7];
      domain->fz(n7si2) += hgfz[7];
#endif
   } );

#if defined(OMP_FINE_SYNC)
   // Collect the data from the local arrays into the final force arrays
   RAJA::forall<node_exec_policy>(domain->getNodeISet(),
                                  [=] RAJA_DEVICE (int gnode) {
      Index_t count = domain->nodeElemCount(gnode) ;
      Index_t *cornerList = domain->nodeElemCornerList(gnode) ;
      Real_t fx_sum = Real_t(0.0) ;
      Real_t fy_sum = Real_t(0.0) ;
      Real_t fz_sum = Real_t(0.0) ;
      for (Index_t i=0 ; i < count ; ++i) {
         Index_t ielem = cornerList[i] ;
         fx_sum += fx_elem[ielem] ;
         fy_sum += fy_elem[ielem] ;
         fz_sum += fz_elem[ielem] ;
      }
      domain->fx(gnode) += fx_sum ;
      domain->fy(gnode) += fy_sum ;
      domain->fz(gnode) += fz_sum ;
   } );

   elemMemPool.release(&fz_elem) ;
   elemMemPool.release(&fy_elem) ;
   elemMemPool.release(&fx_elem) ;
#endif
}

/******************************************/

RAJA_STORAGE
void CalcHourglassControlForElems(Domain* domain,
                                  Real_t determ[], Real_t hgcoef)
{
   Index_t numElem = domain->numElem() ;
   Index_t numElem8 = numElem * 8 ;
   Real_t *dvdx = elemMemPool.allocate(numElem8) ;
   Real_t *dvdy = elemMemPool.allocate(numElem8) ;
   Real_t *dvdz = elemMemPool.allocate(numElem8) ;
   Real_t *x8n  = elemMemPool.allocate(numElem8) ;
   Real_t *y8n  = elemMemPool.allocate(numElem8) ;
   Real_t *z8n  = elemMemPool.allocate(numElem8) ;

   // For negative element volume check
   RAJA::ReduceMin<reduce_policy, Real_t> minvol(Real_t(1.0e+20));

   /* start loop over elements */
   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
        [=] RAJA_DEVICE (int i) {
#if 1
      /* This variant makes overall runtime 2% faster on CPU */
      Real_t  x1[8],  y1[8],  z1[8] ;
      Real_t pfx[8], pfy[8], pfz[8] ;

      Index_t* elemToNode = domain->nodelist(i);
      CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

      CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

      /* load into temporary storage for FB Hour Glass control */
      for(Index_t ii=0;ii<8;++ii) {
         Index_t jj=8*i+ii;

         dvdx[jj] = pfx[ii];
         dvdy[jj] = pfy[ii];
         dvdz[jj] = pfz[ii];
         x8n[jj]  = x1[ii];
         y8n[jj]  = y1[ii];
         z8n[jj]  = z1[ii];
      }
#else
      /* This variant is likely GPU friendly */
      Index_t* elemToNode = domain->nodelist(i);
      CollectDomainNodesToElemNodes(domain, elemToNode,
                                    &x8n[8*i], &y8n[8*i], &z8n[8*i]);

      CalcElemVolumeDerivative(&dvdx[8*i], &dvdy[8*i], &dvdz[8*i],
                               &x8n[8*i], &y8n[8*i], &z8n[8*i]);
#endif

      determ[i] = domain->volo(i) * domain->v(i);

      minvol.min(domain->v(i));

   } );

   /* Do a check for negative volumes */
   if ( Real_t(minvol) <= Real_t(0.0) ) {
#if USE_MPI         
      MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
      exit(VolumeError);
#endif
   }

   if ( hgcoef > Real_t(0.) ) {
      CalcFBHourglassForceForElems( domain,
                                    determ, x8n, y8n, z8n, dvdx, dvdy, dvdz,
                                    hgcoef, numElem ) ;
   }

   elemMemPool.release(&z8n) ;
   elemMemPool.release(&y8n) ;
   elemMemPool.release(&x8n) ;
   elemMemPool.release(&dvdz) ;
   elemMemPool.release(&dvdy) ;
   elemMemPool.release(&dvdx) ;

   return ;
}

/******************************************/

RAJA_STORAGE
void CalcVolumeForceForElems(Domain* domain)
{
   Index_t numElem = domain->numElem() ;
   if (numElem != 0) {
      Real_t  hgcoef = domain->hgcoef() ;
      Real_t *sigxx  = elemMemPool.allocate(numElem) ;
      Real_t *sigyy  = elemMemPool.allocate(numElem) ;
      Real_t *sigzz  = elemMemPool.allocate(numElem) ;
      Real_t *determ = elemMemPool.allocate(numElem) ;

      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(domain, sigxx, sigyy, sigzz);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( domain,
                               sigxx, sigyy, sigzz, determ, numElem );

      // check for negative element volume
      RAJA::ReduceMin<reduce_policy, Real_t> minvol(Real_t(1.0e+20));
      RAJA::forall<elem_exec_policy>(domain->getElemISet(),
           [=] RAJA_DEVICE (int k) {
         minvol.min(determ[k]);
      } );

      if (Real_t(minvol) <= Real_t(0.0)) {
#if USE_MPI            
         MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
         exit(VolumeError);
#endif
      }

      CalcHourglassControlForElems(domain, determ, hgcoef) ;

      elemMemPool.release(&determ) ;
      elemMemPool.release(&sigzz) ;
      elemMemPool.release(&sigyy) ;
      elemMemPool.release(&sigxx) ;
   }
}

/******************************************/

RAJA_STORAGE void CalcForceForNodes(Domain* domain)
{
#if USE_MPI  
  CommRecv(*domain, MSG_COMM_SBN, 3,
           domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
           true, false) ;
#endif  

  RAJA::forall<node_exec_policy>(domain->getNodeISet(),
       [=] RAJA_DEVICE (int i) {
     domain->fx(i) = Real_t(0.0) ;
     domain->fy(i) = Real_t(0.0) ;
     domain->fz(i) = Real_t(0.0) ;
  } );

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems(domain) ;

#if USE_MPI  
  Domain_member fieldData[3] ;
  fieldData[0] = &Domain::fx ;
  fieldData[1] = &Domain::fy ;
  fieldData[2] = &Domain::fz ;
  
  CommSend(*domain, MSG_COMM_SBN, 3, fieldData,
           domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() +  1,
           true, false) ;
  CommSBN(*domain, 3, fieldData) ;
#endif  
}

/******************************************/

RAJA_STORAGE
void CalcAccelerationForNodes(Domain* domain)
{
   
   RAJA::forall<node_exec_policy>(domain->getNodeISet(),
        [=] RAJA_DEVICE (int i) {
      domain->xdd(i) = domain->fx(i) / domain->nodalMass(i);
      domain->ydd(i) = domain->fy(i) / domain->nodalMass(i);
      domain->zdd(i) = domain->fz(i) / domain->nodalMass(i);
   } );
}

/******************************************/

RAJA_STORAGE
void ApplyAccelerationBoundaryConditionsForNodes(Domain* domain)
{
   RAJA::forall<symnode_exec_policy>(domain->getXSymNodeISet(),
        [=] RAJA_DEVICE (int i) {
      domain->xdd(i) = Real_t(0.0) ;
   } );

   RAJA::forall<symnode_exec_policy>(domain->getYSymNodeISet(),
        [=] RAJA_DEVICE (int i) {
      domain->ydd(i) = Real_t(0.0) ;
   } );

   RAJA::forall<symnode_exec_policy>(domain->getZSymNodeISet(),
        [=] RAJA_DEVICE (int i) {
      domain->zdd(i) = Real_t(0.0) ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcVelocityForNodes(Domain* domain, const Real_t dt, const Real_t u_cut)
{

   RAJA::forall<node_exec_policy>(domain->getNodeISet(),
       [=] RAJA_DEVICE (int i) {
     Real_t xdtmp, ydtmp, zdtmp ;

     xdtmp = domain->xd(i) + domain->xdd(i) * dt ;
     if( FABS(xdtmp) < u_cut ) xdtmp = Real_t(0.0);
     domain->xd(i) = xdtmp ;

     ydtmp = domain->yd(i) + domain->ydd(i) * dt ;
     if( FABS(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
     domain->yd(i) = ydtmp ;

     zdtmp = domain->zd(i) + domain->zdd(i) * dt ;
     if( FABS(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
     domain->zd(i) = zdtmp ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcPositionForNodes(Domain* domain, const Real_t dt)
{
   RAJA::forall<node_exec_policy>(domain->getNodeISet(),
       [=] RAJA_DEVICE (int i) {
     domain->x(i) += domain->xd(i) * dt ;
     domain->y(i) += domain->yd(i) * dt ;
     domain->z(i) += domain->zd(i) * dt ;
   } );
}

/******************************************/

RAJA_STORAGE
void LagrangeNodal(Domain* domain)
{
#if defined(SEDOV_SYNC_POS_VEL_EARLY)
   Domain_member fieldData[6] ;
#endif

   const Real_t delt = domain->deltatime() ;
   Real_t u_cut = domain->u_cut() ;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes(domain);

#if USE_MPI  
#if defined(SEDOV_SYNC_POS_VEL_EARLY)
   CommRecv(*domain, MSG_SYNC_POS_VEL, 6,
            domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
            false, false) ;
#endif
#endif
   
   CalcAccelerationForNodes(domain);
   
   ApplyAccelerationBoundaryConditionsForNodes(domain);

   CalcVelocityForNodes( domain, delt, u_cut) ;

   CalcPositionForNodes( domain, delt );
#if USE_MPI
#if defined(SEDOV_SYNC_POS_VEL_EARLY)
  fieldData[0] = &Domain::x ;
  fieldData[1] = &Domain::y ;
  fieldData[2] = &Domain::z ;
  fieldData[3] = &Domain::xd ;
  fieldData[4] = &Domain::yd ;
  fieldData[5] = &Domain::zd ;

   CommSend(*domain, MSG_SYNC_POS_VEL, 6, fieldData,
            domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
            false, false) ;
   CommSyncPosVel(*domain) ;
#endif
#endif
   
  return;
}

/******************************************/

RAJA_STORAGE
RAJA_HOST_DEVICE
Real_t CalcElemVolume( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
               const Real_t z6, const Real_t z7 )
{
  Real_t twelveth = Real_t(1.0)/Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

/******************************************/

//inline
RAJA_HOST_DEVICE
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
Real_t AreaFace( const Real_t x0, const Real_t x1,
                 const Real_t x2, const Real_t x3,
                 const Real_t y0, const Real_t y1,
                 const Real_t y2, const Real_t y3,
                 const Real_t z0, const Real_t z1,
                 const Real_t z2, const Real_t z3)
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      (fx * gx + fy * gy + fz * gz) *
      (fx * gx + fy * gy + fz * gz);
   return area ;
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
Real_t CalcElemCharacteristicLength( const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8],
                                     const Real_t volume)
{
   Real_t a, charLength = Real_t(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = MAX(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = MAX(a,charLength) ;

   charLength = Real_t(4.0) * volume / SQRT(charLength);

   return charLength;
}

/******************************************/

RAJA_STORAGE
RAJA_DEVICE
void CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d )
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
                     + pfx[1] * (xvel[1]-xvel[7])
                     + pfx[2] * (xvel[2]-xvel[4])
                     + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
                     + pfy[1] * (yvel[1]-yvel[7])
                     + pfy[2] * (yvel[2]-yvel[4])
                     + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
                     + pfz[1] * (zvel[1]-zvel[7])
                     + pfz[2] * (zvel[2]-zvel[4])
                     + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
                      + pfx[1] * (yvel[1]-yvel[7])
                      + pfx[2] * (yvel[2]-yvel[4])
                      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
                      + pfy[1] * (xvel[1]-xvel[7])
                      + pfy[2] * (xvel[2]-xvel[4])
                      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
                      + pfx[1] * (zvel[1]-zvel[7])
                      + pfx[2] * (zvel[2]-zvel[4])
                      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
                      + pfz[1] * (xvel[1]-xvel[7])
                      + pfz[2] * (xvel[2]-xvel[4])
                      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
                      + pfy[1] * (zvel[1]-zvel[7])
                      + pfy[2] * (zvel[2]-zvel[4])
                      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
                      + pfz[1] * (yvel[1]-yvel[7])
                      + pfz[2] * (yvel[2]-yvel[4])
                      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}

/******************************************/

//RAJA_STORAGE
void CalcKinematicsForElems( Domain* domain,
                             Real_t deltaTime, Index_t numElem )
{

  // loop over all elements
  RAJA::forall<elem_exec_policy>(domain->getElemISet(),
      [=] RAJA_DEVICE (int k) { 
    Real_t B[3][8] ; /** shape function derivatives */
    Real_t D[6] ;
    Real_t x_local[8] ;
    Real_t y_local[8] ;
    Real_t z_local[8] ;
    Real_t xd_local[8] ;
    Real_t yd_local[8] ;
    Real_t zd_local[8] ;
    Real_t detJ = Real_t(0.0) ;

    Real_t volume ;
    Real_t relativeVolume ;
    const Index_t* const elemToNode = domain->nodelist(k) ;

    // get nodal coordinates from global arrays and copy into local arrays.
    CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local );
    relativeVolume = volume / domain->volo(k) ;
    domain->vnew(k) = relativeVolume ;
    domain->delv(k) = relativeVolume - domain->v(k) ;

    // set characteristic length
    domain->arealg(k) = CalcElemCharacteristicLength(x_local, y_local, z_local,
                                             volume);

    // get nodal velocities from global array and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemToNode[lnode];
      xd_local[lnode] = domain->xd(gnode);
      yd_local[lnode] = domain->yd(gnode);
      zd_local[lnode] = domain->zd(gnode); 
    }

    Real_t dt2 = Real_t(0.5) * deltaTime;
    for ( Index_t j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j];
    }

    CalcElemShapeFunctionDerivatives( x_local, y_local, z_local,
                                      B, &detJ );

    CalcElemVelocityGradient( xd_local, yd_local, zd_local,
                               B, detJ, D );

    // put velocity gradient quantities into their global arrays.
    domain->dxx(k) = D[0];
    domain->dyy(k) = D[1];
    domain->dzz(k) = D[2];
  } );
}

/******************************************/

RAJA_STORAGE
void CalcLagrangeElements(Domain* domain)
{
   Index_t numElem = domain->numElem() ;
   if (numElem > 0) {
      const Real_t deltatime = domain->deltatime() ;

      domain->AllocateStrains(elemMemPool, numElem);

      CalcKinematicsForElems(domain, deltatime, numElem) ;

      // check for negative element volume
      RAJA::ReduceMin<reduce_policy, Real_t> minvol(Real_t(1.0e+20));

      // element loop to do some stuff not included in the elemlib function.
      RAJA::forall<elem_exec_policy>(domain->getElemISet(),
           [=] RAJA_DEVICE (int k) {
         // calc strain rate and apply as constraint (only done in FB element)
         Real_t vdov = domain->dxx(k) + domain->dyy(k) + domain->dzz(k) ;
         Real_t vdovthird = vdov/Real_t(3.0) ;

         // make the rate of deformation tensor deviatoric
         domain->vdov(k) = vdov ;
         domain->dxx(k) -= vdovthird ;
         domain->dyy(k) -= vdovthird ;
         domain->dzz(k) -= vdovthird ;

         minvol.min(domain->vnew(k));
      } );

      if (Real_t(minvol) <= Real_t(0.0)) {
#if USE_MPI           
         MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
         exit(VolumeError);
#endif
      }

      domain->DeallocateStrains(elemMemPool);
   }
}

/******************************************/

RAJA_STORAGE
void CalcMonotonicQGradientsForElems(Domain* domain)
{
   Index_t numElem = domain->numElem();

   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
        [=] RAJA_DEVICE (int i) {
      const Real_t ptiny = Real_t(1.e-36) ;
      Real_t ax,ay,az ;
      Real_t dxv,dyv,dzv ;

      const Index_t *elemToNode = domain->nodelist(i);
      Index_t n0 = elemToNode[0] ;
      Index_t n1 = elemToNode[1] ;
      Index_t n2 = elemToNode[2] ;
      Index_t n3 = elemToNode[3] ;
      Index_t n4 = elemToNode[4] ;
      Index_t n5 = elemToNode[5] ;
      Index_t n6 = elemToNode[6] ;
      Index_t n7 = elemToNode[7] ;

      Real_t x0 = domain->x(n0) ;
      Real_t x1 = domain->x(n1) ;
      Real_t x2 = domain->x(n2) ;
      Real_t x3 = domain->x(n3) ;
      Real_t x4 = domain->x(n4) ;
      Real_t x5 = domain->x(n5) ;
      Real_t x6 = domain->x(n6) ;
      Real_t x7 = domain->x(n7) ;

      Real_t y0 = domain->y(n0) ;
      Real_t y1 = domain->y(n1) ;
      Real_t y2 = domain->y(n2) ;
      Real_t y3 = domain->y(n3) ;
      Real_t y4 = domain->y(n4) ;
      Real_t y5 = domain->y(n5) ;
      Real_t y6 = domain->y(n6) ;
      Real_t y7 = domain->y(n7) ;

      Real_t z0 = domain->z(n0) ;
      Real_t z1 = domain->z(n1) ;
      Real_t z2 = domain->z(n2) ;
      Real_t z3 = domain->z(n3) ;
      Real_t z4 = domain->z(n4) ;
      Real_t z5 = domain->z(n5) ;
      Real_t z6 = domain->z(n6) ;
      Real_t z7 = domain->z(n7) ;

      Real_t xv0 = domain->xd(n0) ;
      Real_t xv1 = domain->xd(n1) ;
      Real_t xv2 = domain->xd(n2) ;
      Real_t xv3 = domain->xd(n3) ;
      Real_t xv4 = domain->xd(n4) ;
      Real_t xv5 = domain->xd(n5) ;
      Real_t xv6 = domain->xd(n6) ;
      Real_t xv7 = domain->xd(n7) ;

      Real_t yv0 = domain->yd(n0) ;
      Real_t yv1 = domain->yd(n1) ;
      Real_t yv2 = domain->yd(n2) ;
      Real_t yv3 = domain->yd(n3) ;
      Real_t yv4 = domain->yd(n4) ;
      Real_t yv5 = domain->yd(n5) ;
      Real_t yv6 = domain->yd(n6) ;
      Real_t yv7 = domain->yd(n7) ;

      Real_t zv0 = domain->zd(n0) ;
      Real_t zv1 = domain->zd(n1) ;
      Real_t zv2 = domain->zd(n2) ;
      Real_t zv3 = domain->zd(n3) ;
      Real_t zv4 = domain->zd(n4) ;
      Real_t zv5 = domain->zd(n5) ;
      Real_t zv6 = domain->zd(n6) ;
      Real_t zv7 = domain->zd(n7) ;

      Real_t vol = domain->volo(i)*domain->vnew(i) ;
      Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

      Real_t dxj = Real_t(-0.25)*((x0+x1+x5+x4) - (x3+x2+x6+x7)) ;
      Real_t dyj = Real_t(-0.25)*((y0+y1+y5+y4) - (y3+y2+y6+y7)) ;
      Real_t dzj = Real_t(-0.25)*((z0+z1+z5+z4) - (z3+z2+z6+z7)) ;

      Real_t dxi = Real_t( 0.25)*((x1+x2+x6+x5) - (x0+x3+x7+x4)) ;
      Real_t dyi = Real_t( 0.25)*((y1+y2+y6+y5) - (y0+y3+y7+y4)) ;
      Real_t dzi = Real_t( 0.25)*((z1+z2+z6+z5) - (z0+z3+z7+z4)) ;

      Real_t dxk = Real_t( 0.25)*((x4+x5+x6+x7) - (x0+x1+x2+x3)) ;
      Real_t dyk = Real_t( 0.25)*((y4+y5+y6+y7) - (y0+y1+y2+y3)) ;
      Real_t dzk = Real_t( 0.25)*((z4+z5+z6+z7) - (z0+z1+z2+z3)) ;

      /* find delvk and delxk ( i cross j ) */

      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      domain->delx_zeta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*((xv4+xv5+xv6+xv7) - (xv0+xv1+xv2+xv3)) ;
      dyv = Real_t(0.25)*((yv4+yv5+yv6+yv7) - (yv0+yv1+yv2+yv3)) ;
      dzv = Real_t(0.25)*((zv4+zv5+zv6+zv7) - (zv0+zv1+zv2+zv3)) ;

      domain->delv_zeta(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */

      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      domain->delx_xi(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*((xv1+xv2+xv6+xv5) - (xv0+xv3+xv7+xv4)) ;
      dyv = Real_t(0.25)*((yv1+yv2+yv6+yv5) - (yv0+yv3+yv7+yv4)) ;
      dzv = Real_t(0.25)*((zv1+zv2+zv6+zv5) - (zv0+zv3+zv7+zv4)) ;

      domain->delv_xi(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */

      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      domain->delx_eta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(-0.25)*((xv0+xv1+xv5+xv4) - (xv3+xv2+xv6+xv7)) ;
      dyv = Real_t(-0.25)*((yv0+yv1+yv5+yv4) - (yv3+yv2+yv6+yv7)) ;
      dzv = Real_t(-0.25)*((zv0+zv1+zv5+zv4) - (zv3+zv2+zv6+zv7)) ;

      domain->delv_eta(i) = ax*dxv + ay*dyv + az*dzv ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcMonotonicQRegionForElems(Domain* domain, Int_t r,
                                  Real_t ptiny)
{
   Real_t monoq_limiter_mult = domain->monoq_limiter_mult();
   Real_t monoq_max_slope = domain->monoq_max_slope();
   Real_t qlc_monoq = domain->qlc_monoq();
   Real_t qqc_monoq = domain->qqc_monoq();

   RAJA::forall<mat_exec_policy>(domain->getRegionISet(r),
        [=] RAJA_DEVICE (int ielem) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Int_t bcMask = domain->elemBC(ielem) ;
      Real_t delvm = 0.0, delvp =0.0;

      /*  phixi     */
      Real_t norm = Real_t(1.) / (domain->delv_xi(ielem)+ ptiny ) ;

      switch (bcMask & XI_M) {
         case XI_M_COMM: /* needs comm data */
         case 0:         delvm = domain->delv_xi(domain->lxim(ielem)); break ;
         case XI_M_SYMM: delvm = domain->delv_xi(ielem) ;       break ;
         case XI_M_FREE: delvm = Real_t(0.0) ;      break ;
         default:        /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvm = 0; /* ERROR - but quiets the compiler */
            break;
      }
      switch (bcMask & XI_P) {
         case XI_P_COMM: /* needs comm data */
         case 0:         delvp = domain->delv_xi(domain->lxip(ielem)) ; break ;
         case XI_P_SYMM: delvp = domain->delv_xi(ielem) ;       break ;
         case XI_P_FREE: delvp = Real_t(0.0) ;      break ;
         default:        /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvp = 0; /* ERROR - but quiets the compiler */
            break;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = Real_t(1.) / ( domain->delv_eta(ielem) + ptiny ) ;

      switch (bcMask & ETA_M) {
         case ETA_M_COMM: /* needs comm data */
         case 0:          delvm = domain->delv_eta(domain->letam(ielem)) ; break ;
         case ETA_M_SYMM: delvm = domain->delv_eta(ielem) ;        break ;
         case ETA_M_FREE: delvm = Real_t(0.0) ;        break ;
         default:         /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvm = 0; /* ERROR - but quiets the compiler */
            break;
      }
      switch (bcMask & ETA_P) {
         case ETA_P_COMM: /* needs comm data */
         case 0:          delvp = domain->delv_eta(domain->letap(ielem)) ; break ;
         case ETA_P_SYMM: delvp = domain->delv_eta(ielem) ;        break ;
         case ETA_P_FREE: delvp = Real_t(0.0) ;        break ;
         default:         /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvp = 0; /* ERROR - but quiets the compiler */
            break;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = Real_t(1.) / ( domain->delv_zeta(ielem) + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case ZETA_M_COMM: /* needs comm data */
         case 0:           delvm = domain->delv_zeta(domain->lzetam(ielem)) ; break ;
         case ZETA_M_SYMM: delvm = domain->delv_zeta(ielem) ;         break ;
         case ZETA_M_FREE: delvm = Real_t(0.0) ;          break ;
         default:          /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvm = 0; /* ERROR - but quiets the compiler */
            break;
      }
      switch (bcMask & ZETA_P) {
         case ZETA_P_COMM: /* needs comm data */
         case 0:           delvp = domain->delv_zeta(domain->lzetap(ielem)) ; break ;
         case ZETA_P_SYMM: delvp = domain->delv_zeta(ielem) ;         break ;
         case ZETA_P_FREE: delvp = Real_t(0.0) ;          break ;
         default:          /* fprintf(stderr, "Error in switch at %s line %d\n",
                                   __FILE__, __LINE__); */
            delvp = 0; /* ERROR - but quiets the compiler */
            break;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

      if ( domain->vdov(ielem) > Real_t(0.) )  {
         qlin  = Real_t(0.) ;
         qquad = Real_t(0.) ;
      }
      else {
         Real_t delvxxi   = domain->delv_xi(ielem)   * domain->delx_xi(ielem)   ;
         Real_t delvxeta  = domain->delv_eta(ielem)  * domain->delx_eta(ielem)  ;
         Real_t delvxzeta = domain->delv_zeta(ielem) * domain->delx_zeta(ielem) ;

         if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
         if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
         if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

         Real_t rho = domain->elemMass(ielem) / (domain->volo(ielem) * domain->vnew(ielem)) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
      }

      domain->qq(ielem) = qquad ;
      domain->ql(ielem) = qlin  ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcMonotonicQForElems(Domain* domain)
{  
   //
   // initialize parameters
   // 
   const Real_t ptiny = Real_t(1.e-36) ;

   //
   // calculate the monotonic q for all regions
   //
   for (Index_t r=0 ; r<domain->numReg() ; ++r) {
      if (domain->regElemSize(r) > 0) {
         CalcMonotonicQRegionForElems(domain, r, ptiny) ;
      }
   }
}

/******************************************/

RAJA_STORAGE
void CalcQForElems(Domain* domain)
{
   //
   // MONOTONIC Q option
   //

   Index_t numElem = domain->numElem() ;

   if (numElem != 0) {
      Int_t allElem = numElem +  /* local elem */
            2*domain->sizeX()*domain->sizeY() + /* plane ghosts */
            2*domain->sizeX()*domain->sizeZ() + /* row ghosts */
            2*domain->sizeY()*domain->sizeZ() ; /* col ghosts */

      domain->AllocateGradients(elemMemPool, numElem, allElem);

#if USE_MPI
      CommRecv(*domain, MSG_MONOQ, 3,
               domain->sizeX(), domain->sizeY(), domain->sizeZ(),
               true, true) ;
#endif      

      /* Calculate velocity gradients */
      CalcMonotonicQGradientsForElems(domain);

#if USE_MPI      
      Domain_member fieldData[3] ;
      
      /* Transfer veloctiy gradients in the first order elements */
      /* problem->commElements->Transfer(CommElements::monoQ) ; */

      fieldData[0] = &Domain::delv_xi ;
      fieldData[1] = &Domain::delv_eta ;
      fieldData[2] = &Domain::delv_zeta ;

      CommSend(*domain, MSG_MONOQ, 3, fieldData,
               domain->sizeX(), domain->sizeY(), domain->sizeZ(),
               true, true) ;

      CommMonoQ(*domain) ;
#endif      

      CalcMonotonicQForElems(domain) ;

      // Free up memory
      domain->DeallocateGradients(elemMemPool);

      /* Don't allow excessive artificial viscosity */
      RAJA::ReduceMax<reduce_policy, Real_t>
             maxQ(domain->qstop() - Real_t(1.0)) ;
      RAJA::forall<elem_exec_policy>(domain->getElemISet(),
           [=] RAJA_DEVICE (int ielem) {
         maxQ.max(domain->q(ielem)) ;
      } ) ;

      if ( Real_t(maxQ) > domain->qstop() ) {
#if USE_MPI         
         MPI_Abort(MPI_COMM_WORLD, QStopError) ;
#else
         exit(QStopError);
#endif
      }
   }
}

/******************************************/

RAJA_STORAGE
void CalcPressureForElems(Real_t* p_new, Real_t* bvc,
                          Real_t* pbvc, Real_t* e_old,
                          Real_t* compression, Real_t *vnewc,
                          Real_t pmin,
                          Real_t p_cut, Real_t eosvmax,
                          LULESH_ISET& regISet)
{
   RAJA::forall<mat_exec_policy>(regISet,
        [=] RAJA_DEVICE (int ielem) {
      Real_t const  c1s = Real_t(2.0)/Real_t(3.0) ;
      bvc[ielem] = c1s * (compression[ielem] + Real_t(1.));
      pbvc[ielem] = c1s;
   } );

   RAJA::forall<mat_exec_policy>(regISet,
        [=] RAJA_DEVICE (int ielem) {
      p_new[ielem] = bvc[ielem] * e_old[ielem] ;

      if    (FABS(p_new[ielem]) <  p_cut   )
         p_new[ielem] = Real_t(0.0) ;

      if    ( vnewc[ielem] >= eosvmax ) /* impossible condition here? */
         p_new[ielem] = Real_t(0.0) ;

      if    (p_new[ielem]       <  pmin)
         p_new[ielem]   = pmin ;
   } );
}

/******************************************/

RAJA_STORAGE
void CalcEnergyForElems(Domain* domain,
                        Real_t* p_new, Real_t* e_new, Real_t* q_new,
                        Real_t* bvc, Real_t* pbvc,
                        Real_t* p_old,
                        Real_t* compression, Real_t* compHalfStep,
                        Real_t* vnewc, Real_t* work, Real_t *pHalfStep,
                        Real_t pmin, Real_t p_cut, Real_t  e_cut,
                        Real_t q_cut, Real_t emin,
                        Real_t rho0,
                        Real_t eosvmax,
                        LULESH_ISET& regISet)
{
   RAJA::forall<mat_exec_policy>(regISet,
        [=] RAJA_DEVICE (int ielem) {  
      e_new[ielem] = domain->e(ielem)
         - Real_t(0.5) * domain->delv(ielem) * (p_old[ielem] + domain->q(ielem))
         + Real_t(0.5) * work[ielem];

      if (e_new[ielem]  < emin ) {
         e_new[ielem] = emin ;
      }
   } );

   CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                        pmin, p_cut, eosvmax, 
                        regISet);

   RAJA::forall<mat_exec_policy>(regISet,
        [=] RAJA_DEVICE (int ielem) {  
      Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[ielem]) ;

      if ( domain->delv(ielem) > Real_t(0.) ) {
         q_new[ielem] /* = domain->qq(ielem) = domain->ql(ielem) */ = Real_t(0.);
      }
      else {
         Real_t ssc = ( pbvc[ielem] * e_new[ielem]
                 + vhalf * vhalf * bvc[ielem] * pHalfStep[ielem] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[ielem] = (ssc*domain->ql(ielem) + domain->qq(ielem)) ;
      }

      e_new[ielem] = e_new[ielem] + Real_t(0.5) * domain->delv(ielem)
         * (  Real_t(3.0)*(p_old[ielem]     + domain->q(ielem))
              - Real_t(4.0)*(pHalfStep[ielem] + q_new[ielem])) ;
   } );

   RAJA::forall<mat_exec_policy>(regISet,
        [=] RAJA_DEVICE (int ielem) {  
      e_new[ielem] += Real_t(0.5) * work[ielem];

      if (FABS(e_new[ielem]) < e_cut) {
         e_new[ielem] = Real_t(0.)  ;
      }
      if (     e_new[ielem]  < emin ) {
         e_new[ielem] = emin ;
      }
   } );

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                        pmin, p_cut, eosvmax, 
                        regISet);

   RAJA::forall<mat_exec_policy>(regISet,
        [=] RAJA_DEVICE (int ielem) {  
      const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
      Real_t q_tilde ;

      if (domain->delv(ielem) > Real_t(0.)) {
         q_tilde = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[ielem] * e_new[ielem]
                 + vnewc[ielem] * vnewc[ielem] * bvc[ielem] * p_new[ielem] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_tilde = (ssc*domain->ql(ielem) + domain->qq(ielem)) ;
      }

      e_new[ielem] -= (  Real_t(7.0)*(p_old[ielem]     + domain->q(ielem))
                       - Real_t(8.0)*(pHalfStep[ielem] + q_new[ielem])
                       + (p_new[ielem] + q_tilde)) * domain->delv(ielem)*sixth ;

      if (FABS(e_new[ielem]) < e_cut) {
         e_new[ielem] = Real_t(0.)  ;
      }
      if (     e_new[ielem]  < emin ) {
         e_new[ielem] = emin ;
      }
   } );

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                        pmin, p_cut, eosvmax, 
                        regISet);

   RAJA::forall<mat_exec_policy>(regISet,
        [=] RAJA_DEVICE (int ielem) {
      if ( domain->delv(ielem) <= Real_t(0.) ) {
         Real_t ssc = ( pbvc[ielem] * e_new[ielem]
            + vnewc[ielem] * vnewc[ielem] * bvc[ielem] * p_new[ielem] ) / rho0;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[ielem] = (ssc*domain->ql(ielem) + domain->qq(ielem)) ;

         if (FABS(q_new[ielem]) < q_cut) q_new[ielem] = Real_t(0.) ;
      }
   } );

   return ;
}

/******************************************/

RAJA_STORAGE
void CalcSoundSpeedForElems(Domain* domain,
                            Real_t *vnewc, Real_t rho0, Real_t *enewc,
                            Real_t *pnewc, Real_t *pbvc,
                            Real_t *bvc, Real_t ss4o3,
                            LULESH_ISET& regISet)
{
   RAJA::forall<mat_exec_policy>(regISet,
        [=] RAJA_DEVICE (int ielem) {
      Real_t ssTmp = (pbvc[ielem] * enewc[ielem] + vnewc[ielem] * vnewc[ielem] *
                 bvc[ielem] * pnewc[ielem]) / rho0;
      if (ssTmp <= Real_t(.1111111e-36)) {
         ssTmp = Real_t(.3333333e-18);
      }
      else {
         ssTmp = SQRT(ssTmp);
      }
      domain->ss(ielem) = ssTmp ;
   } );
}

/******************************************/

RAJA_STORAGE
void EvalEOSForElems(Domain* domain,
                     Real_t *vnewc, Real_t *p_old,
                     Real_t *compression, Real_t *compHalfStep,
                     Real_t *work, Real_t *p_new, Real_t *e_new,
                     Real_t *q_new, Real_t *bvc, Real_t *pbvc,
                     Real_t *pHalfStep, Int_t reg_num, Int_t rep)
{
   Real_t  e_cut = domain->e_cut() ;
   Real_t  p_cut = domain->p_cut() ;
   Real_t  ss4o3 = domain->ss4o3() ;
   Real_t  q_cut = domain->q_cut() ;

   Real_t eosvmax = domain->eosvmax() ;
   Real_t eosvmin = domain->eosvmin() ;
   Real_t pmin    = domain->pmin() ;
   Real_t emin    = domain->emin() ;
   Real_t rho0    = domain->refdens() ;

   LULESH_ISET& regISet = domain->getRegionISet(reg_num);
   Int_t numElemReg = regISet.getLength();
 
   //loop to add load imbalance based on region number 
   for(Int_t j = 0; j < rep; j++) {
      /* compress data, minimal set */
      RAJA::forall<mat_exec_policy>(regISet,
           [=] RAJA_DEVICE (Index_t ielem) {
         p_old[ielem] = domain->p(ielem) ;
         work[ielem] = Real_t(0.0) ;
      } );

      RAJA::forall<mat_exec_policy>(regISet,
           [=] RAJA_DEVICE (Index_t ielem) {
         Real_t vchalf ;
         compression[ielem] = Real_t(1.) / vnewc[ielem] - Real_t(1.);
         vchalf = vnewc[ielem] - domain->delv(ielem) * Real_t(.5);
         compHalfStep[ielem] = Real_t(1.) / vchalf - Real_t(1.);
      } );

      /* Check for v > eosvmax or v < eosvmin */
      if ( eosvmin != Real_t(0.) ) {
         RAJA::forall<mat_exec_policy>(regISet,
              [=] RAJA_DEVICE (Index_t ielem) {
            if (vnewc[ielem] <= eosvmin) { /* impossible due to calling func? */
               compHalfStep[ielem] = compression[ielem] ;
            }
         } );
      }

      if ( eosvmax != Real_t(0.) ) {
         RAJA::forall<mat_exec_policy>(regISet,
              [=] RAJA_DEVICE (Index_t ielem) {
            if (vnewc[ielem] >= eosvmax) { /* impossible due to calling func? */
               p_old[ielem]        = Real_t(0.) ;
               compression[ielem]  = Real_t(0.) ;
               compHalfStep[ielem] = Real_t(0.) ;
            }
         } );
      }

      CalcEnergyForElems(domain, p_new, e_new, q_new, bvc, pbvc,
                         p_old, compression, compHalfStep,
                         vnewc, work, pHalfStep, pmin,
                         p_cut, e_cut, q_cut, emin,
                         rho0, eosvmax,
                         regISet);
   }

   RAJA::forall<mat_exec_policy>(regISet,
        [=] RAJA_DEVICE (Index_t ielem) {
      domain->p(ielem) = p_new[ielem] ;
      domain->e(ielem) = e_new[ielem] ;
      domain->q(ielem) = q_new[ielem] ;
   } );

   CalcSoundSpeedForElems(domain,
                          vnewc, rho0, e_new, p_new,
                          pbvc, bvc, ss4o3,
                          regISet) ;

}

/******************************************/

RAJA_STORAGE
void ApplyMaterialPropertiesForElems(Domain* domain)
{
   Index_t numElem = domain->numElem() ;

  if (numElem != 0) {
    /* Expose all of the variables needed for material evaluation */
    Real_t eosvmin = domain->eosvmin() ;
    Real_t eosvmax = domain->eosvmax() ;
    Real_t *vnewc = elemMemPool.allocate(numElem) ;
    Real_t *p_old = elemMemPool.allocate(numElem) ;
    Real_t *compression = elemMemPool.allocate(numElem) ;
    Real_t *compHalfStep = elemMemPool.allocate(numElem) ;
    Real_t *work = elemMemPool.allocate(numElem) ;
    Real_t *p_new = elemMemPool.allocate(numElem) ;
    Real_t *e_new = elemMemPool.allocate(numElem) ;
    Real_t *q_new = elemMemPool.allocate(numElem) ;
    Real_t *bvc = elemMemPool.allocate(numElem) ;
    Real_t *pbvc = elemMemPool.allocate(numElem) ;
    Real_t *pHalfStep = elemMemPool.allocate(numElem) ;


    RAJA::forall<elem_exec_policy>(domain->getElemISet(),
         [=] RAJA_DEVICE (int i) {
       vnewc[i] = domain->vnew(i) ;
    } );

    // Bound the updated relative volumes with eosvmin/max
    if (eosvmin != Real_t(0.)) {
       RAJA::forall<elem_exec_policy>(domain->getElemISet(), 
            [=] RAJA_DEVICE (int i) {
          if (vnewc[i] < eosvmin)
             vnewc[i] = eosvmin ;
       } );
    }

    if (eosvmax != Real_t(0.)) {
       RAJA::forall<elem_exec_policy>(domain->getElemISet(),
            [=] RAJA_DEVICE (int i) {
          if (vnewc[i] > eosvmax)
             vnewc[i] = eosvmax ;
       } );
    }

    // check for negative element volume
    RAJA::ReduceMin<reduce_policy, Real_t> minvol(Real_t(1.0e+20));

    // This check may not make perfect sense in LULESH, but
    // it's representative of something in the full code -
    // just leave it in, please
    RAJA::forall<elem_exec_policy>(domain->getElemISet(),
         [=] RAJA_DEVICE (int i) {
       Real_t vc = domain->v(i) ;
       if (eosvmin != Real_t(0.)) {
          if (vc < eosvmin)
             vc = -1.0 ;
       }
       if (eosvmax != Real_t(0.)) {
          if (vc > eosvmax)
             vc = -1.0 ;
       }

       minvol.min(vc);
    } );

    if (Real_t(minvol) <= 0.) {
#if USE_MPI             
       MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
       exit(VolumeError);
#endif
    }

    for (Int_t reg_num=0 ; reg_num < domain->numReg() ; reg_num++) {
       Int_t rep;
       //Determine load imbalance for this region
       //round down the number with lowest cost
       if(reg_num < domain->numReg()/2)
	 rep = 1;
       //you don't get an expensive region unless you at least have 5 regions
       else if(reg_num < (domain->numReg() - (domain->numReg()+15)/20))
         rep = 1 + domain->cost();
       //very expensive regions
       else
	 rep = 10 * (1+ domain->cost());
       EvalEOSForElems(domain, vnewc, p_old, compression, compHalfStep,
                       work, p_new, e_new, q_new, bvc, pbvc, pHalfStep,
                       reg_num, rep);
    }

    elemMemPool.release(&pHalfStep) ;
    elemMemPool.release(&pbvc) ;
    elemMemPool.release(&bvc) ;
    elemMemPool.release(&q_new) ;
    elemMemPool.release(&e_new) ;
    elemMemPool.release(&p_new) ;
    elemMemPool.release(&work) ;
    elemMemPool.release(&compHalfStep) ;
    elemMemPool.release(&compression) ;
    elemMemPool.release(&p_old) ;
    elemMemPool.release(&vnewc) ;
  }
}

/******************************************/

RAJA_STORAGE
void UpdateVolumesForElems(Domain* domain, 
                           Real_t v_cut)
{
   RAJA::forall<elem_exec_policy>(domain->getElemISet(),
        [=] RAJA_DEVICE (int i) { 
      Real_t tmpV = domain->vnew(i) ;

      if ( FABS(tmpV - Real_t(1.0)) < v_cut )
         tmpV = Real_t(1.0) ;

      domain->v(i) = tmpV ;
   } );
}

/******************************************/

RAJA_STORAGE
void LagrangeElements(Domain* domain, Index_t numElem)
{
  CalcLagrangeElements(domain) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(domain) ;

  ApplyMaterialPropertiesForElems(domain) ;

  UpdateVolumesForElems(domain,
                        domain->v_cut()) ;
}

/******************************************/

RAJA_STORAGE
void CalcCourantConstraintForElems(Domain* domain, int reg_num,
                                   Real_t qqc, Real_t& dtcourant)
{
   Real_t  qqc2 = Real_t(64.0) * qqc * qqc ;

   RAJA::ReduceMin<reduce_policy, Real_t> dtcourantLoc(dtcourant) ;

   RAJA::forall<mat_exec_policy>(domain->getRegionISet(reg_num),
        [=] RAJA_DEVICE (int indx) {

      Real_t dtf = domain->ss(indx) * domain->ss(indx) ;

      if ( domain->vdov(indx) < Real_t(0.) ) {
         dtf += qqc2 * domain->arealg(indx) * domain->arealg(indx) *
                domain->vdov(indx) * domain->vdov(indx) ;
      }

      Real_t dtf_cmp = (domain->vdov(indx) != Real_t(0.))
                     ?  domain->arealg(indx) / SQRT(dtf) : Real_t(1.0e+20) ;

      /* determine minimum timestep with its corresponding elem */
      dtcourantLoc.min(dtf_cmp) ;
   } ) ;

   /* Don't try to register a time constraint if none of the elements
    * were active */
   if (dtcourantLoc < Real_t(1.0e+20)) {
      dtcourant = dtcourantLoc ;
   }

   return ;
}

/******************************************/

RAJA_STORAGE
void CalcHydroConstraintForElems(Domain* domain, int reg_num,
                                 Real_t dvovmax, Real_t& dthydro)
{
   RAJA::ReduceMin<reduce_policy, Real_t> dthydroLoc(dthydro) ;

   RAJA::forall<mat_exec_policy>(domain->getRegionISet(reg_num),
         [=] RAJA_DEVICE (int indx) {

       Real_t dtvov_cmp = (domain->vdov(indx) != Real_t(0.))
                        ? (dvovmax / (FABS(domain->vdov(indx))+Real_t(1.e-20)))
                        : Real_t(1.0e+20) ;

      dthydroLoc.min(dtvov_cmp) ;

   } ) ;

   if (dthydroLoc < Real_t(1.0e+20)) {
      dthydro = dthydroLoc ;
   }

   return ;
}

/******************************************/

RAJA_STORAGE
void CalcTimeConstraintsForElems(Domain* domain) {

   // Initialize conditions to a very large value
   domain->dtcourant() = 1.0e+20;
   domain->dthydro() = 1.0e+20;

   for (Index_t reg_num=0 ; reg_num < domain->numReg() ; ++reg_num) {
      /* evaluate time constraint */
      CalcCourantConstraintForElems(domain, reg_num,
                                    domain->qqc(),
                                    domain->dtcourant()) ;

      /* check hydro constraint */
      CalcHydroConstraintForElems(domain, reg_num,
                                  domain->dvovmax(),
                                  domain->dthydro()) ;
   }
}

/******************************************/

RAJA_STORAGE
void LagrangeLeapFrog(Domain* domain)
{
#if defined(SEDOV_SYNC_POS_VEL_LATE)
   Domain_member fieldData[6] ;
#endif

   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal(domain);


#if defined(SEDOV_SYNC_POS_VEL_LATE)
#endif

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(domain, domain->numElem());

#if USE_MPI   
#if defined(SEDOV_SYNC_POS_VEL_LATE)
   CommRecv(*domain, MSG_SYNC_POS_VEL, 6,
            domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
            false, false) ; 

   fieldData[0] = &Domain::x ;
   fieldData[1] = &Domain::y ;
   fieldData[2] = &Domain::z ;
   fieldData[3] = &Domain::xd ;
   fieldData[4] = &Domain::yd ;
   fieldData[5] = &Domain::zd ;
   
   CommSend(*domain, MSG_SYNC_POS_VEL, 6, fieldData,
            domain->sizeX() + 1, domain->sizeY() + 1, domain->sizeZ() + 1,
            false, false) ;
#endif
#endif   

   CalcTimeConstraintsForElems(domain);

#if USE_MPI   
#if defined(SEDOV_SYNC_POS_VEL_LATE)
   CommSyncPosVel(*domain) ;
#endif
#endif   
}


/******************************************/

int main(int argc, char *argv[])
{
   Domain *locDom ;
   Int_t numRanks ;
   Int_t myRank ;
   struct cmdLineOpts opts;

#if USE_MPI   
   Domain_member fieldData ;

   MPI_Init(&argc, &argv) ;
   MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
#else
   numRanks = 1;
   myRank = 0;
#endif   

   /* Set defaults that can be overridden by command line opts */
   opts.its = 9999999;
   opts.nx  = 30;
   opts.numReg = 11;
   opts.numFiles = (int)(numRanks+10)/9;
   opts.showProg = 0;
   opts.quiet = 0;
   opts.viz = 0;
   opts.balance = 1;
   opts.cost = 1;

   ParseCommandLineOptions(argc, argv, myRank, &opts);

   if ((myRank == 0) && (opts.quiet == 0)) {
      printf("Running problem size %d^3 per domain until completion\n", opts.nx);
      printf("Num processors: %d\n", numRanks);
#if defined(_OPENMP)
      printf("Num threads: %d\n", omp_get_max_threads());
#endif
      printf("Total number of elements: %lld\n\n", (long long int)(numRanks*opts.nx*opts.nx*opts.nx));
      printf("To run other sizes, use -s <integer>.\n");
      printf("To run a fixed number of iterations, use -i <integer>.\n");
      printf("To run a more or less balanced region set, use -b <integer>.\n");
      printf("To change the relative costs of regions, use -c <integer>.\n");
      printf("To print out progress, use -p\n");
      printf("To write an output file for VisIt, use -v\n");
      printf("See help (-h) for more options\n\n");
   }

   // Set up the mesh and decompose. Assumes regular cubes for now
   Int_t col, row, plane, side;
   InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

   // Build the main data structure and initialize it
   locDom = new Domain(numRanks, col, row, plane, opts.nx,
                       side, opts.numReg, opts.balance, opts.cost) ;


#if USE_MPI   
   fieldData = &Domain::nodalMass ;

   // Initial domain boundary communication 
   CommRecv(*locDom, MSG_COMM_SBN, 1,
            locDom->sizeX() + 1, locDom->sizeY() + 1, locDom->sizeZ() + 1,
            true, false) ;
   CommSend(*locDom, MSG_COMM_SBN, 1, &fieldData,
            locDom->sizeX() + 1, locDom->sizeY() + 1, locDom->sizeZ() +  1,
            true, false) ;
   CommSBN(*locDom, 1, &fieldData) ;

   // End initialization
   MPI_Barrier(MPI_COMM_WORLD);
#endif   
   // BEGIN timestep to solution */
#ifdef RAJA_USE_CALIPER
   RAJA::Timer timer_main; 
   timer_main.start("timer_main");
#else
#if USE_MPI   
   double start = MPI_Wtime();
#else
   timeval start;
   gettimeofday(&start, NULL) ;
#endif
#endif
//debug to see region sizes
// for(Int_t i = 0; i < locDom->numReg(); i++) {
//    std::cout << "region " << i + 1<< " size = " << locDom->regElemSize(i) << std::endl;
//    RAJA::forall<mat_exec_policy>(locDom->getRegionISet(i), [=] (int idx) { printf("%d ", idx) ; }) ;
//    printf("\n\n") ;
// }
   while((locDom->time() < locDom->stoptime()) && (locDom->cycle() < opts.its)) {

      TimeIncrement(*locDom) ;
      LagrangeLeapFrog(locDom) ;

      if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
         printf("cycle = %d, time = %e, dt=%e\n",
                locDom->cycle(), double(locDom->time()), double(locDom->deltatime()) ) ;
      }
   }
double elapsed_time;
#ifdef RAJA_USE_CALIPER
   // Use reduced max elapsed time
   timer_main.stop("timer_main");
   elapsed_time = timer_main.elapsed();
#else
#if USE_MPI   
   elapsed_time = MPI_Wtime() - start;
#else
   timeval end;
   gettimeofday(&end, NULL) ;
   elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000 ;
#endif
#endif
   double elapsed_timeG;
#if USE_MPI   
   MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE,
              MPI_MAX, 0, MPI_COMM_WORLD);
#else
   elapsed_timeG = elapsed_time;
#endif

   // Write out final viz file */
   if (opts.viz) {
      DumpToVisit(*locDom, opts.numFiles, myRank, numRanks) ;
   }
   
   if ((myRank == 0) && (opts.quiet == 0)) {
      VerifyAndWriteFinalOutput(elapsed_timeG, *locDom, opts.nx, numRanks);
   }

   delete locDom;

#if USE_MPI
   MPI_Finalize() ;
#endif

   return 0 ;
}
