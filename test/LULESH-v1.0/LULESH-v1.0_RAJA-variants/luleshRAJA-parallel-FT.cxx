/*

                 Copyright (c) 2010.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 1.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

#include "RAJA/RAJA.hxx"

#include "Timer.hxx"

/*
 ***********************************************
 * Set parameters that define how code will run.
 ***********************************************
 */

//
// Display simulation time and timestep during run.
//
bool show_run_progress = false;

//
// Set stop time and time increment for run.
//
// The absolute value of lulesh_time_step sets the first time step increment.
//   - If < 0, the CFL condition will be used to determine subsequent time
//     step sizes (with some upper bound on the amount the timestep can grow).
//   - If > 0, the time step will be fixed for the entire run.
//
const double lulesh_stop_time = 1.0e-2;
const double lulesh_time_step = -1.0e-7;

//
// Set mesh size (physical domain size is fixed).
//
// Mesh will be lulesh_edge_elems^3.
//
const int lulesh_edge_elems = 45;


//
//   Tiling mode.
//
enum TilingMode
{
   Canonical,       // canonical element ordering -- single range segment
   Tiled_Index,     // canonical ordering, tiled using unstructured segments
   Tiled_Order,     // elements permuted, tiled using range segments
   Tiled_LockFree,  // tiled ordering, lock-free
};
TilingMode lulesh_tiling_mode = Canonical;
//TilingMode lulesh_tiling_mode = Tiled_Index;
//TilingMode lulesh_tiling_mode = Tiled_Order;
//TilingMode lulesh_tiling_mode = Tiled_LockFree;

//
// Set number of tiles in each mesh direction for non-canonical oerderings.
//
const int lulesh_xtile = 2;
const int lulesh_ytile = 2;
const int lulesh_ztile = 2;

//
//   RAJA IndexSet type used in loop traversals.
//
//   Need to verify if this can be set to RangeSegment or ListSegment
//   types. It may be useful to compare IndexSet performance to
//   basic segment types; e.g.,
//
//     - Canonical ordering should be able to use IndexSet or
//                                                RangeSegment.
//     - Tiled_Index ordering should be able to use IndexSet or
//                                                  ListSegment.
//
//   Policies for index set segment iteration and segment execution.
//
//   NOTE: Currently, we apply single policy across all loop patterns.
//
typedef RAJA::seq_segit              IndexSet_Seg_Iter;
//typedef RAJA::omp_parallel_for_segit IndexSet_Seg_Iter;
//typedef RAJA::cilk_for_segit         IndexSet_Seg_Iter;

//typedef RAJA::seq_exec              Segment_Exec;
//typedef RAJA::simd_exec             Segment_Exec;
typedef RAJA::omp_parallel_for_exec Segment_Exec;
//typedef RAJA::cilk_for_exec         Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<IndexSet_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<IndexSet_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<IndexSet_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<IndexSet_Seg_Iter, Segment_Exec> minloc_exec_policy;
typedef                                            Segment_Exec  range_exec_policy;

typedef                                            RAJA::omp_reduce  reduce_policy;

//
// use RAJA data types for loop operations using RAJA
//
typedef RAJA::Index_type  Index_t ; /* array subscript and loop index */
typedef RAJA::Real_type   Real_t ;  /* floating point representation */
typedef RAJA::Real_ptr    Real_p;
typedef RAJA::const_Real_ptr    const_Real_p;
typedef RAJA::Index_type* Index_p;

/****************************************************/
/*                                                  */
/* Allow flexibility for arithmetic representations */
/*                                                  */
/* Think about how to make this consistent w/RAJA   */
/* type parameterization (above)!!                  */
/*                                                  */
/****************************************************/

#define MAX(a, b) ( ((a) > (b)) ? (a) : (b))

/* Could also support fixed point and interval arithmetic types */
typedef float        real4 ;
typedef double       real8 ;
typedef long double  real10 ;  /* 10 bytes on x86 */

typedef int    Int_t ;   /* integer representation */

inline real4  SQRT(real4  arg) { return sqrtf(arg) ; }
inline real8  SQRT(real8  arg) { return sqrt(arg) ; }
inline real10 SQRT(real10 arg) { return sqrtl(arg) ; }

inline real4  CBRT(real4  arg) { return cbrtf(arg) ; }
inline real8  CBRT(real8  arg) { return cbrt(arg) ; }
inline real10 CBRT(real10 arg) { return cbrtl(arg) ; }

inline real4  FABS(real4  arg) { return fabsf(arg) ; }
inline real8  FABS(real8  arg) { return fabs(arg) ; }
inline real10 FABS(real10 arg) { return fabsl(arg) ; }


#define RAJA_STORAGE static inline

enum { VolumeError = -1, QStopError = -2 } ;


#ifdef RAJA_ENABLE_FT
#include <unistd.h>
#include <signal.h>

/* fault_type:   == 0 no fault, < 0 unrecoverable, > 0 recoverable */
namespace RAJA {
volatile int fault_type = 0 ;
}

static struct sigaction sigalrmact ;

static void simulate_fault(int sig)
{
   /* 10% chance of unrecoverable fault */
   RAJA::fault_type = (rand() % 100) - 10 ;
}
#endif

/*********************************/
/* Data structure implementation */
/*********************************/

/* might want to add access methods so that memory can be */
/* better managed, as in luleshFT */

struct Domain {
   /* Elem-centered */

   RAJA::IndexSet *domElemList ;   /* elem indexset */
   RAJA::IndexSet *matElemList ;   /* material indexset */
   Index_p nodelist ;     /* elemToNode connectivity */

   Index_p lxim ;         /* elem connectivity through face */
   Index_p lxip ;
   Index_p letam ;
   Index_p letap ;
   Index_p lzetam ;
   Index_p lzetap ;

   Int_t *elemBC ;         /* elem face symm/free-surface flag */

   Real_p e ;             /* energy */

   Real_p p ;             /* pressure */

   Real_p q ;             /* q */
   Real_p ql ;            /* linear term for q */
   Real_p qq ;            /* quadratic term for q */

   Real_p v ;             /* relative volume */

   Real_p volo ;          /* reference volume */
   Real_p delv ;          /* m_vnew - m_v */
   Real_p vdov ;          /* volume derivative over volume */

   Real_p arealg ;        /* elem characteristic length */

   Real_p ss ;            /* "sound speed" */

   Real_p elemMass ;      /* mass */

   /* Elem temporaries */

   Real_p vnew ;          /* new relative volume -- temporary */

   Real_p delv_xi ;       /* velocity gradient -- temporary */
   Real_p delv_eta ;
   Real_p delv_zeta ;

   Real_p delx_xi ;       /* position gradient -- temporary */
   Real_p delx_eta ;
   Real_p delx_zeta ;

   Real_p dxx ;          /* principal strains -- temporary */
   Real_p dyy ;
   Real_p dzz ;

   /* Node-centered */

   RAJA::IndexSet *domNodeList ;   /* node indexset */

   Real_p x ;             /* coordinates */
   Real_p y ;
   Real_p z ;

   Real_p xd ;            /* velocities */
   Real_p yd ;
   Real_p zd ;

   Real_p xdd ;           /* accelerations */
   Real_p ydd ;
   Real_p zdd ;

   Real_p fx ;            /* forces */
   Real_p fy ;
   Real_p fz ;

   Real_p nodalMass ;     /* mass */

   // OMP hack 
   Index_p nodeElemStart ;
   Index_p nodeElemCornerList ;

   /* Boundary nodesets */

   Index_p symmX ;        /* Nodes on X symmetry plane */
   Index_p symmY ;        /* Nodes on Y symmetry plane */
   Index_p symmZ ;        /* Nodes on Z symmetry plane */

   /* Parameters */

   Real_t  dtfixed ;           /* fixed time increment */
   Real_t  time ;              /* current time */
   Real_t  deltatime ;         /* variable time increment */
   Real_t  deltatimemultlb ;
   Real_t  deltatimemultub ;
   Real_t  stoptime ;          /* end time for simulation */

   Real_t  u_cut ;             /* velocity tolerance */
   Real_t  hgcoef ;            /* hourglass control */
   Real_t  qstop ;             /* excessive q indicator */
   Real_t  monoq_max_slope ;
   Real_t  monoq_limiter_mult ;
   Real_t  e_cut ;             /* energy tolerance */
   Real_t  p_cut ;             /* pressure tolerance */
   Real_t  ss4o3 ;
   Real_t  q_cut ;             /* q tolerance */
   Real_t  v_cut ;             /* relative volume tolerance */
   Real_t  qlc_monoq ;         /* linear term coef for q */
   Real_t  qqc_monoq ;         /* quadratic term coef for q */
   Real_t  qqc ;
   Real_t  eosvmax ;
   Real_t  eosvmin ;
   Real_t  pmin ;              /* pressure floor */
   Real_t  emin ;              /* energy floor */
   Real_t  dvovmax ;           /* maximum allowable volume change */
   Real_t  refdens ;           /* reference density */

   Real_t  dtcourant ;         /* courant constraint */
   Real_t  dthydro ;           /* volume change constraint */
   Real_t  dtmax ;             /* maximum allowable time increment */

   Int_t   cycle ;             /* iteration count for simulation */

   Index_t sizeX ;
   Index_t sizeY ;
   Index_t sizeZ ;
   Index_t numElem ;

   Index_t numNode ;
} ;

// ########################################################
//  Memory allocate/release routines
// ########################################################
#include "luleshMemory.hxx"


/* Stuff needed for boundary conditions */
/* 2 BCs on each of 6 hexahedral faces (12 bits) */
#define XI_M        0x003
#define XI_M_SYMM   0x001
#define XI_M_FREE   0x002

#define XI_P        0x00c
#define XI_P_SYMM   0x004
#define XI_P_FREE   0x008

#define ETA_M       0x030
#define ETA_M_SYMM  0x010
#define ETA_M_FREE  0x020

#define ETA_P       0x0c0
#define ETA_P_SYMM  0x040
#define ETA_P_FREE  0x080

#define ZETA_M      0x300
#define ZETA_M_SYMM 0x100
#define ZETA_M_FREE 0x200

#define ZETA_P      0xc00
#define ZETA_P_SYMM 0x400
#define ZETA_P_FREE 0x800


RAJA_STORAGE
void TimeIncrement(Domain *domain)
{
   Real_t targetdt = domain->stoptime - domain->time ;

   if ((domain->dtfixed <= Real_t(0.0)) && (domain->cycle != Int_t(0))) {
      Real_t ratio ;
      Real_t olddt = domain->deltatime ;

      /* This will require a reduction in parallel */
      Real_t newdt = Real_t(1.0e+20) ;
      if (domain->dtcourant < newdt) {
         newdt = domain->dtcourant / Real_t(2.0) ;
      }
      if (domain->dthydro < newdt) {
         newdt = domain->dthydro * Real_t(2.0) / Real_t(3.0) ;
      }

      ratio = newdt / olddt ;
      if (ratio >= Real_t(1.0)) {
         if (ratio < domain->deltatimemultlb) {
            newdt = olddt ;
         }
         else if (ratio > domain->deltatimemultub) {
            newdt = olddt*domain->deltatimemultub ;
         }
      }

      if (newdt > domain->dtmax) {
         newdt = domain->dtmax ;
      }
      domain->deltatime = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > domain->deltatime) &&
       (targetdt < (Real_t(4.0) * domain->deltatime / Real_t(3.0))) ) {
      targetdt = Real_t(2.0) * domain->deltatime / Real_t(3.0) ;
   }

   if (targetdt < domain->deltatime) {
      domain->deltatime = targetdt ;
   }

   domain->time += domain->deltatime ;

   ++domain->cycle ;
}

RAJA_STORAGE
void InitStressTermsForElems(Real_p p, Real_p q,
                             Real_p sigxx, Real_p sigyy, Real_p sigzz,
                             RAJA::IndexSet *domElemList)
{
   //
   // pull in the stresses appropriate to the hydro integration
   //

   RAJA::forall<elem_exec_policy>(*domElemList, [=] RAJA_DEVICE (int idx) {
      sigxx[idx] = sigyy[idx] = sigzz[idx] =  - p[idx] - q[idx] ;
     }
   ) ;
}

RAJA_STORAGE
void CalcElemShapeFunctionDerivatives( const_Real_p x,
                                       const_Real_p y,
                                       const_Real_p z,
                                       Real_t b[][8],
                                       Real_t* const volume
                                     )
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

RAJA_STORAGE
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

RAJA_STORAGE
void CalcElemNodeNormals(
                         Real_p pfx,
                         Real_p pfy,
                         Real_p pfz,
                         const_Real_p x,
                         const_Real_p y,
                         const_Real_p z
                        )
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

RAJA_STORAGE
void SumElemStressesToNodeForces( const Real_t B[][8],
                                  const Real_t stress_xx,
                                  const Real_t stress_yy,
                                  const Real_t stress_zz,
                                  Real_p fx, Real_p fy, Real_p fz
                                )
{
  Real_t pfx0 = B[0][0] ;   Real_t pfx1 = B[0][1] ;
  Real_t pfx2 = B[0][2] ;   Real_t pfx3 = B[0][3] ;
  Real_t pfx4 = B[0][4] ;   Real_t pfx5 = B[0][5] ;
  Real_t pfx6 = B[0][6] ;   Real_t pfx7 = B[0][7] ;

  Real_t pfy0 = B[1][0] ;   Real_t pfy1 = B[1][1] ;
  Real_t pfy2 = B[1][2] ;   Real_t pfy3 = B[1][3] ;
  Real_t pfy4 = B[1][4] ;   Real_t pfy5 = B[1][5] ;
  Real_t pfy6 = B[1][6] ;   Real_t pfy7 = B[1][7] ;

  Real_t pfz0 = B[2][0] ;   Real_t pfz1 = B[2][1] ;
  Real_t pfz2 = B[2][2] ;   Real_t pfz3 = B[2][3] ;
  Real_t pfz4 = B[2][4] ;   Real_t pfz5 = B[2][5] ;
  Real_t pfz6 = B[2][6] ;   Real_t pfz7 = B[2][7] ;

  fx[0] = -( stress_xx * pfx0 );
  fx[1] = -( stress_xx * pfx1 );
  fx[2] = -( stress_xx * pfx2 );
  fx[3] = -( stress_xx * pfx3 );
  fx[4] = -( stress_xx * pfx4 );
  fx[5] = -( stress_xx * pfx5 );
  fx[6] = -( stress_xx * pfx6 );
  fx[7] = -( stress_xx * pfx7 );

  fy[0] = -( stress_yy * pfy0  );
  fy[1] = -( stress_yy * pfy1  );
  fy[2] = -( stress_yy * pfy2  );
  fy[3] = -( stress_yy * pfy3  );
  fy[4] = -( stress_yy * pfy4  );
  fy[5] = -( stress_yy * pfy5  );
  fy[6] = -( stress_yy * pfy6  );
  fy[7] = -( stress_yy * pfy7  );

  fz[0] = -( stress_zz * pfz0 );
  fz[1] = -( stress_zz * pfz1 );
  fz[2] = -( stress_zz * pfz2 );
  fz[3] = -( stress_zz * pfz3 );
  fz[4] = -( stress_zz * pfz4 );
  fz[5] = -( stress_zz * pfz5 );
  fz[6] = -( stress_zz * pfz6 );
  fz[7] = -( stress_zz * pfz7 );
}

RAJA_STORAGE
void IntegrateStressForElems( Index_t numElem, Index_p nodelist,
                              Real_p x,  Real_p y,  Real_p z,
                              Real_p fx, Real_p fy, Real_p fz,
                              Real_p sigxx, Real_p sigyy, Real_p sigzz,
                              Real_p determ, Index_p nodeElemStart,
                              Index_p nodeElemCornerList,
                              RAJA::IndexSet *domElemList,
                              RAJA::IndexSet *domNodeList )
{
  Real_p fx_elem = Allocate<Real_t>(numElem*8) ;
  Real_p fy_elem = Allocate<Real_t>(numElem*8) ;
  Real_p fz_elem = Allocate<Real_t>(numElem*8) ;

  // loop over all elements
  RAJA::forall<elem_exec_policy>(*domElemList, [=] RAJA_DEVICE (int k) {
    Real_t B[3][8] ;// shape function derivatives
    Real_t x_local[8] ;
    Real_t y_local[8] ;
    Real_t z_local[8] ;

    const Index_p elemNodes = &nodelist[8*k];

    // get nodal coordinates from global arrays and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemNodes[lnode];
      x_local[lnode] = x[gnode];
      y_local[lnode] = y[gnode];
      z_local[lnode] = z[gnode];
    }

    /* Volume calculation involves extra work for numerical consistency. */
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                     B, &determ[k]);

    CalcElemNodeNormals( B[0] , B[1], B[2],
                         x_local, y_local, z_local );

    SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                                 &fx_elem[k*8], &fy_elem[k*8], &fz_elem[k*8]) ;
   }
  ) ;

  RAJA::forall<node_exec_policy>(*domNodeList, [=] RAJA_DEVICE (int gnode) {
     Index_t count = nodeElemStart[gnode+1] - nodeElemStart[gnode] ;
     Index_t *cornerList = &nodeElemCornerList[nodeElemStart[gnode]] ;
     Real_t fx_sum = Real_t(0.0) ;
     Real_t fy_sum = Real_t(0.0) ;
     Real_t fz_sum = Real_t(0.0) ;
     for (Index_t i=0 ; i < count ; ++i) {
        Index_t elem = cornerList[i] ;
        fx_sum += fx_elem[elem] ;
        fy_sum += fy_elem[elem] ;
        fz_sum += fz_elem[elem] ;
     }
     fx[gnode] = fx_sum ;
     fy[gnode] = fy_sum ;
     fz[gnode] = fz_sum ;
   }
  ) ;

  Release(&fz_elem) ;
  Release(&fy_elem) ;
  Release(&fx_elem) ;
}

RAJA_STORAGE
void CollectDomainNodesToElemNodes(Real_p x, Real_p y, Real_p z,
                                   Index_p elemToNode,
                                   Real_p elemX,
                                   Real_p elemY,
                                   Real_p elemZ
                                  )
{
   Index_t nd0i = elemToNode[0] ;
   Index_t nd1i = elemToNode[1] ;
   Index_t nd2i = elemToNode[2] ;
   Index_t nd3i = elemToNode[3] ;
   Index_t nd4i = elemToNode[4] ;
   Index_t nd5i = elemToNode[5] ;
   Index_t nd6i = elemToNode[6] ;
   Index_t nd7i = elemToNode[7] ;

   elemX[0] = x[nd0i];
   elemX[1] = x[nd1i];
   elemX[2] = x[nd2i];
   elemX[3] = x[nd3i];
   elemX[4] = x[nd4i];
   elemX[5] = x[nd5i];
   elemX[6] = x[nd6i];
   elemX[7] = x[nd7i];

   elemY[0] = y[nd0i];
   elemY[1] = y[nd1i];
   elemY[2] = y[nd2i];
   elemY[3] = y[nd3i];
   elemY[4] = y[nd4i];
   elemY[5] = y[nd5i];
   elemY[6] = y[nd6i];
   elemY[7] = y[nd7i];

   elemZ[0] = z[nd0i];
   elemZ[1] = z[nd1i];
   elemZ[2] = z[nd2i];
   elemZ[3] = z[nd3i];
   elemZ[4] = z[nd4i];
   elemZ[5] = z[nd5i];
   elemZ[6] = z[nd6i];
   elemZ[7] = z[nd7i];

}

RAJA_STORAGE
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

RAJA_STORAGE
void CalcElemVolumeDerivative(
                              Real_p dvdx,
                              Real_p dvdy,
                              Real_p dvdz,
                              const_Real_p x,
                              const_Real_p y,
                              const_Real_p z
                             )
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

RAJA_STORAGE
void CalcElemFBHourglassForce(
                              Real_p xd, Real_p yd, Real_p zd,
                              Real_p hourgam0, Real_p hourgam1,
                              Real_p hourgam2, Real_p hourgam3,
                              Real_p hourgam4, Real_p hourgam5,
                              Real_p hourgam6, Real_p hourgam7,
                              Real_t coefficient,
                              Real_p hgfx, Real_p hgfy, Real_p hgfz
                             )
{
   const Index_t i00=0;
   const Index_t i01=1;
   const Index_t i02=2;
   const Index_t i03=3;

   Real_t h00 =
      hourgam0[i00] * xd[0] + hourgam1[i00] * xd[1] +
      hourgam2[i00] * xd[2] + hourgam3[i00] * xd[3] +
      hourgam4[i00] * xd[4] + hourgam5[i00] * xd[5] +
      hourgam6[i00] * xd[6] + hourgam7[i00] * xd[7];

   Real_t h01 =
      hourgam0[i01] * xd[0] + hourgam1[i01] * xd[1] +
      hourgam2[i01] * xd[2] + hourgam3[i01] * xd[3] +
      hourgam4[i01] * xd[4] + hourgam5[i01] * xd[5] +
      hourgam6[i01] * xd[6] + hourgam7[i01] * xd[7];

   Real_t h02 =
      hourgam0[i02] * xd[0] + hourgam1[i02] * xd[1]+
      hourgam2[i02] * xd[2] + hourgam3[i02] * xd[3]+
      hourgam4[i02] * xd[4] + hourgam5[i02] * xd[5]+
      hourgam6[i02] * xd[6] + hourgam7[i02] * xd[7];

   Real_t h03 =
      hourgam0[i03] * xd[0] + hourgam1[i03] * xd[1] +
      hourgam2[i03] * xd[2] + hourgam3[i03] * xd[3] +
      hourgam4[i03] * xd[4] + hourgam5[i03] * xd[5] +
      hourgam6[i03] * xd[6] + hourgam7[i03] * xd[7];

   hgfx[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfx[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfx[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfx[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfx[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfx[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfx[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfx[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * yd[0] + hourgam1[i00] * yd[1] +
      hourgam2[i00] * yd[2] + hourgam3[i00] * yd[3] +
      hourgam4[i00] * yd[4] + hourgam5[i00] * yd[5] +
      hourgam6[i00] * yd[6] + hourgam7[i00] * yd[7];

   h01 =
      hourgam0[i01] * yd[0] + hourgam1[i01] * yd[1] +
      hourgam2[i01] * yd[2] + hourgam3[i01] * yd[3] +
      hourgam4[i01] * yd[4] + hourgam5[i01] * yd[5] +
      hourgam6[i01] * yd[6] + hourgam7[i01] * yd[7];

   h02 =
      hourgam0[i02] * yd[0] + hourgam1[i02] * yd[1]+
      hourgam2[i02] * yd[2] + hourgam3[i02] * yd[3]+
      hourgam4[i02] * yd[4] + hourgam5[i02] * yd[5]+
      hourgam6[i02] * yd[6] + hourgam7[i02] * yd[7];

   h03 =
      hourgam0[i03] * yd[0] + hourgam1[i03] * yd[1] +
      hourgam2[i03] * yd[2] + hourgam3[i03] * yd[3] +
      hourgam4[i03] * yd[4] + hourgam5[i03] * yd[5] +
      hourgam6[i03] * yd[6] + hourgam7[i03] * yd[7];


   hgfy[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfy[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfy[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfy[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfy[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfy[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfy[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfy[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * zd[0] + hourgam1[i00] * zd[1] +
      hourgam2[i00] * zd[2] + hourgam3[i00] * zd[3] +
      hourgam4[i00] * zd[4] + hourgam5[i00] * zd[5] +
      hourgam6[i00] * zd[6] + hourgam7[i00] * zd[7];

   h01 =
      hourgam0[i01] * zd[0] + hourgam1[i01] * zd[1] +
      hourgam2[i01] * zd[2] + hourgam3[i01] * zd[3] +
      hourgam4[i01] * zd[4] + hourgam5[i01] * zd[5] +
      hourgam6[i01] * zd[6] + hourgam7[i01] * zd[7];

   h02 =
      hourgam0[i02] * zd[0] + hourgam1[i02] * zd[1]+
      hourgam2[i02] * zd[2] + hourgam3[i02] * zd[3]+
      hourgam4[i02] * zd[4] + hourgam5[i02] * zd[5]+
      hourgam6[i02] * zd[6] + hourgam7[i02] * zd[7];

   h03 =
      hourgam0[i03] * zd[0] + hourgam1[i03] * zd[1] +
      hourgam2[i03] * zd[2] + hourgam3[i03] * zd[3] +
      hourgam4[i03] * zd[4] + hourgam5[i03] * zd[5] +
      hourgam6[i03] * zd[6] + hourgam7[i03] * zd[7];


   hgfz[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfz[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfz[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfz[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfz[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfz[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfz[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfz[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);
}

const Real_t ggamma[4][8] =
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


RAJA_STORAGE
void CalcFBHourglassForceForElems( Index_t numElem, Index_t numNode,
                                   Index_p nodelist,
                                   Real_p  ss, Real_p  elemMass,
                                   Real_p  xd, Real_p  yd, Real_p  zd,
                                   Real_p  fx, Real_p  fy, Real_p  fz,
                                   Real_p  determ,
                                   Real_p  x8n, Real_p  y8n, Real_p  z8n,
                                   Real_p  dvdx, Real_p  dvdy, Real_p  dvdz,
                                   Real_t hourg, Index_p nodeElemStart,
                                   Index_p nodeElemCornerList,
                                   RAJA::IndexSet *domElemList,
                                   RAJA::IndexSet *domNodeList)
{
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/

   Real_p fx_elem = Allocate<Real_t>(numElem*8) ;
   Real_p fy_elem = Allocate<Real_t>(numElem*8) ;
   Real_p fz_elem = Allocate<Real_t>(numElem*8) ;

/*************************************************/
/*    compute the hourglass modes */

   RAJA::forall<elem_exec_policy>(*domElemList, [=] RAJA_DEVICE (int i2) {
      Real_t coefficient;

      Real_t hourgam0[4], hourgam1[4], hourgam2[4], hourgam3[4] ;
      Real_t hourgam4[4], hourgam5[4], hourgam6[4], hourgam7[4];
      Real_t xd1[8], yd1[8], zd1[8] ;

      Index_p elemToNode = &nodelist[8*i2];
      Index_t i3=8*i2;
      Real_t volinv=Real_t(1.0)/determ[i2];
      Real_t ss1, mass1, volume13 ;
      for(Index_t i1=0;i1<4;++i1){

         Real_t hourmodx =
            x8n[i3] * ggamma[i1][0] + x8n[i3+1] * ggamma[i1][1] +
            x8n[i3+2] * ggamma[i1][2] + x8n[i3+3] * ggamma[i1][3] +
            x8n[i3+4] * ggamma[i1][4] + x8n[i3+5] * ggamma[i1][5] +
            x8n[i3+6] * ggamma[i1][6] + x8n[i3+7] * ggamma[i1][7];

         Real_t hourmody =
            y8n[i3] * ggamma[i1][0] + y8n[i3+1] * ggamma[i1][1] +
            y8n[i3+2] * ggamma[i1][2] + y8n[i3+3] * ggamma[i1][3] +
            y8n[i3+4] * ggamma[i1][4] + y8n[i3+5] * ggamma[i1][5] +
            y8n[i3+6] * ggamma[i1][6] + y8n[i3+7] * ggamma[i1][7];

         Real_t hourmodz =
            z8n[i3] * ggamma[i1][0] + z8n[i3+1] * ggamma[i1][1] +
            z8n[i3+2] * ggamma[i1][2] + z8n[i3+3] * ggamma[i1][3] +
            z8n[i3+4] * ggamma[i1][4] + z8n[i3+5] * ggamma[i1][5] +
            z8n[i3+6] * ggamma[i1][6] + z8n[i3+7] * ggamma[i1][7];

         hourgam0[i1] = ggamma[i1][0] -  volinv*(dvdx[i3  ] * hourmodx +
                                                  dvdy[i3  ] * hourmody +
                                                  dvdz[i3  ] * hourmodz );

         hourgam1[i1] = ggamma[i1][1] -  volinv*(dvdx[i3+1] * hourmodx +
                                                  dvdy[i3+1] * hourmody +
                                                  dvdz[i3+1] * hourmodz );

         hourgam2[i1] = ggamma[i1][2] -  volinv*(dvdx[i3+2] * hourmodx +
                                                  dvdy[i3+2] * hourmody +
                                                  dvdz[i3+2] * hourmodz );

         hourgam3[i1] = ggamma[i1][3] -  volinv*(dvdx[i3+3] * hourmodx +
                                                  dvdy[i3+3] * hourmody +
                                                  dvdz[i3+3] * hourmodz );

         hourgam4[i1] = ggamma[i1][4] -  volinv*(dvdx[i3+4] * hourmodx +
                                                  dvdy[i3+4] * hourmody +
                                                  dvdz[i3+4] * hourmodz );

         hourgam5[i1] = ggamma[i1][5] -  volinv*(dvdx[i3+5] * hourmodx +
                                                  dvdy[i3+5] * hourmody +
                                                  dvdz[i3+5] * hourmodz );

         hourgam6[i1] = ggamma[i1][6] -  volinv*(dvdx[i3+6] * hourmodx +
                                                  dvdy[i3+6] * hourmody +
                                                  dvdz[i3+6] * hourmodz );

         hourgam7[i1] = ggamma[i1][7] -  volinv*(dvdx[i3+7] * hourmodx +
                                                  dvdy[i3+7] * hourmody +
                                                  dvdz[i3+7] * hourmodz );

      }

      /* compute forces */
      /* store forces into h arrays (force arrays) */

      ss1=ss[i2];
      mass1=elemMass[i2];
      volume13=CBRT(determ[i2]);

      Index_t n0si2 = elemToNode[0];
      Index_t n1si2 = elemToNode[1];
      Index_t n2si2 = elemToNode[2];
      Index_t n3si2 = elemToNode[3];
      Index_t n4si2 = elemToNode[4];
      Index_t n5si2 = elemToNode[5];
      Index_t n6si2 = elemToNode[6];
      Index_t n7si2 = elemToNode[7];

      xd1[0] = xd[n0si2];
      xd1[1] = xd[n1si2];
      xd1[2] = xd[n2si2];
      xd1[3] = xd[n3si2];
      xd1[4] = xd[n4si2];
      xd1[5] = xd[n5si2];
      xd1[6] = xd[n6si2];
      xd1[7] = xd[n7si2];

      yd1[0] = yd[n0si2];
      yd1[1] = yd[n1si2];
      yd1[2] = yd[n2si2];
      yd1[3] = yd[n3si2];
      yd1[4] = yd[n4si2];
      yd1[5] = yd[n5si2];
      yd1[6] = yd[n6si2];
      yd1[7] = yd[n7si2];

      zd1[0] = zd[n0si2];
      zd1[1] = zd[n1si2];
      zd1[2] = zd[n2si2];
      zd1[3] = zd[n3si2];
      zd1[4] = zd[n4si2];
      zd1[5] = zd[n5si2];
      zd1[6] = zd[n6si2];
      zd1[7] = zd[n7si2];

      coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

      CalcElemFBHourglassForce(xd1,yd1,zd1,
                      hourgam0,hourgam1,hourgam2,hourgam3,
                      hourgam4,hourgam5,hourgam6,hourgam7, coefficient,
                      &fx_elem[i3], &fy_elem[i3], &fz_elem[i3] );
    }
   ) ; 

   /* added tmp arrays for fault tolerance */
   Real_p fx_tmp  = Allocate<Real_t>(numNode) ;
   Real_p fy_tmp  = Allocate<Real_t>(numNode) ;
   Real_p fz_tmp  = Allocate<Real_t>(numNode) ;

   RAJA::forall<node_exec_policy>(*domNodeList, [=] RAJA_DEVICE (int gnode) {
      fx_tmp[gnode] = fx[gnode] ;
      fy_tmp[gnode] = fy[gnode] ;
      fz_tmp[gnode] = fz[gnode] ;
    }
   ) ;

   RAJA::forall<node_exec_policy>(*domNodeList, [=] RAJA_DEVICE (int gnode) {
      Index_t count = nodeElemStart[gnode+1] - nodeElemStart[gnode] ;
      Index_t *cornerList = &nodeElemCornerList[nodeElemStart[gnode]] ;
      Real_t fx_sum = Real_t(0.0) ;
      Real_t fy_sum = Real_t(0.0) ;
      Real_t fz_sum = Real_t(0.0) ;
      for (Index_t i=0 ; i < count ; ++i) {
         Index_t elem = cornerList[i] ;
         fx_sum += fx_elem[elem] ;
         fy_sum += fy_elem[elem] ;
         fz_sum += fz_elem[elem] ;
      }
      fx[gnode] = fx_tmp[gnode] + fx_sum ;
      fy[gnode] = fy_tmp[gnode] + fy_sum ;
      fz[gnode] = fz_tmp[gnode] + fz_sum ;
    }
   ) ;

   Release(&fz_tmp) ;
   Release(&fy_tmp) ;
   Release(&fx_tmp) ;

   Release(&fz_elem) ;
   Release(&fy_elem) ;
   Release(&fx_elem) ;
}

RAJA_STORAGE
void CalcHourglassControlForElems(Domain *domain,
                                  Real_p determ,
                                  Real_t hgcoef)
{
   Index_t numElem = domain->numElem ;
   Index_t numElem8 = numElem * 8 ;
   Real_p dvdx = Allocate<Real_t>(numElem8) ;
   Real_p dvdy = Allocate<Real_t>(numElem8) ;
   Real_p dvdz = Allocate<Real_t>(numElem8) ;
   Real_p x8n  = Allocate<Real_t>(numElem8) ;
   Real_p y8n  = Allocate<Real_t>(numElem8) ;
   Real_p z8n  = Allocate<Real_t>(numElem8) ;

   // For negative element volume check
   RAJA::ReduceMin<reduce_policy, Real_t> minvol(1.0);

   /* start loop over elements */
   RAJA::forall<elem_exec_policy>(*domain->domElemList, [=] RAJA_DEVICE (int idx) {

      Index_p elemToNode = &domain->nodelist[8*idx];
      CollectDomainNodesToElemNodes(domain->x, domain->y, domain->z, elemToNode,
                                    &x8n[8*idx], &y8n[8*idx], &z8n[8*idx] );

      CalcElemVolumeDerivative(&dvdx[8*idx], &dvdy[8*idx], &dvdz[8*idx],
                               & x8n[8*idx], & y8n[8*idx], & z8n[8*idx]);

      determ[idx] = domain->volo[idx] * domain->v[idx];

      determ[idx] = domain->volo[idx] * domain->v[idx];

      minvol.min(domain->v[idx]);

    }
   ) ;

   if ( Real_t(minvol) <= Real_t(0.0) ) {
      exit(VolumeError) ;
   }

   if ( hgcoef > Real_t(0.) ) {
      CalcFBHourglassForceForElems( numElem, domain->numNode,
                                    domain->nodelist,
                                    domain->ss, domain->elemMass,
                                    domain->xd, domain->yd, domain->zd,
                                    domain->fx, domain->fy, domain->fz,
                                    determ, x8n, y8n, z8n, dvdx, dvdy, dvdz,
                                    hgcoef, domain->nodeElemStart,
                                    domain->nodeElemCornerList,
                                    domain->domElemList, domain->domNodeList) ;
   }

   Release(&z8n) ;
   Release(&y8n) ;
   Release(&x8n) ;
   Release(&dvdz) ;
   Release(&dvdy) ;
   Release(&dvdx) ;

   return ;
}

RAJA_STORAGE
void CalcVolumeForceForElems(Domain *domain)
{
   Index_t numElem = domain->numElem ;
   if (numElem != 0) {
      Real_t  hgcoef = domain->hgcoef ;
      Real_p sigxx  = Allocate<Real_t>(numElem) ;
      Real_p sigyy  = Allocate<Real_t>(numElem) ;
      Real_p sigzz  = Allocate<Real_t>(numElem) ;
      Real_p determ = Allocate<Real_t>(numElem) ;

      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(domain->p, domain->q,
                              sigxx, sigyy, sigzz, domain->domElemList);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( numElem, domain->nodelist,
                               domain->x, domain->y, domain->z,
                               domain->fx, domain->fy, domain->fz,
                               sigxx, sigyy, sigzz, determ,
                               domain->nodeElemStart,
                               domain->nodeElemCornerList,
                               domain->domElemList, domain->domNodeList) ;

      // check for negative element volume
      RAJA::ReduceMin<reduce_policy, Real_t> minvol(1.0);
      RAJA::forall<elem_exec_policy>(*domain->domElemList, [=] RAJA_DEVICE (int k) {
         minvol.min(determ[k]);
       }
      ) ;

      if ( Real_t(minvol) <= Real_t(0.0)) {
         exit(VolumeError) ;
      }

      CalcHourglassControlForElems(domain, determ, hgcoef) ;

      Release(&determ) ;
      Release(&sigzz) ;
      Release(&sigyy) ;
      Release(&sigxx) ;
   }
}

RAJA_STORAGE
void CalcForceForNodes(Domain *domain)
{
  RAJA::forall<node_exec_policy>(*domain->domNodeList, [=] RAJA_DEVICE (int i) {
     domain->fx[i] = Real_t(0.0) ;
     domain->fy[i] = Real_t(0.0) ;
     domain->fz[i] = Real_t(0.0) ;
   }
  ) ;

  /* Calcforce calls partial, force, hourq */
  CalcVolumeForceForElems(domain) ;

  /* Calculate Nodal Forces at domain boundaries */
  /* problem->commSBN->Transfer(CommSBN::forces); */

}

RAJA_STORAGE
void CalcAccelerationForNodes(Real_p xdd, Real_p ydd, Real_p zdd,
                              Real_p fx, Real_p fy, Real_p fz,
                              Real_p nodalMass, RAJA::IndexSet *domNodeList)
{
   RAJA::forall<node_exec_policy>(*domNodeList, [=] RAJA_DEVICE (int i) {
      xdd[i] = fx[i] / nodalMass[i];
      ydd[i] = fy[i] / nodalMass[i];
      zdd[i] = fz[i] / nodalMass[i];
    }
   ) ;
}

RAJA_STORAGE
void ApplyAccelerationBoundaryConditionsForNodes(Real_p xdd, Real_p ydd,
                                                 Real_p zdd, Index_p symmX,
                                                 Index_p symmY, Index_p symmZ,
                                                 Index_t size)
{
  Index_t numNodeBC = (size+1)*(size+1) ;

  /*  !!! Interesting FT discussion here -- not converted !!! */
  /* What if the array index is corrupted? Out of bounds? */
  RAJA::forall<range_exec_policy>(int(0), int(numNodeBC), [=] RAJA_DEVICE (int i) {
     xdd[symmX[i]] = Real_t(0.0) ;
     ydd[symmY[i]] = Real_t(0.0) ;
     zdd[symmZ[i]] = Real_t(0.0) ;
   }
  ) ;
}

RAJA_STORAGE
void CalcVelocityForNodes(Index_t numNode, Real_p xd,  Real_p yd,  Real_p zd,
                          Real_p xdd, Real_p ydd, Real_p zdd,
                          const Real_t dt, const Real_t u_cut,
                          RAJA::IndexSet *domNodeList)
{
   Real_p xd_tmp = Allocate<Real_t>(numNode) ;
   Real_p yd_tmp = Allocate<Real_t>(numNode) ;
   Real_p zd_tmp = Allocate<Real_t>(numNode) ;

   /* for FT */
   RAJA::forall<node_exec_policy>( *domNodeList, [=] RAJA_DEVICE (int i) {
      xd_tmp[i] = xd[i] ;
      yd_tmp[i] = yd[i] ;
      zd_tmp[i] = zd[i] ;
    }
   ) ;

   RAJA::forall<node_exec_policy>( *domNodeList, [=] RAJA_DEVICE (int i) {
     Real_t xdtmp, ydtmp, zdtmp ;

     xdtmp = xd_tmp[i] + xdd[i] * dt ;
     if( FABS(xdtmp) < u_cut ) xdtmp = Real_t(0.0);
     xd[i] = xdtmp ;

     ydtmp = yd_tmp[i] + ydd[i] * dt ;
     if( FABS(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
     yd[i] = ydtmp ;

     zdtmp = zd_tmp[i] + zdd[i] * dt ;
     if( FABS(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
     zd[i] = zdtmp ;
    }
   ) ;

   Release(&zd_tmp) ;
   Release(&yd_tmp) ;
   Release(&xd_tmp) ;
}

RAJA_STORAGE
void CalcPositionForNodes(Index_t numNode, Real_p x,  Real_p y,  Real_p z,
                          Real_p xd, Real_p yd, Real_p zd,
                          const Real_t dt, RAJA::IndexSet *domNodeList)
{
   Real_p x_tmp = Allocate<Real_t>(numNode) ;
   Real_p y_tmp = Allocate<Real_t>(numNode) ;
   Real_p z_tmp = Allocate<Real_t>(numNode) ;

   /* for FT */
   RAJA::forall<node_exec_policy>( *domNodeList, [=] RAJA_DEVICE (int i) {
      x_tmp[i] = x[i] ;
      y_tmp[i] = y[i] ;
      z_tmp[i] = z[i] ;
    }
   ) ;

   RAJA::forall<node_exec_policy>( *domNodeList, [=] RAJA_DEVICE (int i) {
     x[i] = x_tmp[i] + xd[i] * dt ;
     y[i] = y_tmp[i] + yd[i] * dt ;
     z[i] = z_tmp[i] + zd[i] * dt ;
    }
   ) ;

   Release(&z_tmp) ;
   Release(&y_tmp) ;
   Release(&x_tmp) ;
}

RAJA_STORAGE
void LagrangeNodal(Domain *domain)
{
  const Real_t delt = domain->deltatime ;
  Real_t u_cut = domain->u_cut ;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes(domain);

  CalcAccelerationForNodes(domain->xdd, domain->ydd, domain->zdd,
                           domain->fx, domain->fy, domain->fz,
                           domain->nodalMass, domain->domNodeList);

  ApplyAccelerationBoundaryConditionsForNodes(domain->xdd, domain->ydd,
                                              domain->zdd, domain->symmX,
                                              domain->symmY, domain->symmZ,
                                              domain->sizeX );

  CalcVelocityForNodes( domain->numNode,
                        domain->xd,  domain->yd,  domain->zd,
                        domain->xdd, domain->ydd, domain->zdd,
                        delt, u_cut, domain->domNodeList) ;

  CalcPositionForNodes( domain->numNode,
                        domain->x,  domain->y,  domain->z,
                        domain->xd, domain->yd, domain->zd,
                        delt, domain->domNodeList );

  return;
}

RAJA_STORAGE
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

RAJA_STORAGE
Real_t CalcElemVolume(
                       const_Real_p x, const_Real_p y, const_Real_p z
                     )
{
return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

RAJA_STORAGE
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

RAJA_STORAGE
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

RAJA_STORAGE
void CalcElemVelocityGrandient( const Real_t* const xvel,
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

RAJA_STORAGE
void CalcKinematicsForElems( Index_p nodelist,
                             Real_p x,   Real_p y,   Real_p z,
                             Real_p xd,  Real_p yd,  Real_p zd,
                             Real_p dxx, Real_p dyy, Real_p dzz,
                             Real_p v, Real_p volo,
                             Real_p vnew, Real_p delv, Real_p arealg,
                             Real_t deltaTime, RAJA::IndexSet *domElemList )
{
  // loop over all elements
  RAJA::forall<elem_exec_policy>(*domElemList, [=] RAJA_DEVICE (int k) {
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
    const Index_p elemToNode = &nodelist[8*k] ;

    // get nodal coordinates from global arrays and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemToNode[lnode];
      x_local[lnode] = x[gnode];
      y_local[lnode] = y[gnode];
      z_local[lnode] = z[gnode];
    }

    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local );
    relativeVolume = volume / volo[k] ;
    vnew[k] = relativeVolume ;
    delv[k] = relativeVolume - v[k] ;

    // set characteristic length
    arealg[k] = CalcElemCharacteristicLength(x_local, y_local, z_local,
                                             volume);

    // get nodal velocities from global array and copy into local arrays.
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemToNode[lnode];
      xd_local[lnode] = xd[gnode];
      yd_local[lnode] = yd[gnode];
      zd_local[lnode] = zd[gnode];
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

    CalcElemVelocityGrandient( xd_local, yd_local, zd_local,
                               B, detJ, D );

    // put velocity gradient quantities into their global arrays.
    dxx[k] = D[0];
    dyy[k] = D[1];
    dzz[k] = D[2];
   }
  ) ;
}

RAJA_STORAGE
void CalcLagrangeElements(Domain *domain)
{
   Index_t numElem = domain->numElem ;
   if (numElem > 0) {
      const Real_t deltatime = domain->deltatime ;
      Real_p dxx_tmp = Allocate<Real_t>(numElem) ;
      Real_p dyy_tmp = Allocate<Real_t>(numElem) ;
      Real_p dzz_tmp = Allocate<Real_t>(numElem) ;

      domain->dxx  = Allocate<Real_t>(numElem) ; /* principal strains */
      domain->dyy  = Allocate<Real_t>(numElem) ;
      domain->dzz  = Allocate<Real_t>(numElem) ;

      CalcKinematicsForElems(domain->nodelist,
                             domain->x, domain->y, domain->z,
                             domain->xd, domain->yd, domain->zd,
                             domain->dxx, domain->dyy, domain->dzz,
                             domain->v, domain->volo,
                             domain->vnew, domain->delv, domain->arealg,
                             deltatime, domain->domElemList) ;



      /* For FT... since domain dxx, dyy, dzz are not used, not really needed */
      RAJA::forall<elem_exec_policy>( *domain->domElemList, [=] RAJA_DEVICE (int k) {
         dxx_tmp[k] =  domain->dxx[k] ;
         dyy_tmp[k] =  domain->dyy[k] ;
         dzz_tmp[k] =  domain->dzz[k] ;
       }
      ) ;

      // check for negative element volume
      RAJA::ReduceMin<reduce_policy, Real_t> minvol(1.0);

      // element loop to do some stuff not included in the elemlib function.
      RAJA::forall<elem_exec_policy>( *domain->domElemList, [=] RAJA_DEVICE (int k) {
        // calc strain rate and apply as constraint (only done in FB element)
        Real_t vdov = dxx_tmp[k] + dyy_tmp[k] + dzz_tmp[k] ;
        Real_t vdovthird = vdov/Real_t(3.0) ;
        
        // make the rate of deformation tensor deviatoric
        domain->vdov[k] = vdov ;
        domain->dxx[k] = dxx_tmp[k] - vdovthird ;
        domain->dyy[k] = dyy_tmp[k] - vdovthird ;
        domain->dzz[k] = dzz_tmp[k] - vdovthird ;

        minvol.min(domain->vnew[k]);
       }
      ) ;

      if ( Real_t(minvol) <= Real_t(0.0)) {
         exit(VolumeError) ;
      }

      Release(&domain->dzz) ;
      Release(&domain->dyy) ;
      Release(&domain->dxx) ;

      Release(&dzz_tmp) ;
      Release(&dyy_tmp) ;
      Release(&dxx_tmp) ;
   }
}

RAJA_STORAGE
void CalcMonotonicQGradientsForElems(Real_p x,  Real_p y,  Real_p z,
                                     Real_p xd, Real_p yd, Real_p zd,
                                     Real_p volo, Real_p vnew,
                                     Real_p delv_xi,
                                     Real_p delv_eta,
                                     Real_p delv_zeta,
                                     Real_p delx_xi,
                                     Real_p delx_eta,
                                     Real_p delx_zeta,
                                     Index_p nodelist,
                                     RAJA::IndexSet *domElemList)
{
#define SUM4(a,b,c,d) (a + b + c + d)

   RAJA::forall<elem_exec_policy>(*domElemList, [=] RAJA_DEVICE (int i) {
      const Real_t ptiny = Real_t(1.e-36) ;
      Real_t ax,ay,az ;
      Real_t dxv,dyv,dzv ;

      Index_p elemToNode = &nodelist[8*i];
      Index_t n0 = elemToNode[0] ;
      Index_t n1 = elemToNode[1] ;
      Index_t n2 = elemToNode[2] ;
      Index_t n3 = elemToNode[3] ;
      Index_t n4 = elemToNode[4] ;
      Index_t n5 = elemToNode[5] ;
      Index_t n6 = elemToNode[6] ;
      Index_t n7 = elemToNode[7] ;

      Real_t x0 = x[n0] ;
      Real_t x1 = x[n1] ;
      Real_t x2 = x[n2] ;
      Real_t x3 = x[n3] ;
      Real_t x4 = x[n4] ;
      Real_t x5 = x[n5] ;
      Real_t x6 = x[n6] ;
      Real_t x7 = x[n7] ;

      Real_t y0 = y[n0] ;
      Real_t y1 = y[n1] ;
      Real_t y2 = y[n2] ;
      Real_t y3 = y[n3] ;
      Real_t y4 = y[n4] ;
      Real_t y5 = y[n5] ;
      Real_t y6 = y[n6] ;
      Real_t y7 = y[n7] ;

      Real_t z0 = z[n0] ;
      Real_t z1 = z[n1] ;
      Real_t z2 = z[n2] ;
      Real_t z3 = z[n3] ;
      Real_t z4 = z[n4] ;
      Real_t z5 = z[n5] ;
      Real_t z6 = z[n6] ;
      Real_t z7 = z[n7] ;

      Real_t xv0 = xd[n0] ;
      Real_t xv1 = xd[n1] ;
      Real_t xv2 = xd[n2] ;
      Real_t xv3 = xd[n3] ;
      Real_t xv4 = xd[n4] ;
      Real_t xv5 = xd[n5] ;
      Real_t xv6 = xd[n6] ;
      Real_t xv7 = xd[n7] ;

      Real_t yv0 = yd[n0] ;
      Real_t yv1 = yd[n1] ;
      Real_t yv2 = yd[n2] ;
      Real_t yv3 = yd[n3] ;
      Real_t yv4 = yd[n4] ;
      Real_t yv5 = yd[n5] ;
      Real_t yv6 = yd[n6] ;
      Real_t yv7 = yd[n7] ;

      Real_t zv0 = zd[n0] ;
      Real_t zv1 = zd[n1] ;
      Real_t zv2 = zd[n2] ;
      Real_t zv3 = zd[n3] ;
      Real_t zv4 = zd[n4] ;
      Real_t zv5 = zd[n5] ;
      Real_t zv6 = zd[n6] ;
      Real_t zv7 = zd[n7] ;

      Real_t vol = volo[i]*vnew[i] ;
      Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

      Real_t dxj = Real_t(-0.25)*(SUM4(x0,x1,x5,x4) - SUM4(x3,x2,x6,x7)) ;
      Real_t dyj = Real_t(-0.25)*(SUM4(y0,y1,y5,y4) - SUM4(y3,y2,y6,y7)) ;
      Real_t dzj = Real_t(-0.25)*(SUM4(z0,z1,z5,z4) - SUM4(z3,z2,z6,z7)) ;

      Real_t dxi = Real_t( 0.25)*(SUM4(x1,x2,x6,x5) - SUM4(x0,x3,x7,x4)) ;
      Real_t dyi = Real_t( 0.25)*(SUM4(y1,y2,y6,y5) - SUM4(y0,y3,y7,y4)) ;
      Real_t dzi = Real_t( 0.25)*(SUM4(z1,z2,z6,z5) - SUM4(z0,z3,z7,z4)) ;

      Real_t dxk = Real_t( 0.25)*(SUM4(x4,x5,x6,x7) - SUM4(x0,x1,x2,x3)) ;
      Real_t dyk = Real_t( 0.25)*(SUM4(y4,y5,y6,y7) - SUM4(y0,y1,y2,y3)) ;
      Real_t dzk = Real_t( 0.25)*(SUM4(z4,z5,z6,z7) - SUM4(z0,z1,z2,z3)) ;

      /* find delvk and delxk ( i cross j ) */

      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      delx_zeta[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*(SUM4(xv4,xv5,xv6,xv7) - SUM4(xv0,xv1,xv2,xv3)) ;
      dyv = Real_t(0.25)*(SUM4(yv4,yv5,yv6,yv7) - SUM4(yv0,yv1,yv2,yv3)) ;
      dzv = Real_t(0.25)*(SUM4(zv4,zv5,zv6,zv7) - SUM4(zv0,zv1,zv2,zv3)) ;

      delv_zeta[i] = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */

      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      delx_xi[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*(SUM4(xv1,xv2,xv6,xv5) - SUM4(xv0,xv3,xv7,xv4)) ;
      dyv = Real_t(0.25)*(SUM4(yv1,yv2,yv6,yv5) - SUM4(yv0,yv3,yv7,yv4)) ;
      dzv = Real_t(0.25)*(SUM4(zv1,zv2,zv6,zv5) - SUM4(zv0,zv3,zv7,zv4)) ;

      delv_xi[i] = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */

      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      delx_eta[i] = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(-0.25)*(SUM4(xv0,xv1,xv5,xv4) - SUM4(xv3,xv2,xv6,xv7)) ;
      dyv = Real_t(-0.25)*(SUM4(yv0,yv1,yv5,yv4) - SUM4(yv3,yv2,yv6,yv7)) ;
      dzv = Real_t(-0.25)*(SUM4(zv0,zv1,zv5,zv4) - SUM4(zv3,zv2,zv6,zv7)) ;

      delv_eta[i] = ax*dxv + ay*dyv + az*dzv ;
    }
   ) ;

#undef SUM4
}

RAJA_STORAGE
void CalcMonotonicQRegionForElems(
                           RAJA::IndexSet *matElemList, Index_p elemBC,
                           Index_p lxim,   Index_p lxip,
                           Index_p letam,  Index_p letap,
                           Index_p lzetam, Index_p lzetap,
                           Real_p delv_xi,Real_p delv_eta,Real_p delv_zeta,
                           Real_p delx_xi,Real_p delx_eta,Real_p delx_zeta,
                           Real_p vdov, Real_p volo, Real_p vnew,
                           Real_p elemMass, Real_p qq, Real_p ql,
                           Real_t qlc_monoq, Real_t qqc_monoq,
                           Real_t monoq_limiter_mult,
                           Real_t monoq_max_slope,
                           Real_t ptiny )
{
   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int i) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Int_t bcMask = elemBC[i] ;
      Real_t delvm, delvp ;

      /*  phixi     */
      Real_t norm = Real_t(1.) / ( delv_xi[i] + ptiny ) ;

      switch (bcMask & XI_M) {
         case 0:         delvm = delv_xi[lxim[i]] ; break ;
         case XI_M_SYMM: delvm = delv_xi[i] ;       break ;
         case XI_M_FREE: delvm = Real_t(0.0) ;      break ;
         default:        /* ERROR */ ;              break ;
      }
      switch (bcMask & XI_P) {
         case 0:         delvp = delv_xi[lxip[i]] ; break ;
         case XI_P_SYMM: delvp = delv_xi[i] ;       break ;
         case XI_P_FREE: delvp = Real_t(0.0) ;      break ;
         default:        /* ERROR */ ;              break ;
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
      norm = Real_t(1.) / ( delv_eta[i] + ptiny ) ;

      switch (bcMask & ETA_M) {
         case 0:          delvm = delv_eta[letam[i]] ; break ;
         case ETA_M_SYMM: delvm = delv_eta[i] ;        break ;
         case ETA_M_FREE: delvm = Real_t(0.0) ;        break ;
         default:         /* ERROR */ ;                break ;
      }
      switch (bcMask & ETA_P) {
         case 0:          delvp = delv_eta[letap[i]] ; break ;
         case ETA_P_SYMM: delvp = delv_eta[i] ;        break ;
         case ETA_P_FREE: delvp = Real_t(0.0) ;        break ;
         default:         /* ERROR */ ;                break ;
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
      norm = Real_t(1.) / ( delv_zeta[i] + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case 0:           delvm = delv_zeta[lzetam[i]] ; break ;
         case ZETA_M_SYMM: delvm = delv_zeta[i] ;         break ;
         case ZETA_M_FREE: delvm = Real_t(0.0) ;          break ;
         default:          /* ERROR */ ;                  break ;
      }
      switch (bcMask & ZETA_P) {
         case 0:           delvp = delv_zeta[lzetap[i]] ; break ;
         case ZETA_P_SYMM: delvp = delv_zeta[i] ;         break ;
         case ZETA_P_FREE: delvp = Real_t(0.0) ;          break ;
         default:          /* ERROR */ ;                  break ;
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

      if ( vdov[i] > Real_t(0.) )  {
         qlin  = Real_t(0.) ;
         qquad = Real_t(0.) ;
      }
      else {
         Real_t delvxxi   = delv_xi[i]   * delx_xi[i]   ;
         Real_t delvxeta  = delv_eta[i]  * delx_eta[i]  ;
         Real_t delvxzeta = delv_zeta[i] * delx_zeta[i] ;

         if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
         if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
         if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

         Real_t rho = elemMass[i] / (volo[i] * vnew[i]) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
      }

      qq[i] = qquad ;
      ql[i] = qlin  ;
    }
   ) ;
}

RAJA_STORAGE
void CalcMonotonicQForElems(Domain *domain)
{  
   //
   // calculate the monotonic q for pure regions
   //
   Index_t numElem = domain->numElem ;
   if (numElem > 0) {
      //
      // initialize parameters
      // 
      const Real_t ptiny = Real_t(1.e-36) ;

      CalcMonotonicQRegionForElems(
                           domain->matElemList, domain->elemBC,
                           domain->lxim,   domain->lxip,
                           domain->letam,  domain->letap,
                           domain->lzetam, domain->lzetap,
                           domain->delv_xi,domain->delv_eta,domain->delv_zeta,
                           domain->delx_xi,domain->delx_eta,domain->delx_zeta,
                           domain->vdov, domain->volo, domain->vnew,
                           domain->elemMass, domain->qq, domain->ql,
                           domain->qlc_monoq, domain->qqc_monoq,
                           domain->monoq_limiter_mult,
                           domain->monoq_max_slope,
                           ptiny );
   }
}

RAJA_STORAGE
void CalcQForElems(Domain *domain)
{
   //
   // MONOTONIC Q option
   //

   Index_t numElem = domain->numElem ;

   if (numElem != 0) {
      /* allocate domain length arrays */

      domain->delv_xi = Allocate<Real_t>(numElem) ;   /* velocity gradient */
      domain->delv_eta = Allocate<Real_t>(numElem) ;
      domain->delv_zeta = Allocate<Real_t>(numElem) ;

      domain->delx_xi = Allocate<Real_t>(numElem) ;   /* position gradient */
      domain->delx_eta = Allocate<Real_t>(numElem) ;
      domain->delx_zeta = Allocate<Real_t>(numElem) ;

      /* Calculate velocity gradients, applied at the domain level */
      CalcMonotonicQGradientsForElems(domain->x,  domain->y,  domain->z,
                                      domain->xd, domain->yd, domain->zd,
                                      domain->volo, domain->vnew,
                                      domain->delv_xi,
                                      domain->delv_eta,
                                      domain->delv_zeta,
                                      domain->delx_xi,
                                      domain->delx_eta,
                                      domain->delx_zeta,
                                      domain->nodelist,
                                      domain->domElemList) ;

      /* Transfer veloctiy gradients in the first order elements */
      /* problem->commElements->Transfer(CommElements::monoQ) ; */

      /* This will be applied at the region level */
      CalcMonotonicQForElems(domain) ;

      /* release domain length arrays */

      Release(&domain->delx_zeta) ;
      Release(&domain->delx_eta) ;
      Release(&domain->delx_xi) ;

      Release(&domain->delv_zeta) ;
      Release(&domain->delv_eta) ;
      Release(&domain->delv_xi) ;

      /* Don't allow excessive artificial viscosity */
      Real_t qstop = domain->qstop ;
      Index_t idx = -1; 
      RAJA::forall<elem_exec_policy>( *domain->domElemList, [=] RAJA_DEVICE (int i) {
         if ( domain->q[i] > qstop ) {
            idx = i ;
            // break ;
         }
       }
      ) ;

      if(idx >= 0) {
         exit(QStopError) ;
      }
   }
}

RAJA_STORAGE
void CalcPressureForElems(Real_p p_new, Real_p bvc,
                          Real_p pbvc, Real_p e_old,
                          Real_p compression, Real_p vnewc,
                          Real_t pmin,
                          Real_t p_cut, Real_t eosvmax,
                          RAJA::IndexSet *matElemList)
{
   const Real_t c1s = Real_t(2.0)/Real_t(3.0) ;
   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int i) {
      bvc[i] = c1s * (compression[i] + Real_t(1.));
      pbvc[i] = c1s;
    }
   ) ;

   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int i) {
      p_new[i] = bvc[i] * e_old[i] ;

      if    (FABS(p_new[i]) <  p_cut   )
         p_new[i] = Real_t(0.0) ;

      if    ( vnewc[i] >= eosvmax ) /* impossible condition here? */
         p_new[i] = Real_t(0.0) ;

      if    (p_new[i]       <  pmin)
         p_new[i]   = pmin ;
    }
   ) ;
}

RAJA_STORAGE
void CalcEnergyForElems(Real_p p_new, Real_p e_new, Real_p q_new,
                        Real_p bvc, Real_p pbvc,
                        Real_p p_old, Real_p e_old, Real_p q_old,
                        Real_p compression, Real_p compHalfStep,
                        Real_p vnewc, Real_p work, Real_p delvc, Real_t pmin,
                        Real_t p_cut, Real_t  e_cut, Real_t q_cut, Real_t emin,
                        Real_p qq_old, Real_p ql_old,
                        Real_t rho0,
                        Real_t eosvmax,
                        RAJA::IndexSet *matElemList,
                        Index_t length)
{
   const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
   Real_p pHalfStep = Allocate<Real_t>(length) ;
   Real_p e_new_tmp = Allocate<Real_t>(length) ;

   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int i) {
      e_new[i] = e_old[i] - Real_t(0.5) * delvc[i] * (p_old[i] + q_old[i])
         + Real_t(0.5) * work[i];

      if (e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
    }
   ) ;

   CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                   pmin, p_cut, eosvmax, matElemList);

   /* for FT */
   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int i) {
      e_new_tmp[i] = e_new[i] ;
    }
   ) ;

   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int i) {
      Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[i]) ;

      if ( delvc[i] > Real_t(0.) ) {
         q_new[i] /* = qq_old[i] = ql_old[i] */ = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;
      }

      e_new[i] = e_new_tmp[i] + Real_t(0.5) * (delvc[i]
           * (  Real_t(3.0)*(p_old[i]     + q_old[i])
              - Real_t(4.0)*(pHalfStep[i] + q_new[i])) + work[i] ) ;

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
    }
   ) ;

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, matElemList);

   /* for FT */
   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int i) {
      e_new_tmp[i] = e_new[i] ;
    }
   ) ;

   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int i) {
      Real_t q_tilde ;

      if (delvc[i] > Real_t(0.)) {
         q_tilde = Real_t(0.) ;
      }
      else {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_tilde = (ssc*ql_old[i] + qq_old[i]) ;
      }

      e_new[i] = e_new_tmp[i] - (  Real_t(7.0)*(p_old[i]     + q_old[i])
                   - Real_t(8.0)*(pHalfStep[i] + q_new[i])
                   + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
    }
   ) ;

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax, matElemList);

   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int i) {

      if ( delvc[i] <= Real_t(0.) ) {
         Real_t ssc = ( pbvc[i] * e_new[i]
                 + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;

         if ( ssc <= Real_t(.1111111e-36) ) {
            ssc = Real_t(.3333333e-18) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;

         if (FABS(q_new[i]) < q_cut) q_new[i] = Real_t(0.) ;
      }
    }
   ) ;

   Release(&e_new_tmp) ;
   Release(&pHalfStep) ;

   return ;
}

RAJA_STORAGE
void CalcSoundSpeedForElems(RAJA::IndexSet *matElemList, Real_p ss,
                            Real_p vnewc, Real_t rho0, Real_p enewc,
                            Real_p pnewc, Real_p pbvc,
                            Real_p bvc, Real_t ss4o3)
{
   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int iz) {
      Real_t ssTmp = (pbvc[iz] * enewc[iz] + vnewc[iz] * vnewc[iz] *
                 bvc[iz] * pnewc[iz]) / rho0;
      if (ssTmp <= Real_t(.1111111e-36)) {
         ssTmp = Real_t(.3333333e-18);
      }
      else {
         ssTmp = SQRT(ssTmp);
      }
      ss[iz] = ssTmp ;
    }
   ) ;
}

RAJA_STORAGE
void EvalEOSForElems(Domain *domain, Real_p vnewc, Index_t numElem)
{
   Real_t  e_cut = domain->e_cut ;
   Real_t  p_cut = domain->p_cut ;
   Real_t  ss4o3 = domain->ss4o3 ;
   Real_t  q_cut = domain->q_cut ;

   Real_t eosvmax = domain->eosvmax ;
   Real_t eosvmin = domain->eosvmin ;
   Real_t pmin    = domain->pmin ;
   Real_t emin    = domain->emin ;
   Real_t rho0    = domain->refdens ;

   /* allocate *domain length* arrays.  */
   /* wastes memory, but allows us to get */
   /* around a "temporary workset" issue */
   /* we have not yet addressed. */
   Real_p delvc = domain->delv ;
   Real_p p_old = Allocate<Real_t>(numElem) ;
   Real_p compression = Allocate<Real_t>(numElem) ;
   Real_p compHalfStep = Allocate<Real_t>(numElem) ;
   Real_p work = Allocate<Real_t>(numElem) ;
   Real_p p_new = Allocate<Real_t>(numElem) ;
   Real_p e_new = Allocate<Real_t>(numElem) ;
   Real_p q_new = Allocate<Real_t>(numElem) ;
   Real_p bvc = Allocate<Real_t>(numElem) ;
   Real_p pbvc = Allocate<Real_t>(numElem) ;

   /* compress data, minimal set */
   RAJA::forall<mat_exec_policy>( *domain->matElemList, [=] RAJA_DEVICE (int zidx) {
      p_old[zidx] = domain->p[zidx] ;
    }
   ) ;

   RAJA::forall<mat_exec_policy>( *domain->matElemList, [=] RAJA_DEVICE (int zidx) {
      Real_t vchalf ;
      compression[zidx] = Real_t(1.) / vnewc[zidx] - Real_t(1.);
      vchalf = vnewc[zidx] - delvc[zidx] * Real_t(.5);
      compHalfStep[zidx] = Real_t(1.) / vchalf - Real_t(1.);
    }
   ) ;

   /* Check for v > eosvmax or v < eosvmin */
   if ( eosvmin != Real_t(0.) ) {
      RAJA::forall<mat_exec_policy>( *domain->matElemList, [=] RAJA_DEVICE (int zidx) {
         if (vnewc[zidx] <= eosvmin) { /* impossible due to calling func? */
            compHalfStep[zidx] = compression[zidx] ;
         }
       }
      ) ;
   }
   if ( eosvmax != Real_t(0.) ) {
      RAJA::forall<mat_exec_policy>( *domain->matElemList, [=] RAJA_DEVICE (int zidx) {
         if (vnewc[zidx] >= eosvmax) { /* impossible due to calling func? */
            p_old[zidx]        = Real_t(0.) ;
            compression[zidx]  = Real_t(0.) ;
            compHalfStep[zidx] = Real_t(0.) ;
         }
       }
      ) ;
   }

   RAJA::forall<mat_exec_policy>( *domain->matElemList, [=] RAJA_DEVICE (int zidx) {
      work[zidx] = Real_t(0.) ; 
    }
   ) ;

   CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc,
                 p_old, domain->e,  domain->q, compression, compHalfStep,
                 vnewc, work,  delvc, pmin,
                 p_cut, e_cut, q_cut, emin,
                 domain->qq, domain->ql, rho0, eosvmax,
                 domain->matElemList, numElem);


   RAJA::forall<mat_exec_policy>( *domain->matElemList, [=] RAJA_DEVICE (int zidx) {
      domain->p[zidx] = p_new[zidx] ;
      domain->e[zidx] = e_new[zidx] ;
      domain->q[zidx] = q_new[zidx] ;
    }
   ) ;

   CalcSoundSpeedForElems(domain->matElemList, domain->ss,
             vnewc, rho0, e_new, p_new,
             pbvc, bvc, ss4o3) ;

   Release(&pbvc) ;
   Release(&bvc) ;
   Release(&q_new) ;
   Release(&e_new) ;
   Release(&p_new) ;
   Release(&work) ;
   Release(&compHalfStep) ;
   Release(&compression) ;
   Release(&p_old) ;
}

RAJA_STORAGE
void ApplyMaterialPropertiesForElems(Domain *domain)
{
  Index_t numElem = domain->numElem ;

  if (numElem != 0) {
    /* Expose all of the variables needed for material evaluation */
    Real_t eosvmin = domain->eosvmin ;
    Real_t eosvmax = domain->eosvmax ;

    /* create a domain length (not material length) temporary */
    /* we are assuming here that the number of dense ranges is */
    /* much greater than the number of sigletons.  We are also */
    /* assuming it is ok to allocate a domain length temporary */
    /* rather than a material length temporary. */

    Real_p vnewc = Allocate<Real_t>(numElem) ;

    RAJA::forall<mat_exec_policy>( *domain->matElemList, [=] RAJA_DEVICE (int zn) {
       vnewc[zn] = domain->vnew[zn] ;

       if (eosvmin != Real_t(0.)) {
          if (vnewc[zn] < eosvmin) {
             vnewc[zn] = eosvmin ;
          }
       }

       if (eosvmax != Real_t(0.)) {
          if (vnewc[zn] > eosvmax) {
             vnewc[zn] = eosvmax ;
          }
       }

     }
    ) ;

    // check for negative element volume
    RAJA::ReduceMin<reduce_policy, Real_t> minvol(1.0);

    RAJA::forall<mat_exec_policy>( *domain->matElemList, [=] RAJA_DEVICE (int zn) {
       Real_t vc = domain->v[zn] ;
       if (eosvmin != Real_t(0.)) {
          if (vc < eosvmin) {
             vc = eosvmin ;
          }
       }
       if (eosvmax != Real_t(0.)) {
          if (vc > eosvmax) {
             vc = eosvmax ;
          }
       }

       minvol.min(vc);
     }
    ) ;

    if ( Real_t(minvol) <= Real_t(0.) ) {
       exit(VolumeError) ;
    }

    EvalEOSForElems(domain, vnewc, numElem);

    Release(&vnewc) ;

  }
}

RAJA_STORAGE
void UpdateVolumesForElems(Real_p vnew, Real_p v,
                           Real_t v_cut, Index_t length)
{
   if (length != 0) {
      RAJA::forall<range_exec_policy>( int(0), int(length), [=] RAJA_DEVICE (int i) {
         Real_t tmpV = vnew[i] ;

         if ( FABS(tmpV - Real_t(1.0)) < v_cut )
            tmpV = Real_t(1.0) ;

         v[i] = tmpV ;
       }
      ) ;
   }

   return ;
}

RAJA_STORAGE
void LagrangeElements(Domain *domain, Index_t numElem)
{
  /* new relative volume -- temporary */
  domain->vnew = Allocate<Real_t>(numElem) ;

  CalcLagrangeElements(domain) ;

  /* Calculate Q.  (Monotonic q option requires communication) */
  CalcQForElems(domain) ;

  ApplyMaterialPropertiesForElems(domain) ;

  UpdateVolumesForElems(domain->vnew, domain->v,
                        domain->v_cut, numElem) ;

  Release(&domain->vnew) ;
}

RAJA_STORAGE
void CalcCourantConstraintForElems(RAJA::IndexSet *matElemList, Real_p ss,
                                   Real_p vdov, Real_p arealg,
                                   Real_t qqc, Real_t *dtcourant)
{
   RAJA::ReduceMin<reduce_policy, Real_t> dtcourantLoc(Real_t(1.0e+20)) ;
   Real_t  qqc2 = Real_t(64.0) * qqc * qqc ;

   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int indx) {
      Real_t dtf = ss[indx] * ss[indx] ;
      Real_t dtf_cmp ;

      if ( vdov[indx] < Real_t(0.) ) {
         dtf += qqc2 * arealg[indx] * arealg[indx] * vdov[indx] * vdov[indx] ;
      }

      dtf_cmp = (vdov[indx] != Real_t(0.))
              ?  arealg[indx] / SQRT(dtf) : Real_t(1.0e+20) ;

      /* determine minimum timestep with its corresponding elem */
      dtcourantLoc.min(dtf_cmp) ;
   } ) ;

   /* Don't try to register a time constraint if none of the elements
    * were active */
   if (dtcourantLoc < Real_t(1.0e+20)) {
      *dtcourant = dtcourantLoc ;
   }

   return ;
}

RAJA_STORAGE
void CalcHydroConstraintForElems(RAJA::IndexSet *matElemList, Real_p vdov,
                                 Real_t dvovmax, Real_t *dthydro)
{
   RAJA::ReduceMin<reduce_policy, Real_t> dthydroLoc(Real_t(1.0e+20)) ;

   RAJA::forall<mat_exec_policy>( *matElemList, [=] RAJA_DEVICE (int indx) {

      Real_t dtvov_cmp = (vdov[indx] != Real_t(0.))
                       ? (dvovmax / (FABS(vdov[indx])+Real_t(1.e-20)))
                       : Real_t(1.0e+10) ;

      dthydroLoc.min(dtvov_cmp) ;
   } ) ;

   if (dthydroLoc < Real_t(1.0e+20)) {
      *dthydro = dthydroLoc ;
   }

   return ;
}

RAJA_STORAGE
void CalcTimeConstraintsForElems(Domain *domain) {
   /* evaluate time constraint */
   /* normally,  this call is on a per region basis */
   CalcCourantConstraintForElems(domain->matElemList, domain->ss,
                                 domain->vdov, domain->arealg,
                                 domain->qqc, &domain->dtcourant) ;

   /* check hydro constraint */
   CalcHydroConstraintForElems(domain->matElemList, domain->vdov,
                               domain->dvovmax, &domain->dthydro) ;
}

RAJA_STORAGE
void LagrangeLeapFrog(Domain *domain)
{
   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal(domain);

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(domain, domain->numElem);

   CalcTimeConstraintsForElems(domain);

}

int main(int argc, char *argv[])
{

   RAJA::Timer timer_main;
   RAJA::Timer timer_cycle;

   timer_main.start("timer_main");

   Real_t tx, ty, tz ;
   Index_t nidx, zidx ;
   struct Domain domain ;
   int maxIter = 1024*1024 ;

   Index_t edgeElems = lulesh_edge_elems ;

   for (int i=1; i<argc; ++i) {
      if (strcmp(argv[i], "-p") == 0) {
         show_run_progress = true ;
      }
      else if (strcmp(argv[i], "-i") == 0) {
         if ((i+1 < argc) && isdigit(argv[i+1][0])) {
            maxIter = atoi(argv[i+1]) ;
            ++i;
         }
         else  {
            printf("Iteration (-i) option has bad argument -- ignoring\n") ;
         }
      }
      else if (strcmp(argv[i], "-s") == 0) {
         if ((i+1 < argc) && isdigit(argv[i+1][0])) {
            edgeElems = atoi(argv[i+1]) ;
            ++i;
         }
         else  {
            printf("Size (-s) option has bad argument -- ignoring\n") ;
         }
      }
   }

   Index_t edgeNodes = edgeElems+1 ;

#ifdef RAJA_ENABLE_FT
   /* mock up fault tolerance */
   sigalrmact.sa_handler = simulate_fault ;
   sigalrmact.sa_flags = 0 ;
   sigemptyset(&sigalrmact.sa_mask) ;

   printf("signal handler installed\n") ;
   if (sigaction(SIGUSR2, &sigalrmact, NULL) < 0) {
      perror("sigaction") ;
      exit(2) ;
   }
#endif

   /****************************/
   /*  Print run parameters    */
   /****************************/
   printf("LULESH parallel run parameters:\n");
   printf("\t stop time = %e\n", double(lulesh_stop_time)) ;
   if ( lulesh_time_step > 0 ) {
     printf("\t Fixed time step = %e\n", double(lulesh_time_step)) ;
   } else {
     printf("\t CFL-controlled: initial time step = %e\n", 
            double(-lulesh_time_step)) ;
   }
   printf("\t Mesh size = %i x %i x %i\n", 
          lulesh_edge_elems, lulesh_edge_elems, lulesh_edge_elems) ;

   switch (lulesh_tiling_mode) {
      case Canonical:
      { 
         printf("\t Tiling mode is 'Canonical'\n");
         break;
      }
      case Tiled_Index:
      { 
         printf("\t Tiling mode is 'Tiled_Index'\n");
         break;
      }
      case Tiled_Order:
      { 
         printf("\t Tiling mode is 'Tiled_Order'\n");
         break;
      }
      case Tiled_LockFree:
      { 
         printf("\t Tiling mode is 'Canonical'\n");
         break;
      }
      default :
      {
         printf("Unknown tiling mode!!!\n");
      }
   }

   if (lulesh_tiling_mode != Canonical) {
      printf("\t Mesh tiling = %i x %i x %i\n",
             lulesh_xtile, lulesh_ytile, lulesh_ztile) ;
   }

   /****************************/
   /*   Initialize Sedov Mesh  */
   /****************************/

   /* construct a uniform box for this processor */

   domain.sizeX = edgeElems ;
   domain.sizeY = edgeElems ;
   domain.sizeZ = edgeElems ;
   domain.numElem = edgeElems*edgeElems*edgeElems ;

   domain.numNode = edgeNodes*edgeNodes*edgeNodes ;

   Index_t domElems = domain.numElem ;
   Index_t domNodes = domain.numNode ;

   /*************************/
   /* allocate field memory */
   /*************************/
   
   /*****************/
   /* Elem-centered */
   /*****************/

   /* elemToNode connectivity */
   domain.nodelist = Allocate<Index_t>(8*domElems) ;

   /* elem connectivity through face */
   domain.lxim = Allocate<Index_t>(domElems) ;
   domain.lxip = Allocate<Index_t>(domElems)  ;
   domain.letam = Allocate<Index_t>(domElems) ;
   domain.letap = Allocate<Index_t>(domElems) ;
   domain.lzetam = Allocate<Index_t>(domElems) ;
   domain.lzetap = Allocate<Index_t>(domElems) ;

   /* elem face symm/free-surface flag */
   domain.elemBC = Allocate<Int_t>(domElems) ;

   domain.e = Allocate<Real_t>(domElems) ;   /* energy */
   domain.p = Allocate<Real_t>(domElems) ;   /* pressure */

   domain.q = Allocate<Real_t>(domElems) ;   /* q */
   domain.ql = Allocate<Real_t>(domElems) ;  /* linear term for q */
   domain.qq = Allocate<Real_t>(domElems) ;  /* quadratic term for q */

   domain.v = Allocate<Real_t>(domElems) ;     /* relative volume */
   domain.volo = Allocate<Real_t>(domElems) ;  /* reference volume */
   domain.delv = Allocate<Real_t>(domElems) ;  /* m_vnew - m_v */
   domain.vdov = Allocate<Real_t>(domElems) ;  /* volume deriv over volume */

   /* elem characteristic length */
   domain.arealg = Allocate<Real_t>(domElems) ;

   domain.ss = Allocate<Real_t>(domElems) ;    /* "sound speed" */

   domain.elemMass = Allocate<Real_t>(domElems) ;  /* mass */

   /*****************/
   /* Node-centered */
   /*****************/

   domain.x = Allocate<Real_t>(domNodes) ;  /* coordinates */
   domain.y = Allocate<Real_t>(domNodes)  ;
   domain.z = Allocate<Real_t>(domNodes)  ;

   domain.xd = Allocate<Real_t>(domNodes) ; /* velocities */
   domain.yd = Allocate<Real_t>(domNodes)  ;
   domain.zd = Allocate<Real_t>(domNodes) ;

   domain.xdd = Allocate<Real_t>(domNodes)  ; /* accelerations */
   domain.ydd = Allocate<Real_t>(domNodes)  ;
   domain.zdd = Allocate<Real_t>(domNodes)  ;

   domain.fx = Allocate<Real_t>(domNodes) ;  /* forces */
   domain.fy = Allocate<Real_t>(domNodes) ;
   domain.fz = Allocate<Real_t>(domNodes) ;

   domain.nodalMass = Allocate<Real_t>(domNodes) ;  /* mass */

   /* Boundary nodesets */

   domain.symmX = Allocate<Index_t>(edgeNodes*edgeNodes) ;
   domain.symmY = Allocate<Index_t>(edgeNodes*edgeNodes) ;
   domain.symmZ = Allocate<Index_t>(edgeNodes*edgeNodes) ;

   /* Basic Field Initialization */

   for (Index_t i=0; i<domElems; ++i) {
      domain.e[i] = Real_t(0.0) ;
      domain.p[i] = Real_t(0.0) ;
      domain.q[i] = Real_t(0.0) ;
      domain.v[i] = Real_t(1.0) ;
   }

   for (Index_t i=0; i<domNodes; ++i) {
      domain.xd[i] = Real_t(0.0) ;
      domain.yd[i] = Real_t(0.0) ;
      domain.zd[i] = Real_t(0.0) ;
   }

   for (Index_t i=0; i<domNodes; ++i) {
      domain.xdd[i] = Real_t(0.0) ;
      domain.ydd[i] = Real_t(0.0) ;
      domain.zdd[i] = Real_t(0.0) ;
   }

   /* initialize nodal coordinates */

   nidx = 0 ;
   tz  = Real_t(0.) ;
   for (Index_t plane=0; plane<edgeNodes; ++plane) {
      ty = Real_t(0.) ;
      for (Index_t row=0; row<edgeNodes; ++row) {
         tx = Real_t(0.) ;
         for (Index_t col=0; col<edgeNodes; ++col) {
            domain.x[nidx] = tx ;
            domain.y[nidx] = ty ;
            domain.z[nidx] = tz ;
            ++nidx ;
            // tx += ds ; /* may accumulate roundoff... */
            tx = Real_t(1.125)*Real_t(col+1)/Real_t(edgeElems) ;
         }
         // ty += ds ;  /* may accumulate roundoff... */
         ty = Real_t(1.125)*Real_t(row+1)/Real_t(edgeElems) ;
      }
      // tz += ds ;  /* may accumulate roundoff... */
      tz = Real_t(1.125)*Real_t(plane+1)/Real_t(edgeElems) ;
   }


   /* embed hexehedral elements in nodal point lattice */

   nidx = 0 ;
   zidx = 0 ;
   for (Index_t plane=0; plane<edgeElems; ++plane) {
      for (Index_t row=0; row<edgeElems; ++row) {
         for (Index_t col=0; col<edgeElems; ++col) {
            Index_p localNode = &domain.nodelist[8*zidx] ;
            localNode[0] = nidx                                       ;
            localNode[1] = nidx                                   + 1 ;
            localNode[2] = nidx                       + edgeNodes + 1 ;
            localNode[3] = nidx                       + edgeNodes     ;
            localNode[4] = nidx + edgeNodes*edgeNodes                 ;
            localNode[5] = nidx + edgeNodes*edgeNodes             + 1 ;
            localNode[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
            localNode[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
            ++zidx ;
            ++nidx ;
         }
         ++nidx ;
      }
      nidx += edgeNodes ;
   }

   /* initialize material parameters */
   domain.dtfixed = Real_t(lulesh_time_step) ;
   domain.deltatime = Real_t(1.0e-7) ;
   domain.deltatimemultlb = Real_t(1.1) ;
   domain.deltatimemultub = Real_t(1.2) ;
   domain.stoptime  = Real_t(lulesh_stop_time) ;
   domain.dtcourant = Real_t(1.0e+20) ;
   domain.dthydro   = Real_t(1.0e+20) ;
   domain.dtmax     = Real_t(1.0e-2) ;
   domain.time    = Real_t(0.) ;
   domain.cycle   = 0 ;

   domain.e_cut = Real_t(1.0e-7) ;
   domain.p_cut = Real_t(1.0e-7) ;
   domain.q_cut = Real_t(1.0e-7) ;
   domain.u_cut = Real_t(1.0e-7) ;
   domain.v_cut = Real_t(1.0e-10) ;

   domain.hgcoef      = Real_t(3.0) ;
   domain.ss4o3       = Real_t(4.0)/Real_t(3.0) ;

   domain.qstop              =  Real_t(1.0e+12) ;
   domain.monoq_max_slope    =  Real_t(1.0) ;
   domain.monoq_limiter_mult =  Real_t(2.0) ;
   domain.qlc_monoq          = Real_t(0.5) ;
   domain.qqc_monoq          = Real_t(2.0)/Real_t(3.0) ;
   domain.qqc                = Real_t(2.0) ;

   domain.pmin =  Real_t(0.) ;
   domain.emin = Real_t(-1.0e+15) ;

   domain.dvovmax =  Real_t(0.1) ;

   domain.eosvmax =  Real_t(1.0e+9) ;
   domain.eosvmin =  Real_t(1.0e-9) ;

   domain.refdens =  Real_t(1.0) ;

   /* initialize field data */
   for (Index_t i=0; i<domNodes; ++i) {
      domain.nodalMass[i] = 0.0 ;
   }

   for (Index_t i=0; i<domElems; ++i) {
      Real_t x_local[8], y_local[8], z_local[8] ;
      Index_p elemToNode = &domain.nodelist[8*i] ;
      for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      {
        Index_t gnode = elemToNode[lnode];
        x_local[lnode] = domain.x[gnode];
        y_local[lnode] = domain.y[gnode];
        z_local[lnode] = domain.z[gnode];
      }

      // volume calculations
      Real_t volume = CalcElemVolume(x_local, y_local, z_local );
      domain.volo[i] = volume ;
      domain.elemMass[i] = volume ;
      for (Index_t j=0; j<8; ++j) {
         Index_t idx = elemToNode[j] ;
         domain.nodalMass[idx] += volume / Real_t(8.0) ;
      }
   }

   /* deposit energy */
   domain.e[0] = Real_t(3.948746e+7) ;

   /* set up symmetry nodesets */
   nidx = 0 ;
   for (Index_t i=0; i<edgeNodes; ++i) {
      Index_t planeInc = i*edgeNodes*edgeNodes ;
      Index_t rowInc   = i*edgeNodes ;
      for (Index_t j=0; j<edgeNodes; ++j) {
         domain.symmX[nidx] = planeInc + j*edgeNodes ;
         domain.symmY[nidx] = planeInc + j ;
         domain.symmZ[nidx] = rowInc   + j ;
         ++nidx ;
      }
   }

   /* set up elemement connectivity information */
   domain.lxim[0] = 0 ;
   for (Index_t i=1; i<domElems; ++i) {
      domain.lxim[i]   = i-1 ;
      domain.lxip[i-1] = i ;
   }
   domain.lxip[domElems-1] = domElems-1 ;

   for (Index_t i=0; i<edgeElems; ++i) {
      domain.letam[i] = i ; 
      domain.letap[domElems-edgeElems+i] = domElems-edgeElems+i ;
   }
   for (Index_t i=edgeElems; i<domElems; ++i) {
      domain.letam[i] = i-edgeElems ;
      domain.letap[i-edgeElems] = i ;
   }

   for (Index_t i=0; i<edgeElems*edgeElems; ++i) {
      domain.lzetam[i] = i ;
      domain.lzetap[domElems-edgeElems*edgeElems+i] = domElems-edgeElems*edgeElems+i ;
   }
   for (Index_t i=edgeElems*edgeElems; i<domElems; ++i) {
      domain.lzetam[i] = i - edgeElems*edgeElems ;
      domain.lzetap[i-edgeElems*edgeElems] = i ;
   }

   /* set up boundary condition information */
   for (Index_t i=0; i<domElems; ++i) {
      domain.elemBC[i] = 0 ;  /* clear BCs by default */
   }

   /* faces on "external" boundaries will be */
   /* symmetry plane or free surface BCs */
   for (Index_t i=0; i<edgeElems; ++i) {
      Index_t planeInc = i*edgeElems*edgeElems ;
      Index_t rowInc   = i*edgeElems ;
      for (Index_t j=0; j<edgeElems; ++j) {
         domain.elemBC[planeInc+j*edgeElems] |= XI_M_SYMM ;
         domain.elemBC[planeInc+j*edgeElems+edgeElems-1] |= XI_P_FREE ;
         domain.elemBC[planeInc+j] |= ETA_M_SYMM ;
         domain.elemBC[planeInc+j+edgeElems*edgeElems-edgeElems] |= ETA_P_FREE ;
         domain.elemBC[rowInc+j] |= ZETA_M_SYMM ;
         domain.elemBC[rowInc+j+domElems-edgeElems*edgeElems] |= ZETA_P_FREE ;
      }
   }

   /* Create domain IndexSets */

   /* always leave the nodes in a canonical ordering */
   domain.domNodeList = new RAJA::IndexSet() ;
   domain.domNodeList->push_back( RAJA::RangeSegment(0, domNodes) );

   domain.domElemList = new RAJA::IndexSet() ;
   domain.matElemList = new RAJA::IndexSet() ;

   const Index_t xtile = lulesh_xtile ;
   const Index_t ytile = lulesh_ytile ;
   const Index_t ztile = lulesh_ztile ;

   if ( lulesh_tiling_mode == Tiled_LockFree ) {
      printf("Tiled_LockFree ordering not implemented!!! Canonical will be used.\n");
      lulesh_tiling_mode = Canonical;
   }

   switch (lulesh_tiling_mode) {

      case Canonical:
      {
         domain.domElemList->push_back( RAJA::RangeSegment(0, domElems) );

         /* Create a material IndexSet (entire domain same material for now) */
         domain.matElemList->push_back( RAJA::RangeSegment(0, domElems) );
      }
      break ;

      case Tiled_Index:
      {
         for (Index_t zt = 0; zt < ztile; ++zt) {
            for (Index_t yt = 0; yt < ytile; ++yt) {
               for (Index_t xt = 0; xt < xtile; ++xt) {
                  Index_t xbegin =  edgeElems*( xt )/xtile ;
                  Index_t xend   =  edgeElems*(xt+1)/xtile ;
                  Index_t ybegin =  edgeElems*( yt )/ytile ;
                  Index_t yend   =  edgeElems*(yt+1)/ytile ;
                  Index_t zbegin =  edgeElems*( zt )/ztile ;
                  Index_t zend   =  edgeElems*(zt+1)/ztile ;
                  Index_t tileSize = 
                     (xend - xbegin)*(yend-ybegin)*(zend-zbegin) ;
                  Index_t tileIdx[tileSize] ;
                  Index_t idx = 0 ;

                  for (Index_t plane = zbegin; plane<zend; ++plane) {
                     for (Index_t row = ybegin; row<yend; ++row) {
                        for (Index_t col = xbegin; col<xend; ++col) {
                           tileIdx[idx++] = 
                              (plane*edgeElems + row)*edgeElems + col ;
                        }
                     }
                  }
                  domain.domElemList->push_back( RAJA::ListSegment(tileIdx, tileSize) );
                  domain.matElemList->push_back( RAJA::ListSegment(tileIdx, tileSize) );
               }
            }
         }
      }
      break ;

      case Tiled_Order:
      {
         Index_t idx = 0 ;
         Index_t perm[domElems] ;
         Index_t iperm[domElems] ; /* inverse permutation */
         Index_t tileBegin = 0 ;
         for (Index_t zt = 0; zt < ztile; ++zt) {
            for (Index_t yt = 0; yt < ytile; ++yt) {
               for (Index_t xt = 0; xt < xtile; ++xt) {
                  Index_t xbegin =  edgeElems*( xt )/xtile ;
                  Index_t xend   =  edgeElems*(xt+1)/xtile ;
                  Index_t ybegin =  edgeElems*( yt )/ytile ;
                  Index_t yend   =  edgeElems*(yt+1)/ytile ;
                  Index_t zbegin =  edgeElems*( zt )/ztile ;
                  Index_t zend   =  edgeElems*(zt+1)/ztile ;
                  Index_t tileSize = 
                     (xend - xbegin)*(yend-ybegin)*(zend-zbegin) ;

                  for (Index_t plane = zbegin; plane<zend; ++plane) {
                     for (Index_t row = ybegin; row<yend; ++row) {
                        for (Index_t col = xbegin; col<xend; ++col) {
                           perm[idx] = 
                              (plane*edgeElems + row)*edgeElems + col ;
                           iperm[perm[idx]] = idx ;
                           ++idx ;
                        }
                     }
                  }
                  Index_t tileEnd = tileBegin + tileSize ;
                  domain.domElemList->push_back( RAJA::RangeSegment(tileBegin, tileEnd) );
                  domain.matElemList->push_back( RAJA::RangeSegment(tileBegin, tileEnd) );
                  tileBegin = tileEnd ;
               }
            }
         }
         /* permute nodelist connectivity */
         {
            Index_t tmp[8*domElems] ;
            for (Index_t i=0; i<domElems; ++i) {
               for (Index_t j=0; j<8; ++j) {
                  tmp[i*8+j] = domain.nodelist[perm[i]*8+j] ;
               }
            }
            for (Index_t i=0; i<8*domElems; ++i) {
               domain.nodelist[i] = tmp[i] ;
            }
         }
         /* permute volo */
         {
            Real_t tmp[domElems] ;
            for (Index_t i=0; i<domElems; ++i) {
               tmp[i] = domain.volo[perm[i]] ;
            }
            for (Index_t i=0; i<domElems; ++i) {
               domain.volo[i] = tmp[i] ;
            }
         }
         /* permute elemMass */
         {
            Real_t tmp[domElems] ;
            for (Index_t i=0; i<domElems; ++i) {
               tmp[i] = domain.elemMass[perm[i]] ;
            }
            for (Index_t i=0; i<domElems; ++i) {
               domain.elemMass[i] = tmp[i] ;
            }
         }
         /* permute lxim, lxip, letam, letap, lzetam, lzetap */
         {
            Index_t tmp[6*domElems] ;
            for (Index_t i=0; i<domElems; ++i) {
               tmp[i*6+0] = iperm[domain.lxim[perm[i]]] ;
               tmp[i*6+1] = iperm[domain.lxip[perm[i]]] ;
               tmp[i*6+2] = iperm[domain.letam[perm[i]]] ;
               tmp[i*6+3] = iperm[domain.letap[perm[i]]] ;
               tmp[i*6+4] = iperm[domain.lzetam[perm[i]]] ;
               tmp[i*6+5] = iperm[domain.lzetap[perm[i]]] ;
            }
            for (Index_t i=0; i<domElems; ++i) {
               domain.lxim[i] = tmp[i*6+0] ;
               domain.lxip[i] = tmp[i*6+1] ;
               domain.letam[i] = tmp[i*6+2] ;
               domain.letap[i] = tmp[i*6+3] ;
               domain.lzetam[i] = tmp[i*6+4] ;
               domain.lzetap[i] = tmp[i*6+5] ;
            }
         }
         /* permute elemBC */
         {
            Int_t tmp[domElems] ;
            for (Index_t i=0; i<domElems; ++i) {
               tmp[i] = domain.elemBC[perm[i]] ;
            }
            for (Index_t i=0; i<domElems; ++i) {
               domain.elemBC[i] = tmp[i] ;
            }
         }
      }
      break ;

      case Tiled_LockFree:
      {
         // NOT IMPLEMENTED!!!
      }
      break;

      default :
      {
         printf("Unknown index set ordering!!! Left undefined.\n");
      }
   }

   // OMP Hack
   // set up node-centered indexing of elements
   Index_p nodeElemCount = Allocate<Index_t>(domNodes) ;

   for (Index_t i=0; i<domNodes; ++i) {
     nodeElemCount[i] = 0 ;
   }

   for (Index_t i=0; i<domElems; ++i) {
     Index_p nl = &domain.nodelist[8*i] ;
     for (Index_t j=0; j < 8; ++j) {
       ++(nodeElemCount[nl[j]] );
     }
   }

   domain.nodeElemStart = Allocate<Index_t>(domNodes+1) ;

   domain.nodeElemStart[0] = 0;

   for (Index_t i=1; i <= domNodes; ++i) {
     domain.nodeElemStart[i] =
       domain.nodeElemStart[i-1] + nodeElemCount[i-1] ;
   }

   domain.nodeElemCornerList =
      Allocate<Index_t>(domain.nodeElemStart[domNodes]);

   for (Index_t i=0; i < domNodes; ++i) {
     nodeElemCount[i] = 0;
   }

   for (Index_t i=0; i < domElems; ++i) {
     Index_p nl = &domain.nodelist[8*i] ;
     for (Index_t j=0; j < 8; ++j) {
       Index_t m = nl[j];
       Index_t k = i*8 + j ;
       Index_t offset = domain.nodeElemStart[m] + nodeElemCount[m] ;
       domain.nodeElemCornerList[offset] = k;
       ++(nodeElemCount[m]) ;
     }
   }

#ifdef DEBUG_LULESH
   Index_t clSize = domain.nodeElemStart[domNodes] ;
   for (Index_t i=0; i < clSize; ++i) {
     Index_t clv = domain.nodeElemCornerList[i] ;
     if ((clv < 0) || (clv > domElems*8)) {
       fprintf(stderr,
        "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
       exit(-1);
     }
   }
#endif

   Release(&nodeElemCount) ;

   /* Fault Tolerance begins here */

   /* timestep to solution */
   timer_cycle.start("timer_cycle");
   while((domain.time < domain.stoptime) && (domain.cycle < maxIter)) {
      TimeIncrement(&domain) ;
      LagrangeLeapFrog(&domain) ;
      /* problem->commNodes->Transfer(CommNodes::syncposvel) ; */
      if ( show_run_progress ) {
         printf("cycle = %d, time = %e, dt=%e\n",
                domain.cycle,double(domain.time), double(domain.deltatime) ) ;
      }
   }
   timer_cycle.stop("timer_cycle");

   timer_main.stop("timer_main");

   printf("Total Cycle Time (sec) = %Lf\n", timer_cycle.elapsed() );
   printf("Total main Time (sec) = %Lf\n", timer_main.elapsed() );

   return 0 ;
}
