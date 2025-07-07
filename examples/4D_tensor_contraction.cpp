//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"
#include "memoryManager.hpp"


/*
 * RAJA Launch Example: Upper Triangular Pattern + Shared Memory
 *
 * Launch introduces hierarchical parallelism through the concept of
 * teams and threads.  Computation is executed in a pre-defined grid
 * composed of threads and grouped into teams. The teams model enables
 * developers to express parallelism through loops over teams, and inner loops
 * over threads. Team loops are executed in parallel and
 * threads within a team should be treated as sub-parallel regions.
 *
 * Team shared memory is allocated between team and thread loops.
 * Memory allocated within thread loops are thread private.
 * The example below demonstrates composing an upper triangular
 * loop pattern, and using shared memory.
 *
 */

/*
 * Define host/device launch policies
 */
using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
  
using teams_x = RAJA::LoopPolicy<RAJA::seq_exec>;
                                 
using threads_x = RAJA::LoopPolicy<RAJA::seq_exec>;


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  constexpr int TotalMats = 100;
  
  constexpr int I = 2;
  constexpr int J = 2;
  constexpr int L = 2;
  constexpr int K = 2;
  constexpr int M = 2;
  constexpr int N = 2;
  constexpr int O = 2;    
  
  double *Aptr = memoryManager::allocate<double>(TotalMats * I * J * K * L);
  double *Bptr = memoryManager::allocate<double>(TotalMats * L * M * N * O);
  double *Cptr = memoryManager::allocate<double>(TotalMats * I * J * K * M * N * O);

  auto A = RAJA::make_permuted_view<RAJA::layout_right>(Aptr, TotalMats, I, J, K, L);
  auto B = RAJA::make_permuted_view<RAJA::layout_right>(Bptr, TotalMats, L, M, N, O);
  auto C = RAJA::make_permuted_view<RAJA::layout_right>(Cptr, TotalMats, I, J, K, N, O);

  // Initialize A and B with some values
  for(int mat = 0; mat < TotalMats; ++mat) {

    for (int i = 0; i < I; i++) {
      for (int j = 0; j < J; j++) {
	for (int k = 0; k < K; k++) {
	  for (int l = 0; l < L; l++) {
	    A(mat, i, j, k, l) = 1.0;
	  }
	}
      }
    }  
    
    for (int l = 0; l < L; l++) { 
      for (int m = 0; m < M; m++) {
	for (int n = 0; n < N; n++) {
	  for (int o = 0; o < O; o++) {
	    B(mat, l, m, n, o) = 1.0;
	  }
	}
      }
    }
    
  }

  
  
#if 0
    RAJA::launch<launch_policy>
      (select_cpu_or_gpu,
       RAJA::LaunchParams(RAJA::Teams(N_tri), RAJA::Threads<4>(1,2,3,4)),
       //RAJA::LaunchParams(RAJA::Teams(N_tri), RAJA::Threads(N_tri)),
       [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
         printf("in kernel \n");
         RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, N_tri), [&](int r) {

           // Array shared within threads of the same team
           RAJA_TEAM_SHARED int s_A[1];

           RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, 1), [&](int c) {
              s_A[c] = r;
           });  // loop c

           ctx.teamSync();

           RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(r, N_tri), [&](int c) {
               D(r, c) = r * N_tri + c;
               printf("r=%d, c=%d : D=%d : s_A = %d \n", r, c, D(r, c), s_A[0]);
           });  // loop c

         });  // loop r

       });  // outer lambda
#endif    

}  // Main
