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

// Define problem setup
constexpr int TotalMats = 100;

constexpr int I = 2;
constexpr int J = 2;
constexpr int L = 2;
constexpr int K = 2;
constexpr int M = 2;
constexpr int N = 2;
constexpr int O = 2;

/*
 * Define host/device launch policies
 */
const bool async = false;
using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t
#if defined(RAJA_ENABLE_HIP)
                                         ,RAJA::hip_launch_t<async>
#endif
                                         >;

using teams = RAJA::LoopPolicy<RAJA::seq_exec
#if defined(RAJA_ENABLE_HIP)
                               ,RAJA::hip_block_x_direct
#endif
                               >;

using loop_0 = RAJA::LoopPolicy<RAJA::seq_exec
#if defined(RAJA_ENABLE_HIP)
                                ,RAJA::seq_exec
#endif
                                >;
using loop_1 = RAJA::LoopPolicy<RAJA::seq_exec
#if defined(RAJA_ENABLE_HIP)
                                ,RAJA::seq_exec
#endif
                                >;
using loop_2 = RAJA::LoopPolicy<RAJA::seq_exec
#if defined(RAJA_ENABLE_HIP)
                                ,RAJA::seq_exec
#endif
                                >;
using loop_3 = RAJA::LoopPolicy<RAJA::seq_exec
#if defined(RAJA_ENABLE_HIP)
                                ,RAJA::seq_exec
#endif
                                >;

using loop_4 = RAJA::LoopPolicy<RAJA::seq_exec
#if defined(RAJA_ENABLE_HIP)
                                ,RAJA::seq_exec
#endif
                                >;
using loop_5 = RAJA::LoopPolicy<RAJA::seq_exec
#if defined(RAJA_ENABLE_HIP)
                                ,RAJA::seq_exec
#endif
                                >;


template<typename AVIEW, typename BVIEW, typename CVIEW>
void tensor_contraction(AVIEW A, BVIEW B, CVIEW C, RAJA::ExecPlace platform)
{

  RAJA::launch<launch_policy>
    (RAJA::LaunchParams(RAJA::Teams(TotalMats), RAJA::Threads<6>(I, J, K, M, N, O)),
       [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

         RAJA::loop<teams>(ctx, RAJA::RangeSegment(0, TotalMats), [&](int r) {

           RAJA::loop<loop_0>(ctx, RAJA::RangeSegment(0, I), [&](int i) {
             RAJA::loop<loop_1>(ctx, RAJA::RangeSegment(0, J), [&](int j) {
               RAJA::loop<loop_2>(ctx, RAJA::RangeSegment(0, K), [&](int k) {
                 RAJA::loop<loop_3>(ctx, RAJA::RangeSegment(0, M), [&](int m) {
                   RAJA::loop<loop_4>(ctx, RAJA::RangeSegment(0, N), [&](int n) {
                     RAJA::loop<loop_5>(ctx, RAJA::RangeSegment(0, O), [&](int o) {

                       double dot = 0.0;
                       for(int l = 0; l < L; ++l) {
                         dot += A(r, i,j,k,l) * B(r, l,m,n,o);
                       }
                       C(r, i,j,k,m,n,o) = dot;

                      });
                    });
                  });
                });
              });
            });
          });

       });  // outer lambda


}



int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{


  double *Aptr = memoryManager::allocate<double>(TotalMats * I * J * K * L);
  double *Bptr = memoryManager::allocate<double>(TotalMats * L * M * N * O);
  double *Cptr = memoryManager::allocate<double>(TotalMats * I * J * K * M * N * O);

  double *test_Cptr = memoryManager::allocate<double>(TotalMats * I * J * K * M * N * O);

  auto A = RAJA::make_permuted_view<RAJA::layout_right>(Aptr, TotalMats, I, J, K, L);
  auto B = RAJA::make_permuted_view<RAJA::layout_right>(Bptr, TotalMats, L, M, N, O);
  auto C = RAJA::make_permuted_view<RAJA::layout_right>(Cptr, TotalMats, I, J, K, M, N, O);
  auto test_C = RAJA::make_permuted_view<RAJA::layout_right>(test_Cptr, TotalMats, I, J, K, M, N, O);

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


}  // Main
