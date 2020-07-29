//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

/*
 *  RAJA Teams Example:
 *  Matrix-matrix multiplication with shared memory
 */

/*
  Define number of threads in x and y dimensions of a CUDA thread block
*/
#define THREAD_BLOCK_SZ 16


//
// Define dimensionality of matrices.
//
const int DIM = 2;

//
// Define macros to simplify row-col indexing (non-RAJA implementations only)
//
// _matmult_macros_start
#define A(r, c) A[c + N * r]
#define B(r, c) B[c + N * r]
#define C(r, c) C[c + N * r]
// _matmult_macros_end

/*
  Define CUDA matrix multiplication kernel for comparison to RAJA version
  Adapted from CUDA programming guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
*/
#if defined(RAJA_ENABLE_CUDA)
__global__ void matMultKernel(int N, double *C, double *A, double *B)
{
  // Block row and column
  const int by = blockIdx.y;
  const int bx = blockIdx.x;

  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  double Cvalue(0.0);

  // Thread row and column within local Csub
  const int ty = threadIdx.y;  // local row
  const int tx = threadIdx.x;  // local column

  const int row = by * THREAD_BLOCK_SZ + ty;  // Matrix row index
  const int col = bx * THREAD_BLOCK_SZ + tx;  // Matrix column index

  // Shared memory used to store Asub and Bsub respectively
  __shared__ double As[THREAD_BLOCK_SZ][THREAD_BLOCK_SZ];
  __shared__ double Bs[THREAD_BLOCK_SZ][THREAD_BLOCK_SZ];

  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (N / THREAD_BLOCK_SZ); ++m) {

    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[ty][tx] = A[row * N + m * THREAD_BLOCK_SZ + tx];
    Bs[ty][tx] = B[(m * THREAD_BLOCK_SZ + ty) * N + col];

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();

    // Multiply Asub and Bsub together
    for (int e = 0; e < THREAD_BLOCK_SZ; ++e)
      Cvalue += As[ty][e] * Bs[e][tx];

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write Csub to device memory
  // Each thread writes one element
  C[col + N * row] = Cvalue;
}
#endif

//
// Functions for checking results
//
template <typename T>
void checkResult(T *C, int N);

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);

//
// Functions for printing results
//
template <typename T>
void printResult(T *C, int N);

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);

using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t
#if defined(RAJA_ENABLE_OPENMP)
                                         ,
                                         RAJA::omp_launch_t
#endif
#if defined(RAJA_ENABLE_CUDA)
                                         ,
                                         RAJA::cuda_launch_t<false>
#endif
                                         >;

using teams1 = RAJA::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_OPENMP)
                                ,
                                RAJA::omp_parallel_for_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                                ,
                                RAJA::cuda_block_y_direct
#endif
                                >;
using teams0 = RAJA::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_OPENMP)
                                ,
                                RAJA::loop_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                                ,
                                RAJA::cuda_block_x_direct
#endif
                                >;


using teams01 = RAJA::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_OPENMP)

                                 ,
                                 RAJA::omp_parallel_for_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                                 ,
                                 RAJA::cuda_block_xyz_direct<2>
#endif
                                 >;


using threads1 = RAJA::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_OPENMP)
                                  ,
                                  RAJA::loop_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                                  ,
                                  RAJA::cuda_thread_y_loop
#endif
                                  >;

using threads0 = RAJA::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_OPENMP)
                                  ,
                                  RAJA::loop_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                                  ,
                                  RAJA::cuda_thread_x_loop
#endif
                                  >;


int main()
{

  // N is number of blocks in each matrix
  const int NBlocks = 4;
#ifdef RAJA_ENABLE_CUDA
  const int NThreads = THREAD_BLOCK_SZ;
  const int N = NThreads * NBlocks;
#else
  const int NThreads = 1;
  const int N = NThreads * NBlocks;
#endif

  //
  // Allocate and initialize matrix data.
  //
  double *A = memoryManager::allocate<double>(N * N);
  double *B = memoryManager::allocate<double>(N * N);
  double *C = memoryManager::allocate<double>(N * N);

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A(row, col) = row;
      B(row, col) = col;
    }
  }

  std::cout << "\n Running RAJA-Teams examples...\n";
  std::cout << "  N = " << N << std::endl;

  for (int exec_place = 0; exec_place < (int)RAJA::NUM_PLACES; ++exec_place) {
    RAJA::ExecPlace select_cpu_or_gpu = (RAJA::ExecPlace)exec_place;
    // auto select_cpu_or_gpu = RAJA::HOST;
    // auto select_cpu_or_gpu = RAJA::DEVICE;

    /*
     * launch just starts a "kernel" it's doesn't provide any looping.
     *
     * The first argument determines which policy should be executed,
     *
     * The second argument is the number of teams+threads needed for each of the
     * policies.
     *
     * Third argument is the lambda for the policy.
     *
     *
     * The lambda takes a "resource" object, which has the teams+threads and
     * policy selection information.
     */

    //========================
    // Simple Shared memory pattern
    //========================
    std::cout << "\n Running Simple Shared memory pattern example...\n";

    const int NTeams = 2;
    const int Nthreads = 5;
    RAJA::launch<launch_policy>(
        select_cpu_or_gpu,
        RAJA::Resources(RAJA::Teams(NTeams), RAJA::Threads(Nthreads)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams0>(ctx, RAJA::RangeSegment(0, NTeams), [&](int) {

            TEAM_SHARED int s_A[Nthreads];

            RAJA::loop<threads0>(ctx,RAJA::RangeSegment(0, Nthreads), [&](int i) {
                s_A[i] = i; 
            });

            ctx.teamSync();

            RAJA::loop<threads0>(ctx, RAJA::RangeSegment(0, Nthreads), [&](int i) {
                
                printf("s_(%d) = %d \n", i, s_A[Nthreads - 1 - i]);
            });
          });
        });


    //========================
    // Upper triangular pattern
    //========================
    std::cout << "\n Running Upper triangular pattern example...\n";

    const int N_tri = 5;
    using Teams_t = RAJA::TeamExclusive<5>;
    RAJA::launch<launch_policy>(
        select_cpu_or_gpu,
        RAJA::Resources(RAJA::Teams(N_tri), RAJA::Threads(N_tri)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams0>(ctx, RAJA::RangeSegment(0, N_tri), [=](int i) {
            // do a matrix triangular pattern

            Teams_t::ExclusiveMem<int, 1> Val;

            RAJA::loop<threads0>(ctx, RAJA::RangeSegment(i, N_tri), [=](int j) {
              Val(0, j) = j;
            });  // loop j

            RAJA::loop<threads0>(ctx, RAJA::RangeSegment(i, N_tri), [=](int j) {
              printf("i=%d, j=%d\n", i, j);
              printf("Val(0, j)=%d\n", Val(0, j));
            });  // loop j
          });    // loop i
        });      // kernel

   
    //=============================
    // Upper triangular pattern #2
    //=============================
    std::cout << "\n Running Upper triangular pattern #2 example...\n";

    const int N_tri2 = 5;

    int* Ddat = memoryManager::allocate<int>(N_tri2 * N_tri2);
    RAJA::View<int, RAJA::Layout<2>> D(Ddat, N_tri2, N_tri2); 

    for (int r = 0; r < N_tri2; ++r) {
      for (int c = 0; c < N_tri2; ++c) {
        D(r, c) = 0;
      }
    }

    RAJA::launch<launch_policy>( select_cpu_or_gpu,
      RAJA::Resources(RAJA::Teams(N_tri2), RAJA::Threads(N_tri2)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<teams0>(ctx, RAJA::RangeSegment(0, N_tri), [=](int r) {

            RAJA::loop<threads0>(ctx, RAJA::RangeSegment(r, N_tri), [=](int c) {
              D(r, c) = r * N_tri + c;
            });  // loop j

            RAJA::loop<threads0>(ctx, RAJA::RangeSegment(r, N_tri), [=](int c) {
              printf("r=%d, c=%d : D=%d\n", r, c, D(r, c));
            });  // loop c
          });    // loop r
        });      // outer lambda

    memoryManager::deallocate(Ddat);

    //========================
    // Matrix-Matrix Multiplication Example
    //========================
    std::cout << "\n Running Matrix-Matrix Multiplication example...\n";

    // Set up Teams/Threads

    RAJA::RangeSegment TeamRange(0, NBlocks);

    printf("select_cpu_or_gpu %d \n", select_cpu_or_gpu);
    RAJA::launch<launch_policy>(select_cpu_or_gpu,
        RAJA::Resources(RAJA::Teams(NBlocks, NBlocks),
                        RAJA::Threads(NThreads, NThreads)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          //
          // Loop over teams
          //
          // RAJA::loop<teams1>(ctx, RAJA::RangeSegment(0, NBlocks), [&](int by)
          // { RAJA::loop<teams0>(ctx, RAJA::RangeSegment(0, NBlocks), [&](int
          // bx) {

          RAJA::loop<teams01>(ctx, TeamRange, TeamRange, [&](int bx, int by) {

            TEAM_SHARED double As[THREAD_BLOCK_SZ][THREAD_BLOCK_SZ];
            TEAM_SHARED double Bs[THREAD_BLOCK_SZ][THREAD_BLOCK_SZ];
            // TEAM_SHARED double Cs[THREAD_BLOCK_SZ][THREAD_BLOCK_SZ];
            RAJA::TeamSharedArray<double, THREAD_BLOCK_SZ, THREAD_BLOCK_SZ> Cs;//Not correct???

            // printf("size of  Cs %d As %d \n", sizeof(Cs), sizeof(As));

            // Team parallel loop
            RAJA::loop<threads1>(ctx, RAJA::RangeSegment(0, NThreads), [&](int ty) {                                 
                RAJA::loop<threads0>(ctx, RAJA::RangeSegment(0, NThreads), [&](int tx) { 

                    Cs(ty, tx) = 0.0;

                  });                                                                          
                });

            // Slide across matrix
            for (int m = 0; m < (N / NThreads); ++m) {

              RAJA::loop<threads1>(ctx, RAJA::RangeSegment(0, NThreads), [&](int ty) {                                   
                    RAJA::loop<threads0>(ctx, RAJA::RangeSegment(0, NThreads), [&](int tx) {
                                         
                          const int row = by * NThreads + ty;  // Matrix row index                            
                          const int col = bx * NThreads + tx;  // Matrix column index

                          As[ty][tx] = A[row * N + m * NThreads + tx];
                          Bs[ty][tx] = B[(m * NThreads + ty) * N + col];

                      });
                });

              ctx.teamSync();

              RAJA::loop<threads1>(ctx, RAJA::RangeSegment(0, NThreads), [&](int ty) {                  
                  RAJA::loop<threads0>(ctx, RAJA::RangeSegment(0, NThreads), [&](int tx) {
                                       
                      for (int e = 0; e < NThreads; ++e) {
                        
                        Cs(ty, tx) += As[ty][e] * Bs[e][tx];
                          
                      }
                    });
                });

              ctx.teamSync();
            }  // slide across matrix


            RAJA::loop<threads1>(ctx, RAJA::RangeSegment(0, NThreads), [&](int ty) {                                 
                RAJA::loop<threads0>(ctx, RAJA::RangeSegment(0, NThreads), [&](int tx) {
                                     
                        const int row = by * NThreads + ty;  // Matrix row index
                        const int col = bx * NThreads + tx;  // Matrix column index
                        C[col + N * row] = Cs(ty, tx);

                  });
                });
            //});
          });
        });  // kernel

    checkResult<double>(C, N);
    printf("\n");
  }
}


//
// Functions to check result and report P/F.
//
template <typename T>
void checkResult(T *C, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if (std::abs(C(row, col) - row * col * N) > 10e-12) {
        match = false;
      }
    }
  }
  if (match) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if (std::abs(Cview(row, col) - row * col * N) > 10e-12) {
        match = false;
      }
    }
  }
  if (match) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Functions to print result.
//
template <typename T>
void printResult(T *C, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = " << C(row, col)
                << std::endl;
    }
  }
  std::cout << std::endl;
}

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = " << Cview(row, col)
                << std::endl;
    }
  }
  std::cout << std::endl;
}
