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
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
#endif

#if defined(__CUDA_ARCH__)
#define TEAM_SHARED __shared__
#define TEAM_SYNC() __syncthreads()
#else
#define TEAM_SHARED
#define TEAM_SYNC()
#endif

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

  const int row = by * CUDA_BLOCK_SIZE + ty;  // Matrix row index
  const int col = bx * CUDA_BLOCK_SIZE + tx;  // Matrix column index

  // Shared memory used to store Asub and Bsub respectively
  __shared__ double As[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
  __shared__ double Bs[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (N / CUDA_BLOCK_SIZE); ++m) {

    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[ty][tx] = A[row * N + m * CUDA_BLOCK_SIZE + tx];
    Bs[ty][tx] = B[(m * CUDA_BLOCK_SIZE + ty) * N + col];

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();

    // Multiply Asub and Bsub together
    for (int e = 0; e < CUDA_BLOCK_SIZE; ++e)
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


using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t, RAJA::cuda_launch_t<false>>;


#ifdef RAJA_ENABLE_CUDA
using outer0 = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::cuda_block_x_direct>;
using outer1 = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::cuda_block_y_direct>;
#else
using outer0 = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
using outer1 = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
#endif


#ifdef RAJA_ENABLE_CUDA
using team0 = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::cuda_thread_x_loop>;
using team1 = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::cuda_thread_y_loop>;
#else
using team0 = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
using team1 = RAJA::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
#endif


int main()
{

  // N is number of blocks in each matrix
  const int NBlocks = 4;
#ifdef RAJA_ENABLE_CUDA
  const int NThreads = CUDA_BLOCK_SIZE;
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

  std::cout << "\n Running RAJA-Teams V2-version of matrix multiplication...\n";
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
    // Upper triangular pattern
    //========================
    const int N_tri = 5;
    RAJA::launch<launch_policy>(
        select_cpu_or_gpu,
        RAJA::ResourceList{
            RAJA::Resources(RAJA::Threads(N_tri)),
            RAJA::Resources(RAJA::Teams(N_tri), RAJA::Threads(N_tri))},
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<outer0>(ctx, RAJA::RangeSegment(0, N_tri), [=](int i) {
            // do a matrix triangular pattern
            RAJA::loop<team0>(ctx, RAJA::RangeSegment(i, N_tri), [=](int j) {
              printf("i=%d, j=%d\n", i, j);
            });  // loop j
          });    // loop i
        });      // kernel

    //========================
    // Matrix-Matrix Multiplication Example
    //========================
    // Set up Teams/Threads

    RAJA::launch<launch_policy>(select_cpu_or_gpu,
          RAJA::ResourceList{
            RAJA::Resources(RAJA::Threads(NBlocks, NBlocks)),
            RAJA::Resources(RAJA::Teams(NBlocks, NBlocks), RAJA::Threads(NThreads, NThreads))},
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          //
          // Loop over teams
          //
          RAJA::loop<outer1>(ctx, RAJA::RangeSegment(0, NBlocks), [&](int by) {
            RAJA::loop<outer0>(ctx, RAJA::RangeSegment(0, NBlocks), [&](int bx) {

                  TEAM_SHARED double As[NThreads][NThreads];
                  TEAM_SHARED double Bs[NThreads][NThreads];
                  TEAM_SHARED double Cs[NThreads][NThreads];

                  // Team parallel loop
                  RAJA::loop<team1>(ctx, RAJA::RangeSegment(0, NThreads), [&](int ty) {
                      RAJA::loop<team0>(ctx, RAJA::RangeSegment(0, NThreads), [&](int tx) {

                          Cs[ty][tx] = 0.0;

                        });
                      });

                  // Slide across matrix
                  for (int m = 0; m < (N / NThreads); ++m) {

                    RAJA::loop<team1>(ctx, RAJA::RangeSegment(0, NThreads), [&](int ty) {
                        RAJA::loop<team0>(ctx, RAJA::RangeSegment(0, NThreads), [&](int tx) {

                                const int row = by * NThreads + ty;  // Matrix row index
                                const int col = bx * NThreads + tx;  // Matrix column index

                                As[ty][tx] = A[row * N + m * NThreads + tx];
                                Bs[ty][tx] = B[(m * NThreads + ty) * N + col];
                              });
                        });

                    TEAM_SYNC();

                    RAJA::loop<team1>(ctx, RAJA::RangeSegment(0, NThreads), [&](int ty) {
                        RAJA::loop<team0>(ctx, RAJA::RangeSegment(0, NThreads), [&](int tx) {

                            for (int e = 0; e < NThreads; ++e) {

                              Cs[ty][tx] += As[ty][e] * Bs[e][tx];

                            }
                          });
                      });
                    TEAM_SYNC();
                  }  // slide across matrix


                  RAJA::loop<team1>(ctx, RAJA::RangeSegment(0, NThreads), [&](int ty) {
                      RAJA::loop<team0>(ctx, RAJA::RangeSegment(0, NThreads), [&](int tx) {

                              const int row = by * NThreads + ty;  // Matrix row index

                              const int col = bx * NThreads + tx;  // Matrix column index

                              C[col + N * row] = Cs[ty][tx];
                        });
                    });
              });
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
