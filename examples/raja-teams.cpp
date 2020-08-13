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

using launch_policy = RAJA::expt::LaunchPolicy<
#if defined(RAJA_ENABLE_OPENMP)
  RAJA::expt::omp_launch_t
#else
  RAJA::expt::seq_launch_t
#endif
#if defined(RAJA_ENABLE_CUDA)
                                         ,
                                         RAJA::expt::cuda_launch_t<false>
#endif
#if defined(RAJA_ENABLE_HIP)
                                         ,
                                         RAJA::expt::hip_launch_t<false>
#endif
                                         >;

using teams1 = RAJA::expt::LoopPolicy<
#if defined(RAJA_ENABLE_OPENMP)
RAJA::omp_parallel_for_exec
#else
RAJA::loop_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                                ,
                                RAJA::cuda_block_y_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                ,
                                RAJA::hip_block_y_direct
#endif
                                >;
using teams_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                ,
                                RAJA::cuda_block_x_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                ,
                                 RAJA::hip_block_x_direct
#endif
                                >;

using teams_xy = RAJA::expt::LoopPolicy<
#if defined(RAJA_ENABLE_OPENMP)
RAJA::omp_parallel_for_exec
#else
RAJA::loop_exec
#endif
#if defined(RAJA_ENABLE_CUDA)
                                 ,
                                 RAJA::expt::cuda_block_xy_nested_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                 ,
                                 RAJA::expt::hip_block_xy_nested_direct
#endif
                                 >;


using threads_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                  ,
                                  RAJA::cuda_thread_y_loop
#endif
#if defined(RAJA_ENABLE_HIP)
                                  ,
                                  RAJA::hip_thread_y_loop
#endif
                                  >;

using threads_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                  ,
                                  RAJA::cuda_thread_x_loop
#endif
#if defined(RAJA_ENABLE_HIP)
                                  ,
                                  RAJA::hip_thread_x_loop
#endif
                                  >;


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  // N is number of blocks in each matrix
  const int NBlocks = 4;

  const int NThreads = THREAD_BLOCK_SZ;
  const int N = NThreads * NBlocks;

  printf("NThreads = %d \n", NThreads);
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


  for (int exec_place = 0; exec_place < (int)RAJA::expt::NUM_PLACES; ++exec_place) {
    RAJA::expt::ExecPlace select_cpu_or_gpu = (RAJA::expt::ExecPlace)exec_place;

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
    constexpr int Nthreads = 5;
    RAJA::expt::launch<launch_policy>(
        select_cpu_or_gpu,
        RAJA::expt::Resources(RAJA::expt::Teams(NTeams), RAJA::expt::Threads(Nthreads)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {
          RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, NTeams), [&](int) {

            //Appveyor has trouble with constexpr.
            //Had to hardcode the number
            TEAM_SHARED int s_A[5];

            RAJA::expt::loop<threads_x>(ctx,RAJA::RangeSegment(0, Nthreads), [&](int i) {
                s_A[i] = i;
            });

            ctx.teamSync();

            RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, Nthreads), [&](int i) {

                printf("s_(%d) = %d \n", i, s_A[Nthreads - 1 - i]);
            });
          });
        });


    //========================
    // Upper triangular pattern
    //========================
    std::cout << "\n Running Upper triangular pattern example...\n";

    const int N_tri = 5;
    RAJA::expt::launch<launch_policy>(
        select_cpu_or_gpu,
        RAJA::expt::Resources(RAJA::expt::Teams(N_tri), RAJA::expt::Threads(N_tri)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {
          RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, N_tri), [&](int i) {
            // do a matrix triangular pattern

              RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(i, N_tri), [&](int j) {
              printf("i=%d, j=%d\n", i, j);
            });  // loop j

          });    // loop i
        });      // kernel


    //=============================
    // Upper triangular pattern #2
    //=============================
    std::cout << "\n Running Upper triangular pattern #2 example...\n";

    const int N_tri2 = 5;

    int* Ddat = memoryManager::allocate<int>(N_tri2 * N_tri2);
    int *Ddat_ptr = Ddat;

#if defined(RAJA_ENABLE_HIP)
   int* Ddat_device = memoryManager::allocate_gpu<int>(N_tri2 * N_tri2);
   hipErrchk(hipMemcpy(Ddat_device, Ddat, N_tri2 * N_tri2 * sizeof(int),
                       hipMemcpyHostToDevice ));
   if(exec_place == RAJA::expt::ExecPlace::DEVICE)
    {
      Ddat_ptr = Ddat_device;
    }
#endif

    RAJA::View<int, RAJA::Layout<2>> D(Ddat_ptr, N_tri2, N_tri2);

    RAJA::expt::launch<launch_policy>( select_cpu_or_gpu,
      RAJA::expt::Resources(RAJA::expt::Teams(N_tri2), RAJA::expt::Threads(N_tri2)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {


        RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, N_tri), [&](int r) {

            RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(r, N_tri), [&](int c) {
              D(r, c) = r * N_tri + c;
            });  // loop j

            RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(r, N_tri), [&](int c) {
              printf("r=%d, c=%d : D=%d\n", r, c, D(r, c));
            });  // loop c
          });    // loop r

        });      // outer lambda

    memoryManager::deallocate(Ddat);

#if defined(RAJA_ENABLE_HIP)
    memoryManager::deallocate_gpu(Ddat_device);
#endif


    //========================
    // Matrix-Matrix Multiplication Example
    //========================
    std::cout << "\n Running Matrix-Matrix Multiplication example...\n";

    double *A_ptr=A;
    double *B_ptr=B;
    double *C_ptr=C;
#if defined(RAJA_ENABLE_HIP)
    double* A_device = memoryManager::allocate_gpu<double>(N*N);
    double* B_device = memoryManager::allocate_gpu<double>(N*N);
    double* C_device = memoryManager::allocate_gpu<double>(N*N);
    hipErrchk(hipMemcpy( A_device, A, N * N * sizeof(double), hipMemcpyHostToDevice ));
    hipErrchk(hipMemcpy( B_device, B, N * N * sizeof(double), hipMemcpyHostToDevice ));
    hipErrchk(hipMemcpy( C_device, C, N * N * sizeof(double), hipMemcpyHostToDevice ));

    if(exec_place == RAJA::expt::ExecPlace::DEVICE)
    {
      A_ptr=A_device;
      B_ptr=B_device;
      C_ptr=C_device;
    }
#endif

    // Set up Teams/Threads

    printf("select_cpu_or_gpu %d \n", select_cpu_or_gpu);
    RAJA::expt::launch<launch_policy>(select_cpu_or_gpu,
      RAJA::expt::Resources(RAJA::expt::Teams(NBlocks, NBlocks),
                            RAJA::expt::Threads(NThreads, NThreads)),
       [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::RangeSegment TeamRange(0, NBlocks);
          RAJA::RangeSegment ThreadRange(0, NThreads);
          //
          // Loop over teams
          //
          // RAJA::loop<teams1>(ctx, RAJA::RangeSegment(0, NBlocks), [&](int by)
          // { RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, NBlocks), [&](int
          // bx) {
          RAJA::expt::loop<teams_xy>(ctx, TeamRange, TeamRange, [&](int bx, int by) {

            TEAM_SHARED double As[THREAD_BLOCK_SZ][THREAD_BLOCK_SZ];
            TEAM_SHARED double Bs[THREAD_BLOCK_SZ][THREAD_BLOCK_SZ];
            TEAM_SHARED double Cs[THREAD_BLOCK_SZ][THREAD_BLOCK_SZ];

            // Team parallel loop
            RAJA::expt::loop<threads_y>(ctx, ThreadRange, [&](int ty) {
                RAJA::expt::loop<threads_x>(ctx, ThreadRange, [&](int tx) {

                    Cs[ty][tx] = 0.0;

                  });
                });

            // Slide across matrix
            for (int m = 0; m < (N / NThreads); ++m) {

              RAJA::expt::loop<threads_y>(ctx, ThreadRange, [&](int ty) {
                  RAJA::expt::loop<threads_x>(ctx, ThreadRange, [&](int tx) {

                          const int row = by * NThreads + ty;  // Matrix row index
                          const int col = bx * NThreads + tx;  // Matrix column index

                          As[ty][tx] = A_ptr[row * N + m * NThreads + tx];
                          Bs[ty][tx] = B_ptr[(m * NThreads + ty) * N + col];

                      });
                });

              ctx.teamSync();

              RAJA::expt::loop<threads_y>(ctx, ThreadRange, [&](int ty) {
                  RAJA::expt::loop<threads_x>(ctx, ThreadRange, [&](int tx) {

                      for (int e = 0; e < NThreads; ++e) {

                        Cs[ty][tx] += As[ty][e] * Bs[e][tx];

                      }
                    });
                });

              ctx.teamSync();

            }  // slide across matrix


            RAJA::expt::loop<threads_y>(ctx, ThreadRange, [&](int ty) {
                RAJA::expt::loop<threads_x>(ctx, ThreadRange, [&](int tx) {

                        const int row = by * NThreads + ty;  // Matrix row index
                        const int col = bx * NThreads + tx;  // Matrix column index
                        C_ptr[col + N * row] = Cs[ty][tx];

                  });
                });
            //});

          });

        });  // kernel

#if defined(RAJA_ENABLE_HIP)
    if(exec_place == RAJA::expt::ExecPlace::DEVICE)
    {
      hipErrchk(hipMemcpy(C, C_device, N * N * sizeof(double), hipMemcpyDeviceToHost ));
    }
#endif

    checkResult<double>(C_ptr, N);
    printf("\n");

  }


}//Main


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
