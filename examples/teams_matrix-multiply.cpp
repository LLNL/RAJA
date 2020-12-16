//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Matrix Multiplication Examples using RAJA Teams
 *
 *  Example computes the product of two square matrices and introduces
 *  RAJA nested loop capabilities via a sequence of implementations.
 *
 *  RAJA features shown:
 *    - Index range segment
 *    - View abstraction
 *    - Basic usage of 'RAJA Teams' abstractions for nested loops
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
#define GPU_TB_SZ 16
#endif

/*
 * Define host/device launch policies
 */
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


using loop_policy = RAJA::loop_exec;

#if defined(RAJA_ENABLE_CUDA)
using gpu_block_x_policy = RAJA::cuda_block_x_direct;
using gpu_block_y_policy = RAJA::cuda_block_y_direct;
using gpu_thread_x_policy = RAJA::cuda_thread_x_loop;
using gpu_thread_y_policy = RAJA::cuda_thread_y_loop;
using gpu_global_thread_x_policy = RAJA::expt::cuda_global_thread_x;
using gpu_global_thread_y_policy = RAJA::expt::cuda_global_thread_y;
#endif

#if defined(RAJA_ENABLE_HIP)
using gpu_block_x_policy = RAJA::hip_block_x_direct;
using gpu_block_y_policy = RAJA::hip_block_y_direct;
using gpu_thread_x_policy = RAJA::hip_thread_x_loop;
using gpu_thread_y_policy = RAJA::hip_thread_y_loop;
using gpu_global_thread_x_policy = RAJA::expt::hip_global_thread_x;
using gpu_global_thread_y_policy = RAJA::expt::hip_global_thread_y;
#endif

/*
  Define RAJA Team policies
*/
using teams_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_block_x_policy
#endif
                                       >;

using teams_y = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_block_y_policy
#endif
                                       >;

using threads_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_thread_x_policy
#endif
                                       >;

using threads_y = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_thread_y_policy
#endif
                                       >;

using global_thread_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_global_thread_x_policy
#endif
                                       >;

using global_thread_y = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_global_thread_y_policy
#endif
                                       >;

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
*/
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
__global__ void matMultKernel(int N, double* C, double* A, double* B)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ( row < N && col < N ) {
    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += A(row, k) * B(k, col);
    }

    C(row, col) = dot;
  }
}

__global__ void sharedMatMultKernel(int N, double* C, double* A, double* B)
{

  int Row = blockIdx.y*GPU_TB_SZ + threadIdx.y;
  int Col = blockIdx.x*GPU_TB_SZ + threadIdx.x;

  __shared__ float As[GPU_TB_SZ][GPU_TB_SZ];
  __shared__ float Bs[GPU_TB_SZ][GPU_TB_SZ];
  __shared__ float Cs[GPU_TB_SZ][GPU_TB_SZ];

  for (int k = 0; k < (GPU_TB_SZ + N - 1)/GPU_TB_SZ; k++) {

    Cs[threadIdx.y][threadIdx.x] = 0.0;

    if (k*GPU_TB_SZ + threadIdx.x < N && Row < N)
      As[threadIdx.y][threadIdx.x] = A[Row*N + k*GPU_TB_SZ + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*GPU_TB_SZ + threadIdx.y < N && Col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(k*GPU_TB_SZ + threadIdx.y)*N + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < GPU_TB_SZ; ++n)
      Cs[threadIdx.y][threadIdx.x] += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }

  if (Row < N && Col < N)
    C[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
      (blockIdx.x * blockDim.x)+ threadIdx.x] = Cs[threadIdx.y][threadIdx.x];
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


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA matrix multiplication example...\n";

//
// Define num rows/cols in matrix
//
  const int N = 1000;
  const int NTeams = (N - 1)/GPU_TB_SZ + 1;

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

//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of matrix multiplication...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_cstyle_start
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += A(row, k) * B(k, col);
      }
      C(row, col) = dot;

    }
  }
  // _matmult_cstyle_end

  checkResult<double>(C, N);
//printResult<double>(C, N);


//----------------------------------------------------------------------------//

//
// We define RAJA range segments to define the ranges of
// row, column, and dot-product loops for RAJA variants
//
  // _matmult_ranges_start
  RAJA::RangeSegment row_range(0, N);
  RAJA::RangeSegment col_range(0, N);
  RAJA::RangeSegment dot_range(0, N);
  // _matmult_ranges_end

//----------------------------------------------------------------------------//

//
// For the RAJA implementations of matrix multiplication, we
// use RAJA 'View' objects to access the matrix data. A RAJA view
// holds a pointer to a data array and enables multi-dimensional indexing
// into that data, similar to the macros we defined above.
//
  // _matmult_views_start
  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, N);
  // _matmult_views_end

//----------------------------------------------------------------------------//

//
// In the next few examples, we show ways that we can use RAJA::forall
// statements for the matrix multiplication kernel. This usage is not
// recommended for performance reasons. Specifically, it limits the amount
// of parallelism that can be exposed to less than is possible. We show
// this usage here, to make this point clear. Later in this file, we
// introduce RAJA nested loop abstractions and show that we can extract all
// available parallelism.
//
//
// In the first RAJA implementation, we replace the outer 'row' loop
// with a RAJA::forall statement. The lambda expression contains the
// inner loops.
//

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential mat-mult (RAJA-row)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_outerforall_start
  RAJA::forall<RAJA::loop_exec>( row_range, [=](int row) {

    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
      Cview(row, col) = dot;

    }

  });
  // _matmult_outerforall_end

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

//
// Next, we replace the outer 'row' loop and the inner 'col' loop
// with RAJA::forall statements. This will also work with parallel
// execution policies, such as OpenMP and CUDA, with caveats and
// restrictions.
//
// However, nesting RAJA::forall calls like this is not recommended as
// it limits the ability to expose parallelism and flexibility for
// implementation alternatives.
//

  std::cout << "\n Running sequential mat-mult (RAJA-row, RAJA-col)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_nestedforall_start
  RAJA::forall<RAJA::loop_exec>( row_range, [=](int row) {

    RAJA::forall<RAJA::loop_exec>( col_range, [=](int col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
      Cview(row, col) = dot;

    });

  });
  // _matmult_nestedforall_end

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

//
// Next, we use a RAJA::launch method to execute the kernel. These examples,
// illustrate the basic interface and mechanics.
//
// This is different than RAJA::forall and so a few points of exmplanation
// are in order:
//

  std::cout << "\n Running sequential mat-mult (RAJA-nested)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_basickernel_start
  RAJA::expt::launch<launch_policy>(RAJA::expt::HOST,
   RAJA::expt::Resources(RAJA::expt::Teams(NTeams,NTeams),
                         RAJA::expt::Threads(GPU_TB_SZ,GPU_TB_SZ)),
       [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

   RAJA::expt::loop<global_thread_y>(ctx, col_range, [&] (int col) {
       RAJA::expt::loop<global_thread_x>(ctx, row_range, [&] (int row) {

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          Cview(row, col) = dot;
      });
    });

  });
  // _matmult_basickernel_end

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - omp outer)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  using omp_col_policy0 = RAJA::expt::LoopPolicy<RAJA::omp_parallel_for_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                                 ,gpu_global_thread_y_policy
#endif
    >;

  using omp_row_policy0 = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                                 ,gpu_global_thread_x_policy
#endif
    >;


  RAJA::expt::launch<launch_policy>(RAJA::expt::HOST,
   RAJA::expt::Resources(RAJA::expt::Teams(NTeams,NTeams),
                         RAJA::expt::Threads(GPU_TB_SZ,GPU_TB_SZ)),
       [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

   RAJA::expt::loop<omp_col_policy0>(ctx, col_range, [&] (int col) {
       RAJA::expt::loop<omp_row_policy0>(ctx, row_range, [&] (int row) {

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          Cview(row, col) = dot;
      });
    });

  });

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - collapse)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy collapses the row and col loops in an OpenMP parallel region.
  // This is the same as using an OpenMP 'parallel for' directive on the
  // outer loop with a 'collapse(2) clause.
  //
  using global_thread_xy = RAJA::expt::LoopPolicy<RAJA::expt::omp_parallel_nested_for_exec,
                                                  RAJA::expt::cuda_global_thread_xy>;

   RAJA::expt::launch<launch_policy>(RAJA::expt::HOST,
    RAJA::expt::Resources(RAJA::expt::Teams(NTeams,NTeams),
                          RAJA::expt::Threads(GPU_TB_SZ,GPU_TB_SZ)),
   [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

     RAJA::expt::loop<global_thread_xy>(ctx, col_range, row_range, [&] (int col, int row) {

           double dot = 0.0;
           for (int k = 0; k < N; ++k) {
             dot += Aview(row, k) * Bview(k, col);
           }
           Cview(row, col) = dot;

     });

   });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_OPENMP

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA mat-mult (RAJA-nested)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy replaces the loop nest with a single CUDA kernel launch
  // (kernel body is the lambda loop body) where the row indices are
  // assigned to thread blocks and the col indices are assigned to
  // threads within each block.
  //
  // This is equivalent to launching a CUDA kernel with grid dimension N
  // and blocksize N; i.e., kernel<<<N, N>>> and defining row = blockIdx.x
  // and col = threadIdx.x in the kernel.
  //
  //
   RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
    RAJA::expt::Resources(RAJA::expt::Teams(N),
                          RAJA::expt::Threads(N)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

    RAJA::expt::loop<teams_x>(ctx, col_range, [&] (int col) {
        RAJA::expt::loop<threads_x>(ctx, row_range, [&] (int row) {

           double dot = 0.0;
           for (int k = 0; k < N; ++k) {
             dot += Aview(row, k) * Bview(k, col);
           }
           Cview(row, col) = dot;
       });
     });

   });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA tiled mat-mult ...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy collapses the col and row loops into a single CUDA kernel
  // using two-dimensional CUDA thread blocks with x and y dimensions defined
  // by GPU_TB_SZ arguments.
  //
  // When the matrix dimension N is an integer multiple of GPU_TB_SZ,
  // the CUDA grid and thread dimension kernel launch parameters will be the
  // same as in this kernel and the one above.
  //
   RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
    RAJA::expt::Resources(RAJA::expt::Teams(NTeams,NTeams),
                          RAJA::expt::Threads(GPU_TB_SZ,GPU_TB_SZ)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

        RAJA::expt::tile<teams_y>
          (ctx, GPU_TB_SZ, row_range, [&] (RAJA::RangeSegment const &row_tile) {
            RAJA::expt::tile<teams_x>
              (ctx, GPU_TB_SZ, col_range, [&] (RAJA::RangeSegment const &col_tile) {

                RAJA::expt::loop<threads_y>(ctx, row_tile, [&] (int col) {
                    RAJA::expt::loop<threads_x>(ctx, col_tile, [&] (int row) {

                        double dot = 0.0;
                        for (int k = 0; k < N; ++k) {
                          dot += Aview(row, k) * Bview(k, col);
                        }
                        Cview(row, col) = dot;

                      });
                 });

            });
       });

   });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

#endif // if RAJA_ENABLE_CUDA

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  double *d_A = memoryManager::allocate_gpu<double>(N * N);
  double *d_B = memoryManager::allocate_gpu<double>(N * N);
  double *d_C = memoryManager::allocate_gpu<double>(N * N);

  std::cout << "\n Running HIP mat-mult (RAJA-nested - POL4)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  hipErrchk(hipMemcpy( d_A, A, N * N * sizeof(double), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_B, B, N * N * sizeof(double), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice ));

  RAJA::View<double, RAJA::Layout<DIM>> d_Aview(d_A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> d_Bview(d_B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> d_Cview(d_C, N, N);

  //
  // This policy replaces the loop nest with a single HIP kernel launch
  // (kernel body is the lambda loop body) where the row indices are
  // assigned to thread blocks and the col indices are assigned to
  // threads within each block.
  //
  // This is equivalent to launching a HIP kernel with grid dimension N
  // and blocksize N; i.e., kernel<<<N, N>>> and defining row = blockIdx.x
  // and col = threadIdx.x in the kernel.
  //
   RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
    RAJA::expt::Resources(RAJA::expt::Teams(N),
                          RAJA::expt::Threads(N)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

     RAJA::expt::loop<teams_x>(ctx, col_range, [&] (int col) {
       RAJA::expt::loop<threads_x>(ctx, row_range, [&] (int row) {

            double dot = 0.0;
            for (int k = 0; k < N; ++k) {
              dot += d_Aview(row, k) * d_Bview(k, col);
            }

            d_Cview(row, col) = dot;

        });
     });
  });

  hipErrchk(hipMemcpy( C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost ));
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running HIP tiled mat-mult (RAJA-POL5)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  hipErrchk(hipMemcpy( d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice ));

  //
  // This policy collapses the col and row loops into a single HIP kernel
  // using two-dimensional HIP thread blocks with x and y dimensions defined
  // by HIP_BLOCK_SIZE arguments.
  //
  // When the matrix dimension N is an integer multiple of HIP_BLOCK_SIZE,
  // the HIP grid and thread dimension kernel launch parameters will be the
  // same as in this kernel and the one above.
  //
  RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
    RAJA::expt::Resources(RAJA::expt::Teams(NTeams,NTeams),
                          RAJA::expt::Threads(GPU_TB_SZ,GPU_TB_SZ)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

        RAJA::expt::tile<teams_y>
          (ctx, GPU_TB_SZ, row_range, [&] (RAJA::RangeSegment const &x_tile) {
            RAJA::expt::tile<teams_x>
              (ctx, GPU_TB_SZ, col_range, [&] (RAJA::RangeSegment const &y_tile) {

                RAJA::expt::loop<threads_y>(ctx, y_tile, [&] (int col) {
                    RAJA::expt::loop<threads_x>(ctx, x_tile, [&] (int row) {

                        double dot = 0.0;
                        for (int k = 0; k < N; ++k) {
                          dot += d_Aview(row, k) * d_Bview(k, col);
                        }

                        d_Cview(row, col) = dot;
                      });
                 });

            });
       });

   });
  hipErrchk(hipMemcpy( C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost ));
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_HIP

//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA tiled mat-mult with shared memory ...\n";

  std::memset(C, 0, N*N * sizeof(double));

  using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;

  RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
    RAJA::expt::Resources(RAJA::expt::Teams(NTeams,NTeams),
                          RAJA::expt::Threads(GPU_TB_SZ,GPU_TB_SZ)),
     [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {
   //
   // Loop over teams
   //
   RAJA::expt::tile<teams_y>
     (ctx, GPU_TB_SZ, row_range, [&] (RAJA::RangeSegment const &y_tile) {
     RAJA::expt::tile<teams_x>
       (ctx, GPU_TB_SZ, col_range, [&] (RAJA::RangeSegment const &x_tile) {

         RAJA_TEAM_SHARED double As[GPU_TB_SZ][GPU_TB_SZ];
         RAJA_TEAM_SHARED double Bs[GPU_TB_SZ][GPU_TB_SZ];
         RAJA_TEAM_SHARED double Cs[GPU_TB_SZ][GPU_TB_SZ];

         // Team parallel loop
         RAJA::expt::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
             RAJA::expt::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
                 Cs[ty][tx] = 0.0;
               });
           });

         // Slide across matrix
         RAJA::expt::tile<seq_loop>
           (ctx, GPU_TB_SZ, dot_range, [&] (RAJA::RangeSegment const &k_tile) {

           RAJA::expt::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
               RAJA::expt::loop_icount<threads_x>(ctx, k_tile, [&](int k_id, int tx) {
                   As[ty][tx] = Aview(row,k_id);
                 });
             });

           RAJA::expt::loop_icount<threads_y>(ctx, k_tile, [&](int k_id, int ty) {
               RAJA::expt::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
                   Bs[ty][tx] = Bview(k_id,col);
               });
             });

           ctx.teamSync();

           RAJA::expt::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
               RAJA::expt::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {

                   RAJA::expt::loop_icount<seq_loop>(ctx, k_tile, [&] (int gid, int e) {
                       Cs[ty][tx] += As[ty][e] * Bs[e][tx];
                     });

                 });
             });
           ctx.teamSync();

         });  // slide across matrix

          RAJA::expt::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
               RAJA::expt::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
                   Cview(col,row) = Cs[ty][tx];
               });
           });
       });
     });
  });  // kernel

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif
//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA tiled mat-mult (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // Define thread block dimensions
  dim3 blockdim(GPU_TB_SZ, GPU_TB_SZ);
  // Define grid dimensions to match the RAJA version above
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

//printf("griddim = (%d,%d), blockdim = (%d,%d)\n", (int)griddim.x, (int)griddim.y, (int)blockdim.x, (int)blockdim.y);

  // Launch CUDA kernel defined near the top of this file.
  matMultKernel<<<griddim, blockdim>>>(N, C, A, B);

  cudaDeviceSynchronize();

  checkResult<double>(Cview, N);

  std::cout << "\n Running CUDA tiled mat-mult with shared memory (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  matMultKernel<<<griddim, blockdim>>>(N, C, A, B);

  cudaDeviceSynchronize();

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

#endif // if RAJA_ENABLE_CUDA

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  std::cout << "\n Running HIP tiled mat-mult (no RAJA)...\n";

  // Define thread block dimensions
  dim3 blockdim(HIP_BLOCK_SIZE, HIP_BLOCK_SIZE);
  // Define grid dimensions to match the RAJA version above
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

//printf("griddim = (%d,%d), blockdim = (%d,%d)\n", (int)griddim.x, (int)griddim.y, (int)blockdim.x, (int)blockdim.y);

  std::memset(C, 0, N*N * sizeof(double));
  hipErrchk(hipMemcpy( d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice ));

  // Launch HIP kernel defined near the top of this file.
  hipLaunchKernelGGL((matMultKernel), dim3(griddim), dim3(blockdim), 0, 0, N, d_C, d_A, d_B);

  hipDeviceSynchronize();

  hipErrchk(hipMemcpy( C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost ));
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

  std::cout << "\n Running HIP tiled mat-mult with shared memory (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  hipErrchk(hipMemcpy( d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice ));

  // Launch HIP kernel defined near the top of this file.
  hipLaunchKernelGGL((sharedMatMultKernel), dim3(griddim), dim3(blockdim), 0, 0, N, d_C, d_A, d_B);

  hipDeviceSynchronize();

  hipErrchk(hipMemcpy( C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost ));
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

  memoryManager::deallocate_gpu(d_A);
  memoryManager::deallocate_gpu(d_B);
  memoryManager::deallocate_gpu(d_C);
#endif // if RAJA_ENABLE_HIP

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Functions to check result and report P/F.
//
template <typename T>
void checkResult(T* C, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( std::abs( C(row, col) - row * col * N ) > 10e-12 ) {
        match = false;
      }
    }
  }
  if ( match ) {
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
      if ( std::abs( Cview(row, col) - row * col * N ) > 10e-12 ) {
        match = false;
      }
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Functions to print result.
//
template <typename T>
void printResult(T* C, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = "
                << C(row, col) << std::endl;
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
      std::cout << "C(" << row << "," << col << ") = "
                << Cview(row, col) << std::endl;
    }
  }
  std::cout << std::endl;
}
