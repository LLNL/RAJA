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
#include <chrono>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

typedef std::chrono::high_resolution_clock Clock;

using launch_policy = RAJA::expt::LaunchPolicy<
    RAJA::expt::seq_launch_t
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
using gpu_global_thread_xy_policy = RAJA::expt::cuda_global_thread_xy;
#endif

#if defined(RAJA_ENABLE_HIP)
using gpu_block_x_policy = RAJA::hip_block_x_direct;
using gpu_block_y_policy = RAJA::hip_block_y_direct;
using gpu_thread_x_policy = RAJA::hip_thread_x_loop;
using gpu_thread_y_policy = RAJA::hip_thread_y_loop;
using gpu_global_thread_x_policy = RAJA::expt::hip_global_thread_x;
using gpu_global_thread_y_policy = RAJA::expt::hip_global_thread_y;
using gpu_global_thread_xy_policy = RAJA::expt::hip_global_thread_xy;
#endif

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

/*
  Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
#endif

#if defined(RAJA_ENABLE_HIP)
#define HIP_BLOCK_SIZE 16
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
*/
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
__global__ void matMultKernel2(int N, double* C, double* A, double* B)
{

  int Row = blockIdx.y*CUDA_BLOCK_SIZE + threadIdx.y;
  int Col = blockIdx.x*CUDA_BLOCK_SIZE + threadIdx.x;

  __shared__ double As[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
  __shared__ double Bs[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
  __shared__ double Cs[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

  Cs[threadIdx.y][threadIdx.x] = 0.0;

  for (int k = 0; k < (CUDA_BLOCK_SIZE + N - 1)/CUDA_BLOCK_SIZE; k++) {

    if (k*CUDA_BLOCK_SIZE + threadIdx.x < N && Row < N)
      As[threadIdx.y][threadIdx.x] = A[Row*N + k*CUDA_BLOCK_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*CUDA_BLOCK_SIZE + threadIdx.y < N && Col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(k*CUDA_BLOCK_SIZE + threadIdx.y)*N + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < CUDA_BLOCK_SIZE; ++n)
      Cs[threadIdx.y][threadIdx.x] += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }

  if (Row < N && Col < N)
    C[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
      (blockIdx.x * blockDim.x)+ threadIdx.x] = Cs[threadIdx.y][threadIdx.x];
}

__global__ void matMultKernel1(int N, double* C, double* A, double* B)
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

__global__ void matMultKernel0(int N, double* C, double* A, double* B)
{

  {int row = blockIdx.x; 
    for(int col = threadIdx.x; col < N; col+=blockDim.x) { 

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += A(row, k) * B(k, col);
    }

    C(row, col) = dot;

    }
  }

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
  const int N = 10000;
  //const int N = 25;
//const int N = CUDA_BLOCK_SIZE * CUDA_BLOCK_SIZE;

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

  //std::cout << "\n Running C-version of matrix multiplication...\n";

  std::memset(C, 0, N*N * sizeof(double));

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

  //Dry run

  matMultKernel0<<<N, CUDA_BLOCK_SIZE>>>(N, C, A, B);
  cudaDeviceSynchronize();

  {
    printf("CUDA kernel 0 \n");
    auto t0 = Clock::now();
    matMultKernel0<<<N, 256>>>(N, C, A, B);
    cudaDeviceSynchronize();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
  }

  dim3 blockdim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));
  {
    printf("CUDA kernel 1 \n");
    auto t0 = Clock::now();
    matMultKernel1<<<griddim, blockdim>>>(N, C, A, B);
    cudaDeviceSynchronize();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
  }


  {
    printf("CUDA kernel 2 \n");
    auto t0 = Clock::now();
    matMultKernel2<<<griddim, blockdim>>>(N, C, A, B);
    cudaDeviceSynchronize();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
  }


  printf("\n");
  //-------
  //RAJA Teams 
  //--------
  {
    printf("RAJA TEAM kernel 1  \n");
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
                        RAJA::expt::Resources(RAJA::expt::Teams(N),
                         RAJA::expt::Threads(256)),
    [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

    RAJA::expt::loop<teams_x>(ctx, row_range, [&] (int row) {
       RAJA::expt::loop<threads_x>(ctx, col_range, [&] (int col) {

           double dot = 0.0;
           for (int k = 0; k < N; ++k) {
             dot += Aview(row, k) * Bview(k, col);
           }
           
           Cview(row, col) = dot;
           
        });
      });

    });

    cudaDeviceSynchronize();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
  }

  {
    printf("RAJA TEAM kernel 2  \n");
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
            RAJA::expt::Resources(RAJA::expt::Teams(griddim.x,griddim.y),
                          RAJA::expt::Threads(CUDA_BLOCK_SIZE,CUDA_BLOCK_SIZE)),
      [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

#if 0
   RAJA::expt::loop<global_thread_y>(ctx, row_range, [&] (int row) {
       RAJA::expt::loop<global_thread_x>(ctx, col_range, [&] (int col) {

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          Cview(row, col) = dot;
      });
    });

#else
      RAJA::expt::tile<teams_y>
        (ctx, CUDA_BLOCK_SIZE, row_range, [&] (RAJA::RangeSegment const &row_tile) {
          RAJA::expt::tile<teams_x>
            (ctx, CUDA_BLOCK_SIZE, col_range, [&] (RAJA::RangeSegment const &col_tile) {

              RAJA::expt::loop<threads_y>(ctx, row_tile, [&] (int row) {
                  RAJA::expt::loop<threads_x>(ctx, col_tile, [&] (int col) {

                    double dot = 0.0;
                    for (int k = 0; k < N; ++k) {
                      dot += Aview(row, k) * Bview(k, col);
                    }

                    Cview(row, col) = dot;

                  });
                });
            });
        });
#endif
    });

    cudaDeviceSynchronize();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
  }


  {
    printf("RAJA TEAM kernel 3  \n");
    using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
      RAJA::expt::Resources(RAJA::expt::Teams(griddim.x,griddim.y),
                        RAJA::expt::Threads(CUDA_BLOCK_SIZE,CUDA_BLOCK_SIZE)),
     [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {
                                        
   // Loop over teams
   //
   RAJA::expt::tile<teams_y>
     (ctx, CUDA_BLOCK_SIZE, row_range, [&] (RAJA::RangeSegment const &y_tile) {
     RAJA::expt::tile<teams_x>
       (ctx, CUDA_BLOCK_SIZE, col_range, [&] (RAJA::RangeSegment const &x_tile) {

         RAJA_TEAM_SHARED double As[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
         RAJA_TEAM_SHARED double Bs[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
         RAJA_TEAM_SHARED double Cs[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

         RAJA::expt::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
             RAJA::expt::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
               Cs[ty][tx] = 0.0;
             });
         });

         RAJA::expt::tile<seq_loop>
           (ctx, CUDA_BLOCK_SIZE, dot_range, [&] (RAJA::RangeSegment const &k_tile) {

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
    cudaDeviceSynchronize();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
  }



  //=======================================
  //Tiling with kernel 
  printf("\n");
  {
    printf("RAJA Kernel 2  \n");
    auto t0 = Clock::now();

    using EXEC_POL5 =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernel<
          RAJA::statement::Tile<1, RAJA::tile_fixed<CUDA_BLOCK_SIZE>, RAJA::cuda_block_y_direct,
            RAJA::statement::Tile<0, RAJA::tile_fixed<CUDA_BLOCK_SIZE>, RAJA::cuda_block_x_direct,
              RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
                RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >
      >;

    RAJA::kernel<EXEC_POL5>(RAJA::make_tuple(col_range, row_range),
      [=] RAJA_DEVICE (int col, int row) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
      Cview(row, col) = dot;
      
   });

    cudaDeviceSynchronize();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
   
  } 




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
