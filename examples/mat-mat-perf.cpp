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

//------------------
//CUDA variants
//------------------

//#0,1
//Maps rows to cuda blocks,
//and uses block stride loop for column loop.
#define CUDA_KERNEL_0
#define CUDA_KERNEL_0_BOUNDS //adds launch bounds

//#2,3
//Maps 2D global thread id to output matrix, each thread
//is responsible for computing one dot product.
#define CUDA_KERNEL_1
#define CUDA_KERNEL_1_BOUNDS //adds launch bounds

//#4,5
//Performs multiplication by accomulating result
//in terms of tiles (shmem) then writing out to
//global memory.
#define CUDA_KERNEL_2
#define CUDA_KERNEL_2_BOUNDS // adds launch bounds

//#6,7
//Performs multiplication by accomulating result
//in terms of tiles (shmem) then writing out to
//global memory. Accumulates result by using
//thread private memory.
#define CUDA_KERNEL_2_REG
#define CUDA_KERNEL_2_REG_BOUNDS // adds launch bounds

//------------------
//RAJA teams variants
//------------------

//#8,9
//Same as Device kernel0.
#define TEAMS_KERNEL_0
#define TEAMS_KERNEL_0_BOUNDS //adds launch bounds

//#10,11
//Breaks up the iteration space into 2D tiles
//maps each tile to a CUDA block and threads
//to values within the block.
#define TEAMS_KERNEL_1_RAJA_TILED
#define TEAMS_KERNEL_1_RAJA_TILED_BOUNDS //adds launch bounds

//#12,13
//Maps 2D global thread id to output matrix, each thread
//is responsible for computing one dot product.
#define TEAMS_KERNEL_1_GLOBAL
#define TEAMS_KERNEL_1_GLOBAL_BOUNDS //adds launch bounds

//#14,15
//Performs multiplication by accumulating the result
//in terms of tiles (shmem) then writing out to
//global memory. The RAJA tiling methods are used
//to break up the iteration space into tiles, and
//the solution is accumulated within a tile.
#define TEAMS_KERNEL_2_RAJA_TILED_SHARED
#define TEAMS_KERNEL_2_RAJA_TILED_SHARED_BOUNDS //adds launch bounds

//#16,17
//Performs multiplication by accumulating the result
//in terms of tiles (shmem) then writing out to
//global memory. The RAJA tiling methods are used
//to break up the iteration space into tiles, and
//the solution is accumulated within a tile.
//Uses thread local register value to accumulate the
//result.
#define TEAMS_KERNEL_2_RAJA_TILED_SHARED_REG
#define TEAMS_KERNEL_2_RAJA_TILED_SHARED_REG_BOUNDS //adds launch bounds

//#18,19
//Performs multiplication by accumulating the result
//in terms of tiles (shmem) then writing out to
//global memory. Manual indexing is used to perform
//calculations.
#define TEAMS_KERNEL_2_DIRECT_TILED_SHARED
#define TEAMS_KERNEL_2_DIRECT_TILED_SHARED_BOUNDS //adds launch bounds

//#20,21
//Performs multiplication by accumulating the result
//in terms of tiles (shmem) then writing out to
//global memory. Manual indexing is used to perform
//calculations. Uses thread local register value to
//accumulate the result.
#define TEAMS_KERNEL_2_DIRECT_TILED_SHARED_REG
#define TEAMS_KERNEL_2_DIRECT_TILED_SHARED_REG_BOUNDS //adds launch bounds

//------------------
//RAJA kernel variants
//------------------

//#22,23
//Breaks up the iteration space into 2D tiles
//maps each tile to a CUDA block and threads
//to values within the block.
#define RAJA_KERNEL_1_TILED
#define RAJA_KERNEL_1_TILED_BOUNDS //adds launch bounds

//#24,25
//Performs multiplication by accumulating the result
//in terms of tiles (shmem) then writing out to
//global memory. The RAJA tiling methods are used
//to break up the iteration space into tiles, and
//the solution is accumulated within a tile.
#define RAJA_KERNEL_2_TILED_SHARED
#define RAJA_KERNEL_2_TILED_SHARED_BOUNDS //adds launch bounds

//#26,27
//Performs multiplication by accumulating the result
//in terms of tiles (shmem) then writing out to
//global memory. The RAJA tiling methods are used
//to break up the iteration space into tiles, and
//the solution is accumulated within a tile.
//Uses thread local register value to accumulate the
//result.
#define RAJA_KERNEL_2_TILED_SHARED_REG
#define RAJA_KERNEL_2_TILED_SHARED_REG_BOUNDS //adds launch bounds


typedef std::chrono::high_resolution_clock Clock;

using launch_policy = RAJA::expt::LaunchPolicy<
    RAJA::expt::seq_launch_t
#if defined(RAJA_ENABLE_CUDA)
    ,
    RAJA::expt::cuda_launch_t<true>
#endif
#if defined(RAJA_ENABLE_HIP)
    ,
    RAJA::expt::hip_launch_t<true>
#endif
    >;

using launch_policy_bounds = RAJA::expt::LaunchPolicy<
    RAJA::expt::seq_launch_t
#if defined(RAJA_ENABLE_CUDA)
    ,
    RAJA::expt::cuda_launch_t<true, 256>
#endif
#if defined(RAJA_ENABLE_HIP)
    ,
    RAJA::expt::hip_launch_t<true, 256>
#endif
    >;

using loop_policy = RAJA::loop_exec;

#if defined(RAJA_ENABLE_CUDA)
using gpu_block_x_policy = RAJA::cuda_block_x_direct;
using gpu_block_y_policy = RAJA::cuda_block_y_direct;
using gpu_thread_x_policy = RAJA::cuda_thread_x_direct;
using gpu_thread_y_policy = RAJA::cuda_thread_y_direct;
using gpu_thread_x_loop_policy = RAJA::cuda_thread_x_loop;
using gpu_thread_y_loop_policy = RAJA::cuda_thread_y_loop;
using gpu_global_thread_x_policy = RAJA::expt::cuda_global_thread_x;
using gpu_global_thread_y_policy = RAJA::expt::cuda_global_thread_y;
using gpu_global_thread_xy_policy = RAJA::expt::cuda_global_thread_xy;
#endif

#if defined(RAJA_ENABLE_HIP)
using gpu_block_x_policy = RAJA::hip_block_x_direct;
using gpu_block_y_policy = RAJA::hip_block_y_direct;
using gpu_thread_x_policy = RAJA::hip_thread_x_direct;
using gpu_thread_y_policy = RAJA::hip_thread_y_direct;
using gpu_thread_x_loop_policy = RAJA::hip_thread_x_loop;
using gpu_thread_y_loop_policy = RAJA::hip_thread_y_loop;
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

using threads_x_loop = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_thread_x_loop_policy
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
#if defined(RAJA_DEVICE_ACTIVE)
#define DEVICE_BLOCK_SIZE 16
#endif

#if defined (RAJA_ENABLE_CUDA)
#define DEVICE_SYNC() cudaDeviceSynchronize();
using kernel_sync_threads = RAJA::statement::CudaSyncThreads;
using gpu_shared_mem = RAJA::cuda_shared_mem;
#endif

#if defined (RAJA_ENABLE_HIP)
#define DEVICE_SYNC() hipDeviceSynchronize();
using kernel_sync_threads = RAJA::statement::HipSyncThreads;
using gpu_shared_mem = RAJA::hip_shared_mem;
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
__global__
void matMultKernel2_shared(int N, double* C, double* A, double* B)
{

  int Row = blockIdx.y*DEVICE_BLOCK_SIZE + threadIdx.y;
  int Col = blockIdx.x*DEVICE_BLOCK_SIZE + threadIdx.x;

  __shared__ double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
  __shared__ double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
  __shared__ double Cs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];

  Cs[threadIdx.y][threadIdx.x] = 0.0;

  for (int k = 0; k < (DEVICE_BLOCK_SIZE + N - 1)/DEVICE_BLOCK_SIZE; k++) {

    if (k*DEVICE_BLOCK_SIZE + threadIdx.x < N && Row < N)
      As[threadIdx.y][threadIdx.x] = A[Row*N + k*DEVICE_BLOCK_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*DEVICE_BLOCK_SIZE + threadIdx.y < N && Col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(k*DEVICE_BLOCK_SIZE + threadIdx.y)*N + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < DEVICE_BLOCK_SIZE; ++n)
      Cs[threadIdx.y][threadIdx.x] += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }

  if (Row < N && Col < N)
    C[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
      (blockIdx.x * blockDim.x)+ threadIdx.x] = Cs[threadIdx.y][threadIdx.x];
}

__launch_bounds__(DEVICE_BLOCK_SIZE*DEVICE_BLOCK_SIZE, 1) __global__
void matMultKernel2_bounds_shared(int N, double* C, double* A, double* B)
{

  int Row = blockIdx.y*DEVICE_BLOCK_SIZE + threadIdx.y;
  int Col = blockIdx.x*DEVICE_BLOCK_SIZE + threadIdx.x;

  __shared__ double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
  __shared__ double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
  __shared__ double Cs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];

  Cs[threadIdx.y][threadIdx.x] = 0.0;

  for (int k = 0; k < (DEVICE_BLOCK_SIZE + N - 1)/DEVICE_BLOCK_SIZE; k++) {

    if (k*DEVICE_BLOCK_SIZE + threadIdx.x < N && Row < N)
      As[threadIdx.y][threadIdx.x] = A[Row*N + k*DEVICE_BLOCK_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*DEVICE_BLOCK_SIZE + threadIdx.y < N && Col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(k*DEVICE_BLOCK_SIZE + threadIdx.y)*N + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < DEVICE_BLOCK_SIZE; ++n)
      Cs[threadIdx.y][threadIdx.x] += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }

  if (Row < N && Col < N)
    C[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
      (blockIdx.x * blockDim.x)+ threadIdx.x] = Cs[threadIdx.y][threadIdx.x];
}

__global__
void matMultKernel2_register(int N, double* C, double* A, double* B)
{

  int Row = blockIdx.y*DEVICE_BLOCK_SIZE + threadIdx.y;
  int Col = blockIdx.x*DEVICE_BLOCK_SIZE + threadIdx.x;

  __shared__ double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
  __shared__ double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
  double dot = 0.0;

  for (int k = 0; k < (DEVICE_BLOCK_SIZE + N - 1)/DEVICE_BLOCK_SIZE; k++) {

    if (k*DEVICE_BLOCK_SIZE + threadIdx.x < N && Row < N)
      As[threadIdx.y][threadIdx.x] = A[Row*N + k*DEVICE_BLOCK_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*DEVICE_BLOCK_SIZE + threadIdx.y < N && Col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(k*DEVICE_BLOCK_SIZE + threadIdx.y)*N + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < DEVICE_BLOCK_SIZE; ++n)
      dot += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }

  if (Row < N && Col < N)
    C[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
      (blockIdx.x * blockDim.x)+ threadIdx.x] = dot;
}

__launch_bounds__(DEVICE_BLOCK_SIZE*DEVICE_BLOCK_SIZE, 1) __global__
void matMultKernel2_bounds_register(int N, double* C, double* A, double* B)
{

  int Row = blockIdx.y*DEVICE_BLOCK_SIZE + threadIdx.y;
  int Col = blockIdx.x*DEVICE_BLOCK_SIZE + threadIdx.x;

  __shared__ double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
  __shared__ double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
  double dot = 0.0;

  for (int k = 0; k < (DEVICE_BLOCK_SIZE + N - 1)/DEVICE_BLOCK_SIZE; k++) {

    if (k*DEVICE_BLOCK_SIZE + threadIdx.x < N && Row < N)
      As[threadIdx.y][threadIdx.x] = A[Row*N + k*DEVICE_BLOCK_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*DEVICE_BLOCK_SIZE + threadIdx.y < N && Col < N)
      Bs[threadIdx.y][threadIdx.x] = B[(k*DEVICE_BLOCK_SIZE + threadIdx.y)*N + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < DEVICE_BLOCK_SIZE; ++n)
      dot += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }

  if (Row < N && Col < N)
    C[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
      (blockIdx.x * blockDim.x)+ threadIdx.x] = dot;
}

__global__
void matMultKernel1(int N, double* C, double* A, double* B)
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

__launch_bounds__(DEVICE_BLOCK_SIZE*DEVICE_BLOCK_SIZE, 1) __global__
void matMultKernel1_bounds(int N, double* C, double* A, double* B)
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

__global__
void matMultKernel0(int N, double* C, double* A, double* B)
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

__launch_bounds__(256, 1) __global__
void matMultKernel0_bounds(int N, double* C, double* A, double* B)
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
//const int N = DEVICE_BLOCK_SIZE * DEVICE_BLOCK_SIZE;

//
// Allocate and initialize matrix data.
//
  double *A = memoryManager::allocate<double>(N * N);
  double *B = memoryManager::allocate<double>(N * N);
  double *C = memoryManager::allocate<double>(N * N);

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A(row, col) = 1;
      B(row, col) = 1;
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
  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, N);
  RAJA::TypedRangeSegment<int> dot_range(0, N);
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

//First run to get the data on the device
#if defined(RAJA_ENABLE_HIP)
  hipLaunchKernelGGL((matMultKernel0), N, 256, 0, 0, N, C, A, B);
#else
  matMultKernel0<<<N, 256>>>(N, C, A, B);
#endif
  DEVICE_SYNC();

  dim3 blockdim(DEVICE_BLOCK_SIZE, DEVICE_BLOCK_SIZE);
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

#if defined(CUDA_KERNEL_0)
  {
    printf("# 0");
    printf("\nDevice kernel 0 \n");
    printf("Maps CUDA blocks to rows \n");
    printf("Uses block stride loop for columns \n");
    auto t0 = Clock::now();
#if defined(RAJA_ENABLE_HIP)
  hipLaunchKernelGGL((matMultKernel0), N, 256, 0, 0, N, C, A, B);
#else
  matMultKernel0<<<N, 256>>>(N, C, A, B);
#endif
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(CUDA_KERNEL_0_BOUNDS)
  {
    printf("# 1");
    printf("\nDevice kernel 0 (w launch bounds) \n");
    printf("Maps CUDA blocks to rows \n");
    printf("Uses block stride loop for columns \n");
    printf("Adds launch bounds \n");
    auto t0 = Clock::now();
#if defined(RAJA_ENABLE_HIP)
    hipLaunchKernelGGL((matMultKernel0_bounds), N, 256, 0, 0, N, C, A, B);
#else
    matMultKernel0_bounds<<<N, 256>>>(N, C, A, B);
#endif
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(CUDA_KERNEL_1)
  {
    printf("# 2");
    printf("\nDevice kernel 1 \n");
    printf("Generates global 2D threads \n");
    printf("Each thread performs dot product \n");
    auto t0 = Clock::now();
#if defined(RAJA_ENABLE_HIP)
    hipLaunchKernelGGL((matMultKernel1), dim3(griddim), dim3(blockdim),
                       0, 0, N, C, A, B);
#else
    matMultKernel1<<<griddim, blockdim>>>(N, C, A, B);
#endif
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(CUDA_KERNEL_1_BOUNDS)
  {
    printf("# 3");
    printf("\nDevice kernel 1 (w launch bounds) \n");
    printf("Generates global 2D threads \n");
    printf("Each thread performs dot product \n");
    printf("Adds launch bounds \n");
    auto t0 = Clock::now();
#if defined(RAJA_ENABLE_HIP)
    hipLaunchKernelGGL((matMultKernel1_bounds), dim3(griddim), dim3(blockdim),
                       0, 0, N, C, A, B);
#else
    matMultKernel1_bounds<<<griddim, blockdim>>>(N, C, A, B);
#endif
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(CUDA_KERNEL_2)
  {
    printf("# 4");
    printf("\nDevice kernel 2 shared \n");
    printf("Performs multiplication by loading submatrices \n");
    printf("into shared memory tiles \n");
    printf("Solution is accumulated in shared memory \n");
    auto t0 = Clock::now();
#if defined(RAJA_ENABLE_HIP)
    hipLaunchKernelGGL((matMultKernel2_shared), dim3(griddim), dim3(blockdim),
                       0, 0, N, C, A, B);
#else
    matMultKernel2_shared<<<griddim, blockdim>>>(N, C, A, B);
#endif
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(CUDA_KERNEL_2_BOUNDS)
  {
    printf("# 5");
    printf("\nDevice kernel 2 shared (w launch bounds) \n");
    printf("Performs multiplication by loading submatrices \n");
    printf("into shared memory tiles \n");
    printf("Solution is accumulated in shared memory \n");
    printf("Adds launch bounds \n");
    auto t0 = Clock::now();
#if defined(RAJA_ENABLE_HIP)
    hipLaunchKernelGGL((matMultKernel2_bounds_shared),
                       dim3(griddim), dim3(blockdim),
                       0, 0, N, C, A, B);
#else
    matMultKernel2_bounds_shared<<<griddim, blockdim>>>(N, C, A, B);
#endif
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(CUDA_KERNEL_2_REG)
  {
    printf("# 6");
    printf("\nDevice kernel 2 shared + register \n");
    printf("Performs multiplication by loading submatrices \n");
    printf("into shared memory tiles \n");
    printf("Solution is accumulated in register space (not portable) \n");
    auto t0 = Clock::now();
#if defined(RAJA_ENABLE_HIP)
    hipLaunchKernelGGL((matMultKernel2_register),
                       dim3(griddim), dim3(blockdim),
                       0, 0, N, C, A, B);
#else
    matMultKernel2_register<<<griddim, blockdim>>>(N, C, A, B);
#endif
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(CUDA_KERNEL_2_REG_BOUNDS)
  {
    printf("# 7");
    printf("\nDevice kernel 2 shared + register \n");
    printf("Performs multiplication by loading submatrices \n");
    printf("into shared memory tiles \n");
    printf("Solution is accumulated in register space (not portable) \n");
    printf("Adds launch bounds \n");
    auto t0 = Clock::now();
#if defined(RAJA_ENABLE_HIP)
    hipLaunchKernelGGL((matMultKernel2_register),
                       dim3(griddim), dim3(blockdim),
                        0, 0, N, C, A, B);
#else
    matMultKernel2_bounds_register<<<griddim, blockdim>>>(N, C, A, B);
#endif

    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

  printf("\n");
  //-------
  //RAJA Teams
  //--------
#if defined(TEAMS_KERNEL_0)
  {
    printf("# 8");
    printf("\nRAJA Teams kernel 0  \n");
    printf("Maps CUDA blocks to rows \n");
    printf("Uses block stride loop for columns \n");
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(N),
                              RAJA::expt::Threads(256)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::loop<teams_x>(ctx, row_range, [&] (int row) {
        RAJA::expt::loop<threads_x_loop>(ctx, col_range, [&] (int col) {

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          Cview(row, col) = dot;

        });
      });
    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_0_BOUNDS)
  {
    printf("# 9");
    printf("\nRAJA Teams kernel 0 (w launch bounds) \n");
    printf("Blocks to rows, and block stride loop for columns \n");
    printf("Adds launch bounds \n");
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy_bounds>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(N),
                              RAJA::expt::Threads(256)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::loop<teams_x>(ctx, row_range, [&] (int row) {
        RAJA::expt::loop<threads_x_loop>(ctx, col_range, [&] (int col) {

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          Cview(row, col) = dot;

        });
      });
    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_1_RAJA_TILED)
  {
    printf("# 10");
    printf("\nRAJA Teams kernel 1 w RAJA tiling  \n");
    printf("Uses RAJA tiling to map blocks to tiles \n");
    printf("Maps threads to iterates within a tile \n");
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(griddim.x,griddim.y),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::tile<teams_y>(ctx, DEVICE_BLOCK_SIZE, row_range, [&] (RAJA::TypedRangeSegment<int> const &row_tile) {
        RAJA::expt::tile<teams_x>(ctx, DEVICE_BLOCK_SIZE, col_range, [&] (RAJA::TypedRangeSegment<int> const &col_tile) {

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
    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_1_RAJA_TILED_BOUNDS)
  {
    printf("# 11");
    printf("\nRAJA Teams kernel 1 w RAJA tiling (with launch bounds) \n");
    printf("Uses RAJA tiling to map blocks to tiles \n");
    printf("Maps threads to iterates within a tile \n");
    printf("Adds launch bounds \n");
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy_bounds>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(griddim.x,griddim.y),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::tile<teams_y>(ctx, DEVICE_BLOCK_SIZE, row_range, [&] (RAJA::TypedRangeSegment<int> const &row_tile) {
        RAJA::expt::tile<teams_x>(ctx, DEVICE_BLOCK_SIZE, col_range, [&] (RAJA::TypedRangeSegment<int> const &col_tile) {

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
    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_1_GLOBAL)
  {
    printf("# 12");
    printf("\nRAJA Teams kernel 1 global  \n");
    printf("Generates global 2D threads, each thread performs a dot product \n");
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(griddim.x,griddim.y),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::loop<global_thread_y>(ctx, row_range, [&] (int row) {
        RAJA::expt::loop<global_thread_x>(ctx, col_range, [&] (int col) {

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          Cview(row, col) = dot;

        });
      });
    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_1_GLOBAL_BOUNDS)
  {
    printf("# 13");
    printf("\nRAJA Teams kernel 1 global (w launch bounds) \n");
    printf("Generates global 2D threads, each thread performs a dot product \n");
    printf("Adds launch bounds \n");
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy_bounds>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(griddim.x,griddim.y),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::loop<global_thread_y>(ctx, row_range, [&] (int row) {
        RAJA::expt::loop<global_thread_x>(ctx, col_range, [&] (int col) {

          double dot = 0.0;
          for (int k = 0; k < N; ++k) {
            dot += Aview(row, k) * Bview(k, col);
          }
          Cview(row, col) = dot;

        });
      });
    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_2_RAJA_TILED_SHARED)
  {
    printf("# 14");
    printf("\nRAJA Teams kernel 2 RAJA tiled shared \n");
    printf("Uses RAJA methods to break up the iteration space into tiles\n");
    printf("Performs multiplication by loading submatrices into tiles of shmem \n");
    printf("Solution is accumulated in shared memory \n");
    using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
    int Nx = griddim.x;
    int Ny = griddim.y;
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      // Loop over teams
      //
      RAJA::expt::tile<teams_y>(ctx, DEVICE_BLOCK_SIZE, row_range, [&] (RAJA::TypedRangeSegment<int> const &y_tile) {
        RAJA::expt::tile<teams_x>(ctx, DEVICE_BLOCK_SIZE, col_range, [&] (RAJA::TypedRangeSegment<int> const &x_tile) {

          RAJA_TEAM_SHARED double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Cs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];

          RAJA::expt::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
            RAJA::expt::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
              Cs[ty][tx] = 0.0;
            });
          });

          RAJA::expt::tile<seq_loop>(ctx, DEVICE_BLOCK_SIZE, dot_range, [&] (RAJA::TypedRangeSegment<int> const &k_tile) {

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
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_2_RAJA_TILED_SHARED_BOUNDS)
  {
    printf("# 15");
    printf("\nRAJA Teams kernel 2 RAJA tiled shared (with launch bounds) \n");
    printf("Uses RAJA methods to break up the iteration space into tiles\n");
    printf("Performs multiplication by loading submatrices into tiles of shmem \n");
    printf("Solution is accumulated in shared memory \n");
    printf("Adds launch bounds \n");
    using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
    int Nx = griddim.x;
    int Ny = griddim.y;
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy_bounds>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      // Loop over teams
      //
      RAJA::expt::tile<teams_y>(ctx, DEVICE_BLOCK_SIZE, row_range, [&] (RAJA::TypedRangeSegment<int> const &y_tile) {
        RAJA::expt::tile<teams_x>(ctx, DEVICE_BLOCK_SIZE, col_range, [&] (RAJA::TypedRangeSegment<int> const &x_tile) {

          RAJA_TEAM_SHARED double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Cs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];

          RAJA::expt::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
            RAJA::expt::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
              Cs[ty][tx] = 0.0;
            });
          });

          RAJA::expt::tile<seq_loop>(ctx, DEVICE_BLOCK_SIZE, dot_range, [&] (RAJA::TypedRangeSegment<int> const &k_tile) {

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
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_2_RAJA_TILED_SHARED_REG)
  {
    printf("# 16");
    printf("\nRAJA Teams kernel 2 RAJA tiling + register \n");
    printf("Uses RAJA methods to break up the iteration space into tiles\n");
    printf("Performs multiplication by loading submatrices into tiles \n");
    printf("Solution is accumulated in register (not currently portable) \n");
    using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
    int Nx = griddim.x;
    int Ny = griddim.y;
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      // Loop over teams
      //
      RAJA::expt::tile<teams_y>(ctx, DEVICE_BLOCK_SIZE, row_range, [&] (RAJA::TypedRangeSegment<int> const &y_tile) {
        RAJA::expt::tile<teams_x>(ctx, DEVICE_BLOCK_SIZE, col_range, [&] (RAJA::TypedRangeSegment<int> const &x_tile) {

          RAJA_TEAM_SHARED double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          double dot = 0.0;

          RAJA::expt::tile<seq_loop>(ctx, DEVICE_BLOCK_SIZE, dot_range, [&] (RAJA::TypedRangeSegment<int> const &k_tile) {

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
                  dot += As[ty][e] * Bs[e][tx];
                });

              });
            });

            ctx.teamSync();

          });  // slide across matrix

          RAJA::expt::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
            RAJA::expt::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
              Cview(col,row) = dot;
            });
          });

        });
      });

    });  // kernel
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_2_RAJA_TILED_SHARED_REG_BOUNDS)
  {
    printf("# 17");
    printf("\nRAJA Teams kernel 2 RAJA tiling + register (with launch bounds) \n");
    printf("Uses RAJA methods to break up the iteration space into tiles\n");
    printf("Performs multiplication by loading submatrices into tiles \n");
    printf("Solution is accumulated in register (not currently portable) \n");
    printf("Adds launch bounds \n");
    using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
    int Nx = griddim.x;
    int Ny = griddim.y;
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy_bounds>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      // Loop over teams
      //
      RAJA::expt::tile<teams_y>(ctx, DEVICE_BLOCK_SIZE, row_range, [&] (RAJA::TypedRangeSegment<int> const &y_tile) {
        RAJA::expt::tile<teams_x>(ctx, DEVICE_BLOCK_SIZE, col_range, [&] (RAJA::TypedRangeSegment<int> const &x_tile) {

          RAJA_TEAM_SHARED double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          double dot = 0.0;

          RAJA::expt::tile<seq_loop>(ctx, DEVICE_BLOCK_SIZE, dot_range, [&] (RAJA::TypedRangeSegment<int> const &k_tile) {

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
                  dot += As[ty][e] * Bs[e][tx];
                });

              });
            });

            ctx.teamSync();

          });  // slide across matrix

          RAJA::expt::loop_icount<threads_y>(ctx, y_tile, [&](int row, int ty) {
            RAJA::expt::loop_icount<threads_x>(ctx, x_tile, [&](int col, int tx) {
              Cview(col,row) = dot;
            });
          });

        });
      });

    });  // kernel
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_2_DIRECT_TILED_SHARED)
  {
    printf("# 18");
    printf("\nRAJA Teams kernel 2 direct tiling\n");
    printf("Performs multiplication by loading submatrices into tiles \n");
    printf("Solution is accumulated in shared memory \n");
    using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
    int Nx = griddim.x;
    int Ny = griddim.y;
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::loop<teams_y>(ctx, RAJA::TypedRangeSegment<int>(0,Ny), [&] (int by) {
        RAJA::expt::loop<teams_x>(ctx, RAJA::TypedRangeSegment<int>(0,Nx), [&] (int bx) {

          RAJA_TEAM_SHARED double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Cs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];

          RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
            RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

              Cs[ty][tx] = 0.0;

            });
          });

          for (int k = 0; k < (DEVICE_BLOCK_SIZE + N - 1)/DEVICE_BLOCK_SIZE; k++) {

            RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
              RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

                const int Row = by*DEVICE_BLOCK_SIZE + ty;
                const int Col = bx*DEVICE_BLOCK_SIZE + tx;

                if (k*DEVICE_BLOCK_SIZE + tx < N && Row < N)
                  As[ty][tx] = Aview(Row, (k*DEVICE_BLOCK_SIZE + tx));
                              //A[Row*N + k*DEVICE_BLOCK_SIZE + tx];
                else
                  As[ty][tx] = 0.0;

                if (k*DEVICE_BLOCK_SIZE + ty < N && Col < N)
                  Bs[ty][tx] = Bview((k*DEVICE_BLOCK_SIZE + ty), Col);
                 // Bs[ty][tx] = B[(k*DEVICE_BLOCK_SIZE + ty)*N + Col];
                else
                  Bs[ty][tx] = 0.0;

              });
            });

            ctx.teamSync();

            RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
              RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

                for (int n = 0; n < DEVICE_BLOCK_SIZE; ++n)
                  Cs[ty][tx] += As[ty][n] * Bs[n][tx];

              });
            });

            ctx.teamSync();

          }

          RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
            RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

              int Row = by*DEVICE_BLOCK_SIZE + ty;
              int Col = bx*DEVICE_BLOCK_SIZE + tx;

              if (Row < N && Col < N)
                Cview(Row, Col) = Cs[ty][tx];
            });
          });

        });
      });

    });  // kernel
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_2_DIRECT_TILED_SHARED_BOUNDS)
  {
    printf("# 19");
    printf("\nRAJA Teams kernel 2 direct tiling (with launch bounds) \n");
    printf("Performs multiplication by loading submatrices into tiles \n");
    printf("Solution is accumulated in shared mem \n");
    printf("Adds launch bounds \n");
    using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
    int Nx = griddim.x;
    int Ny = griddim.y;
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy_bounds>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::loop<teams_y>(ctx, RAJA::TypedRangeSegment<int>(0,Ny), [&] (int by) {
        RAJA::expt::loop<teams_x>(ctx, RAJA::TypedRangeSegment<int>(0,Nx), [&] (int bx) {

          RAJA_TEAM_SHARED double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Cs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];

          RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
            RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

              Cs[ty][tx] = 0.0;

            });
          });

          for (int k = 0; k < (DEVICE_BLOCK_SIZE + N - 1)/DEVICE_BLOCK_SIZE; k++) {

            RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
              RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

                const int Row = by*DEVICE_BLOCK_SIZE + ty;
                const int Col = bx*DEVICE_BLOCK_SIZE + tx;

                if (k*DEVICE_BLOCK_SIZE + tx < N && Row < N)
                  As[ty][tx] = Aview(Row, (k*DEVICE_BLOCK_SIZE + tx));
                  //As[ty][tx] = A[Row*N + k*DEVICE_BLOCK_SIZE + tx];
                else
                  As[ty][tx] = 0.0;

                if (k*DEVICE_BLOCK_SIZE + ty < N && Col < N)
                  Bs[ty][tx] = Bview((k*DEVICE_BLOCK_SIZE + ty), Col);
                  //Bs[ty][tx] = B[(k*DEVICE_BLOCK_SIZE + ty)*N + Col];
                else
                  Bs[ty][tx] = 0.0;

              });
            });

            ctx.teamSync();

            RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
              RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

                for (int n = 0; n < DEVICE_BLOCK_SIZE; ++n)
                  Cs[ty][tx] += As[ty][n] * Bs[n][tx];

              });
            });

            ctx.teamSync();

          }

          RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
            RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

              int Row = by*DEVICE_BLOCK_SIZE + ty;
              int Col = bx*DEVICE_BLOCK_SIZE + tx;

              if (Row < N && Col < N)
                Cview(Row, Col) = Cs[ty][tx];
            });
          });

        });
      });

    });  // kernel
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_2_DIRECT_TILED_SHARED_REG)
  {
    printf("# 20");
    printf("\nRAJA Teams kernel 2 direct tiling with register \n");
    printf("Performs multiplication by loading submatrices into tiles \n");
    printf("Solution is accumulated in thread register \n");
    using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
    int Nx = griddim.x;
    int Ny = griddim.y;
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::loop<teams_y>(ctx, RAJA::TypedRangeSegment<int>(0,Ny), [&] (int by) {
        RAJA::expt::loop<teams_x>(ctx, RAJA::TypedRangeSegment<int>(0,Nx), [&] (int bx) {

          RAJA_TEAM_SHARED double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          double dot = 0.0;

          for (int k = 0; k < (DEVICE_BLOCK_SIZE + N - 1)/DEVICE_BLOCK_SIZE; k++) {

            RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
              RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

                const int Row = by*DEVICE_BLOCK_SIZE + ty;
                const int Col = bx*DEVICE_BLOCK_SIZE + tx;

                if (k*DEVICE_BLOCK_SIZE + tx < N && Row < N)
                  As[ty][tx] = Aview(Row, (k*DEVICE_BLOCK_SIZE + tx));
                  //As[ty][tx] = A[Row*N + k*DEVICE_BLOCK_SIZE + tx];
                else
                  As[ty][tx] = 0.0;

                if (k*DEVICE_BLOCK_SIZE + ty < N && Col < N)
                  Bs[ty][tx] = Bview((k*DEVICE_BLOCK_SIZE + ty), Col);
                  //Bs[ty][tx] = B[(k*DEVICE_BLOCK_SIZE + ty)*N + Col];
                else
                  Bs[ty][tx] = 0.0;

              });
            });

            ctx.teamSync();

            RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
              RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

                for (int n = 0; n < DEVICE_BLOCK_SIZE; ++n)
                  dot += As[ty][n] * Bs[n][tx];

              });
            });

            ctx.teamSync();

          }

          RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
            RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

              int Row = by*DEVICE_BLOCK_SIZE + ty;
              int Col = bx*DEVICE_BLOCK_SIZE + tx;

              if (Row < N && Col < N)
                Cview(Row, Col) = dot;
            });
          });

        });
      });

    });  // kernel
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(TEAMS_KERNEL_2_DIRECT_TILED_SHARED_REG_BOUNDS)
  {
    printf("# 21");
    printf("\nRAJA Teams kernel 2 direct tiling + register (w launch bounds) \n");
    printf("Performs multiplication by loading submatrices into tiles \n");
    printf("Solution is accumulated in thread register \n");
    printf("Adds launch bounds \n");
    using seq_loop =  RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;
    int Nx = griddim.x;
    int Ny = griddim.y;
    auto t0 = Clock::now();
    RAJA::expt::launch<launch_policy_bounds>(RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                              RAJA::expt::Threads(DEVICE_BLOCK_SIZE,DEVICE_BLOCK_SIZE)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

      RAJA::expt::loop<teams_y>(ctx, RAJA::TypedRangeSegment<int>(0,Ny), [&] (int by) {
        RAJA::expt::loop<teams_x>(ctx, RAJA::TypedRangeSegment<int>(0,Nx), [&] (int bx) {

          RAJA_TEAM_SHARED double As[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          RAJA_TEAM_SHARED double Bs[DEVICE_BLOCK_SIZE][DEVICE_BLOCK_SIZE];
          double dot = 0.0;

          for (int k = 0; k < (DEVICE_BLOCK_SIZE + N - 1)/DEVICE_BLOCK_SIZE; k++) {

            RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
              RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

                const int Row = by*DEVICE_BLOCK_SIZE + ty;
                const int Col = bx*DEVICE_BLOCK_SIZE + tx;

                if (k*DEVICE_BLOCK_SIZE + tx < N && Row < N)
                  As[ty][tx] = Aview(Row, (k*DEVICE_BLOCK_SIZE + tx));
                  //As[ty][tx] = A[Row*N + k*DEVICE_BLOCK_SIZE + tx];
                else
                  As[ty][tx] = 0.0;

                if (k*DEVICE_BLOCK_SIZE + ty < N && Col < N)
                  Bs[ty][tx] = Bview((k*DEVICE_BLOCK_SIZE + ty), Col);
                  //Bs[ty][tx] = B[(k*DEVICE_BLOCK_SIZE + ty)*N + Col];
                else
                  Bs[ty][tx] = 0.0;

              });
            });

            ctx.teamSync();

            RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
              RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

                for (int n = 0; n < DEVICE_BLOCK_SIZE; ++n)
                  dot += As[ty][n] * Bs[n][tx];

              });
            });

            ctx.teamSync();

          }

          RAJA::expt::loop<threads_y>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int ty) {
            RAJA::expt::loop<threads_x>(ctx, RAJA::TypedRangeSegment<int>(0,DEVICE_BLOCK_SIZE), [&] (int tx) {

              int Row = by*DEVICE_BLOCK_SIZE + ty;
              int Col = bx*DEVICE_BLOCK_SIZE + tx;

              if (Row < N && Col < N)
                Cview(Row, Col) = dot;
            });
          });

        });
      });

    });  // kernel
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif


  //=======================================
  //Tiling with kernel
#if 0
  printf("\n");
#if defined(RAJA_KERNEL_1_TILED)
  {
    printf("# 22");
    printf("\nRAJA Kernel 1  \n");
    printf("Uses RAJA methods to perform tiling\n");
    auto t0 = Clock::now();

    using EXEC_POL5 =
      RAJA::KernelPolicy<
#if defined(RAJA_ENABLE_CUDA)
        RAJA::statement::CudaKernelOccAsync<
#elif defined(RAJA_ENABLE_HIP)
        RAJA::statement::HipKernelOccAsync<
#endif
          RAJA::statement::Tile<1, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_y_policy,
            RAJA::statement::Tile<0, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_x_policy,
              RAJA::statement::For<1, gpu_thread_y_policy,
                RAJA::statement::For<0, gpu_thread_x_policy,
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
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(RAJA_KERNEL_1_TILED_BOUNDS)
  {
    printf("# 23");
    printf("\nRAJA Kernel 1 (w launch bounds) \n");
    printf("Uses RAJA methods to perform tiling\n");
    printf("Adds launch bounds \n");
    auto t0 = Clock::now();

    using EXEC_POL5 =
      RAJA::KernelPolicy<
#if defined(RAJA_ENABLE_CUDA)
        RAJA::statement::CudaKernelFixedAsync<DEVICE_BLOCK_SIZE*DEVICE_BLOCK_SIZE,
#elif defined(RAJA_ENABLE_HIP)
        RAJA::statement::HipKernelFixedAsync<DEVICE_BLOCK_SIZE*DEVICE_BLOCK_SIZE,
#endif
          RAJA::statement::Tile<1, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_y_policy,
            RAJA::statement::Tile<0, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_x_policy,
              RAJA::statement::For<1, gpu_thread_y_policy,
                RAJA::statement::For<0, gpu_thread_x_policy,
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
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(RAJA_KERNEL_2_TILED_SHARED)
  {
    printf("# 24");
    printf("\nRAJA Kernel 2 shared \n");
    printf("Uses RAJA methods to break up the iteration space into tiles\n");
    printf("Performs multiplication by loading submatrices into tiles \n");
    printf("Solution is accumulated in shared memory \n");
    auto t0 = Clock::now();

    using Shmem      = RAJA::LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<DEVICE_BLOCK_SIZE, DEVICE_BLOCK_SIZE>>;

    using shmem_Lambda0 = RAJA::statement::Lambda<0, RAJA::Offsets<0, 2>, RAJA::Params<2>>;
    using shmem_Lambda1 = RAJA::statement::Lambda<1, RAJA::Segs<0, 1>, RAJA::Offsets<0, 1>, RAJA::Params<0>>;
    using shmem_Lambda2 = RAJA::statement::Lambda<2, RAJA::Segs<1, 2>, RAJA::Offsets<1, 2>, RAJA::Params<1>>;
    using shmem_Lambda3 = RAJA::statement::Lambda<3, RAJA::Offsets<0, 1, 2>, RAJA::Params<0, 1, 2>>;
    using shmem_Lambda4 = RAJA::statement::Lambda<4, RAJA::Segs<0, 2>, RAJA::Offsets<0, 2>, RAJA::Params<2>>;

    using EXEC_POL10 =
    RAJA::KernelPolicy<
#if defined(RAJA_ENABLE_CUDA)
      RAJA::statement::CudaKernelOccAsync<
#elif defined(RAJA_ENABLE_HIP)
     RAJA::statement::HipKernelOccAsync<
#endif
        //Initalize thread private value
#if defined(RAJA_ENABLE_CUDA)
     RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<2,1,0>,
#elif defined(RAJA_ENABLE_HIP)
        RAJA::statement::InitLocalMem<RAJA::hip_shared_mem, RAJA::ParamList<2,1,0>,
#endif
          // Tile rows and cols of C (the result matrix C)
          RAJA::statement::Tile<0, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_y_policy,
            RAJA::statement::Tile<2, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_x_policy,

              // zero out dot
              RAJA::statement::For<0, gpu_thread_y_policy,
                RAJA::statement::For<2, gpu_thread_x_policy,
                  shmem_Lambda0
                >
              >,

              // Slide window across matrix: Load tiles of global matrices A, B and compute
              // local dot products
              RAJA::statement::Tile<1, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, RAJA::loop_exec,

                // Load tile of A into shmem
                RAJA::statement::For<0, gpu_thread_y_policy,
                  RAJA::statement::For<1, gpu_thread_x_policy,
                    shmem_Lambda1
                  >
                >,

                // Load tile of B into shmem
                RAJA::statement::For<1, gpu_thread_y_policy,
                  RAJA::statement::For<2, gpu_thread_x_policy,
                    shmem_Lambda2
                  >
                >,

                kernel_sync_threads,

                //Partial multiplication
                RAJA::statement::For<1, RAJA::loop_exec,
                  RAJA::statement::For<0, gpu_thread_y_policy,
                    RAJA::statement::For<2, gpu_thread_x_policy,
                      shmem_Lambda3
                    >
                  >
                >,

                kernel_sync_threads

              >, //sliding window

              //Write memory out to global matrix
              RAJA::statement::For<0, gpu_thread_y_policy,
                RAJA::statement::For<2, gpu_thread_x_policy,
                  shmem_Lambda4
                >
              >
            >
          >
        > //Create shared memory
      >//Cuda kernel
    >;

    Shmem aShared, bShared, cShared;

    RAJA::kernel_param<EXEC_POL10>(RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, N),
                                                    RAJA::TypedRangeSegment<int>(0, N),
                                                    RAJA::TypedRangeSegment<int>(0, N)),
                                   RAJA::make_tuple(aShared, bShared, cShared),

    // Zero out thread local memory for storing dot products
    [=] RAJA_HOST_DEVICE (int tn, int tp, Shmem &cShared) {

      cShared(tn,tp) = 0.0;

    },

    // Load tile of A
    [=] RAJA_HOST_DEVICE (int n, int m, int tn, int tm, Shmem &aShared) {

      aShared(tn, tm) = Aview(n, m);

    },

    // Load tile of B
    [=] RAJA_HOST_DEVICE (int m, int p, int tm, int tp, Shmem &bShared) {

      bShared(tm, tp) = Bview(m, p);

    },

    // Do partial update in shmem
    [=] RAJA_HOST_DEVICE (int tn, int tm, int tp, Shmem &aShared,  Shmem &bShared, Shmem & cShared) {

      cShared(tn,tp) += aShared(tn,tm) * bShared(tm, tp);

    },

    // Write out complete result
    [=] RAJA_HOST_DEVICE (int n, int p, int tn, int tp, Shmem &cShared) {

      Cview(n,p) = cShared(tn,tp);

    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(RAJA_KERNEL_2_TILED_SHARED_BOUNDS)
  {
    printf("# 25");
    printf("\nRAJA Kernel 2 shared (with launch bounds) \n");
    printf("Uses RAJA methods to break up the iteration space into tiles\n");
    printf("Performs multiplication within tiles \n");
    printf("Solution is accumulated in shared memory \n");
    printf("Adds launch bounds \n");
    auto t0 = Clock::now();

    using Shmem      = RAJA::LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<DEVICE_BLOCK_SIZE, DEVICE_BLOCK_SIZE>>;

    using shmem_Lambda0 = RAJA::statement::Lambda<0, RAJA::Offsets<0, 2>, RAJA::Params<2>>;
    using shmem_Lambda1 = RAJA::statement::Lambda<1, RAJA::Segs<0, 1>, RAJA::Offsets<0, 1>, RAJA::Params<0>>;
    using shmem_Lambda2 = RAJA::statement::Lambda<2, RAJA::Segs<1, 2>, RAJA::Offsets<1, 2>, RAJA::Params<1>>;
    using shmem_Lambda3 = RAJA::statement::Lambda<3, RAJA::Offsets<0, 1, 2>, RAJA::Params<0, 1, 2>>;
    using shmem_Lambda4 = RAJA::statement::Lambda<4, RAJA::Segs<0, 2>, RAJA::Offsets<0, 2>, RAJA::Params<2>>;

    using EXEC_POL10 =
    RAJA::KernelPolicy<
#if defined (RAJA_ENABLE_CUDA)
      RAJA::statement::CudaKernelFixedAsync<DEVICE_BLOCK_SIZE*DEVICE_BLOCK_SIZE,
#elif defined(RAJA_ENABLE_HIP)
    RAJA::statement::HipKernelFixedAsync<DEVICE_BLOCK_SIZE*DEVICE_BLOCK_SIZE,
#endif

        //Initalize thread private value
        RAJA::statement::InitLocalMem<gpu_shared_mem, RAJA::ParamList<2,1,0>,

          // Tile rows and cols of C (the result matrix C)
          RAJA::statement::Tile<0, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_y_policy,
            RAJA::statement::Tile<2, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_x_policy,

              // zero out dot
              RAJA::statement::For<0, gpu_thread_y_policy,
                RAJA::statement::For<2, gpu_thread_x_policy,
                  shmem_Lambda0
                >
              >,

              // Slide window across matrix: Load tiles of global matrices A, B and compute
              // local dot products
              RAJA::statement::Tile<1, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, RAJA::loop_exec,

                // Load tile of A into shmem
                RAJA::statement::For<0, gpu_thread_y_policy,
                  RAJA::statement::For<1, gpu_thread_x_policy,
                    shmem_Lambda1
                  >
                >,

                // Load tile of B into shmem
                RAJA::statement::For<1, gpu_thread_y_policy,
                  RAJA::statement::For<2, gpu_thread_x_policy,
                    shmem_Lambda2
                  >
                >,

                kernel_sync_threads,

                //Partial multiplication
                RAJA::statement::For<1, RAJA::loop_exec,
                  RAJA::statement::For<0, gpu_thread_y_policy,
                    RAJA::statement::For<2, gpu_thread_x_policy,
                      shmem_Lambda3
                    >
                  >
                >,

                kernel_sync_threads

              >, //sliding window

              //Write memory out to global matrix
              RAJA::statement::For<0, gpu_thread_y_policy,
                RAJA::statement::For<2, gpu_thread_x_policy,
                  shmem_Lambda4
                >
              >
            >
          >
        > //Create shared memory
      >//Cuda kernel
    >;

    Shmem aShared, bShared, cShared;

    RAJA::kernel_param<EXEC_POL10>(RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, N),
                                                    RAJA::TypedRangeSegment<int>(0, N),
                                                    RAJA::TypedRangeSegment<int>(0, N)),
                                   RAJA::make_tuple(aShared, bShared, cShared),

    // Zero out thread local memory for storing dot products
    [=] RAJA_HOST_DEVICE (int tn, int tp, Shmem &cShared) {

      cShared(tn,tp) = 0.0;

    },

    // Load tile of A
    [=] RAJA_HOST_DEVICE (int n, int m, int tn, int tm, Shmem &aShared) {

      aShared(tn, tm) = Aview(n, m);

    },

    // Load tile of B
    [=] RAJA_HOST_DEVICE (int m, int p, int tm, int tp, Shmem &bShared) {

      bShared(tm, tp) = Bview(m, p);

    },

    // Do partial update in shmem
    [=] RAJA_HOST_DEVICE (int tn, int tm, int tp, Shmem &aShared,  Shmem &bShared, Shmem & cShared) {

      cShared(tn,tp) += aShared(tn,tm) * bShared(tm, tp);

    },

    // Write out complete result
    [=] RAJA_HOST_DEVICE (int n, int p, int tn, int tp, Shmem &cShared) {

      Cview(n,p) = cShared(tn,tp);

    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(RAJA_KERNEL_2_TILED_SHARED_REG)
  {
    printf("# 26");
    printf("\nRAJA Kernel 2 + reg \n");
    printf("Uses RAJA methods to break up the iteration space into tiles\n");
    printf("Performs multiplication within tiles \n");
    printf("Solution is accumulated in register \n");
    auto t0 = Clock::now();

    using Shmem      = RAJA::LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<DEVICE_BLOCK_SIZE, DEVICE_BLOCK_SIZE>>;

    using shmem_Lambda0 = RAJA::statement::Lambda<0, RAJA::Params<2>>;
    using shmem_Lambda1 = RAJA::statement::Lambda<1, RAJA::Segs<0, 1>, RAJA::Offsets<0, 1>, RAJA::Params<0>>;
    using shmem_Lambda2 = RAJA::statement::Lambda<2, RAJA::Segs<1, 2>, RAJA::Offsets<1, 2>, RAJA::Params<1>>;
    using shmem_Lambda3 = RAJA::statement::Lambda<3, RAJA::Offsets<0, 1, 2>, RAJA::Params<0, 1, 2>>;
    using shmem_Lambda4 = RAJA::statement::Lambda<4, RAJA::Segs<0, 2>, RAJA::Offsets<0, 2>, RAJA::Params<2>>;

    using EXEC_POL10 =
    RAJA::KernelPolicy<
#if defined (RAJA_ENABLE_CUDA)
      RAJA::statement::CudaKernelOccAsync<
#elif defined(RAJA_ENABLE_HIP)
      RAJA::statement::HipKernelOccAsync<
#endif
        //Initalize thread private value
        RAJA::statement::InitLocalMem<gpu_shared_mem, RAJA::ParamList<1,0>,

          // Tile rows and cols of C (the result matrix C)
          RAJA::statement::Tile<0, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_y_policy,
            RAJA::statement::Tile<2, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_x_policy,

              // zero out dot
              shmem_Lambda0,

              // Slide window across matrix: Load tiles of global matrices A, B and compute
              // local dot products
              RAJA::statement::Tile<1, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, RAJA::loop_exec,

                // Load tile of A into shmem
                RAJA::statement::For<0, gpu_thread_y_policy,
                  RAJA::statement::For<1, gpu_thread_x_policy,
                    shmem_Lambda1
                  >
                >,

                // Load tile of B into shmem
                RAJA::statement::For<1, gpu_thread_y_policy,
                  RAJA::statement::For<2, gpu_thread_x_policy,
                    shmem_Lambda2
                  >
                >,

                kernel_sync_threads,

                //Partial multiplication
                RAJA::statement::For<1, RAJA::loop_exec,
                  RAJA::statement::For<0, gpu_thread_y_policy,
                    RAJA::statement::For<2, gpu_thread_x_policy,
                      shmem_Lambda3
                    >
                  >
                >,

                kernel_sync_threads

              >, //sliding window

              //Write memory out to global matrix
              RAJA::statement::For<0, gpu_thread_y_policy,
                RAJA::statement::For<2, gpu_thread_x_policy,
                  shmem_Lambda4
                >
              >
            >
          >
        > //Create shared memory
      >//Cuda kernel
    >;

    Shmem aShared, bShared, cShared;

    RAJA::kernel_param<EXEC_POL10>(RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, N),
                                                    RAJA::TypedRangeSegment<int>(0, N),
                                                    RAJA::TypedRangeSegment<int>(0, N)),
                                   RAJA::make_tuple(aShared, bShared, 0.0),

    // Zero out thread local memory for storing dot products
    [=] RAJA_HOST_DEVICE (double &dot) {

      dot = 0.0;

    },

    // Load tile of A
    [=] RAJA_HOST_DEVICE (int n, int m, int tn, int tm, Shmem &aShared) {

      aShared(tn, tm) = Aview(n, m);

    },

    // Load tile of B
    [=] RAJA_HOST_DEVICE (int m, int p, int tm, int tp, Shmem &bShared) {

      bShared(tm, tp) = Bview(m, p);

    },

    // Do partial update in shmem
    [=] RAJA_HOST_DEVICE (int tn, int tm, int tp, Shmem &aShared,  Shmem &bShared, double &dot) {

      dot += aShared(tn,tm) * bShared(tm, tp);

    },

    // Write out complete result
    [=] RAJA_HOST_DEVICE (int n, int p, int tn, int tp, double &dot) {

      Cview(n,p) = dot;

    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#if defined(RAJA_KERNEL_2_TILED_SHARED_REG_BOUNDS)
  {
    printf("# 27");
    printf("\nRAJA Kernel 2 + register (with launch bounds) \n");
    printf("Uses RAJA methods to break up the iteration space into tiles\n");
    printf("Performs multiplication by loading submatrices into tiles \n");
    printf("Solution is accumulated in register \n");
    auto t0 = Clock::now();

    using Shmem      = RAJA::LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<DEVICE_BLOCK_SIZE, DEVICE_BLOCK_SIZE>>;

    using shmem_Lambda0 = RAJA::statement::Lambda<0, RAJA::Params<2>>;
    using shmem_Lambda1 = RAJA::statement::Lambda<1, RAJA::Segs<0, 1>, RAJA::Offsets<0, 1>, RAJA::Params<0>>;
    using shmem_Lambda2 = RAJA::statement::Lambda<2, RAJA::Segs<1, 2>, RAJA::Offsets<1, 2>, RAJA::Params<1>>;
    using shmem_Lambda3 = RAJA::statement::Lambda<3, RAJA::Offsets<0, 1, 2>, RAJA::Params<0, 1, 2>>;
    using shmem_Lambda4 = RAJA::statement::Lambda<4, RAJA::Segs<0, 2>, RAJA::Offsets<0, 2>, RAJA::Params<2>>;

    using EXEC_POL10 =
    RAJA::KernelPolicy<
#if defined (RAJA_ENABLE_CUDA)
      RAJA::statement::CudaKernelFixedAsync<DEVICE_BLOCK_SIZE*DEVICE_BLOCK_SIZE,
#elif defined(RAJA_ENABLE_HIP)
      RAJA::statement::HipKernelFixedAsync<DEVICE_BLOCK_SIZE*DEVICE_BLOCK_SIZE,
#endif
        //Initalize thread private value
        RAJA::statement::InitLocalMem<gpu_shared_mem, RAJA::ParamList<1,0>,

          // Tile rows and cols of C (the result matrix C)
          RAJA::statement::Tile<0, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_y_policy,
            RAJA::statement::Tile<2, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, gpu_block_x_policy,

              // zero out dot
              shmem_Lambda0,

              // Slide window across matrix: Load tiles of global matrices A, B and compute
              // local dot products
              RAJA::statement::Tile<1, RAJA::tile_fixed<DEVICE_BLOCK_SIZE>, RAJA::loop_exec,

                // Load tile of A into shmem
                RAJA::statement::For<0, gpu_thread_y_policy,
                  RAJA::statement::For<1, gpu_thread_x_policy,
                    shmem_Lambda1
                  >
                >,

                // Load tile of B into shmem
                RAJA::statement::For<1, gpu_thread_y_policy,
                  RAJA::statement::For<2, gpu_thread_x_policy,
                    shmem_Lambda2
                  >
                >,

                kernel_sync_threads,

                //Partial multiplication
                RAJA::statement::For<1, RAJA::loop_exec,
                  RAJA::statement::For<0, gpu_thread_y_policy,
                    RAJA::statement::For<2, gpu_thread_x_policy,
                      shmem_Lambda3
                    >
                  >
                >,

                kernel_sync_threads

              >, //sliding window

              //Write memory out to global matrix
              RAJA::statement::For<0, gpu_thread_y_policy,
                RAJA::statement::For<2, gpu_thread_x_policy,
                  shmem_Lambda4
                >
              >
            >
          >
        > //Create shared memory
      >//Cuda kernel
    >;

    Shmem aShared, bShared, cShared;

    RAJA::kernel_param<EXEC_POL10>(RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, N),
                                                    RAJA::TypedRangeSegment<int>(0, N),
                                                    RAJA::TypedRangeSegment<int>(0, N)),
                                   RAJA::make_tuple(aShared, bShared, 0.0),

    // Zero out thread local memory for storing dot products
    [=] RAJA_HOST_DEVICE (double &dot) {

      dot = 0.0;

    },

    // Load tile of A
    [=] RAJA_HOST_DEVICE (int n, int m, int tn, int tm, Shmem &aShared) {

      aShared(tn, tm) = Aview(n, m);

    },

    // Load tile of B
    [=] RAJA_HOST_DEVICE (int m, int p, int tm, int tp, Shmem &bShared) {

      bShared(tm, tp) = Bview(m, p);

    },

    // Do partial update in shmem
    [=] RAJA_HOST_DEVICE (int tn, int tm, int tp, Shmem &aShared,  Shmem &bShared, double &dot) {

      dot += aShared(tn,tm) * bShared(tm, tp);

    },

    // Write out complete result
    [=] RAJA_HOST_DEVICE (int n, int p, int tn, int tp, double &dot) {

      Cview(n,p) = dot;

    });
    DEVICE_SYNC();
    auto tf = Clock::now();

    std::cout << "Delta t0-tf: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0).count()
              << " milliseconds" << std::endl;
    checkResult(C, N);
  }
#endif

#endif//ENABLE RAJA kernel

//
// Clean up.
//
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);

  std::cout << "\n DONE!...\n";

  return 0;
}


template <typename T>
__global__ void checkKernel(int N, const T* C, int* match)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ( row < N && col < N ) {
    if ( abs( C(row, col) -  N ) > 10e-12 ) {
      atomicExch(match, 0);
    }
  }
}

template <typename T>
__global__ void checkViewKernel(int N, RAJA::View<T, RAJA::Layout<DIM>> Cview, int* match)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ( row < N && col < N ) {
    if ( abs( Cview(row, col) -  N ) > 10e-12 ) {
      atomicExch(match, 0);
    }
  }
}

//
// Functions to check result and report P/F.
//
template <typename T>
void checkResult(T* C, int N)
{
  int *match = memoryManager::allocate<int>(1);
  *match = 1;
  dim3 blockdim(DEVICE_BLOCK_SIZE, DEVICE_BLOCK_SIZE);
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));
  checkKernel<<<griddim, blockdim>>>(N, C, match);
  DEVICE_SYNC();
  if ( *match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
  memoryManager::deallocate(match);
};

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  int *match = memoryManager::allocate<int>(1);
  *match = 1;
  dim3 blockdim(DEVICE_BLOCK_SIZE, DEVICE_BLOCK_SIZE);
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));
  checkViewKernel<<<griddim, blockdim>>>(N, Cview, match);
  DEVICE_SYNC();
  if ( *match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
  memoryManager::deallocate(match);
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
