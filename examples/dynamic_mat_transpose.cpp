//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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
 *  Matrix Transpose Example
 *
 *  In this example, an input matrix A of dimension N_r x N_c is
 *  transposed and returned as a second matrix At of size N_c x N_r.
 *
 *  This operation is carried out using a local memory tiling
 *  algorithm. The algorithm first loads matrix entries into an
 *  iteraion shared tile, a two-dimensional array, and then
 *  reads from the tile with row and column indices swapped for
 *  the output matrix.
 *
 *  The algorithm is expressed as a collection of ``outer``
 *  and ``inner`` for loops. Iterations of the inner loops will load/read
 *  data into the tile; while outer loops will iterate over the number
 *  of tiles needed to carry out the transpose.
 *
 *  RAJA variants of the example use RAJA dynamic shared memory as tile memory.
 *  RAJA shared memory is mapped to device shared memory which
 *  enables threads in the same thread block to share data.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::launch' abstractions for nested loops
 *    - Hierachial parallism
 *    - Dynamic shared memory
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

//
// Define dimensionality of matrices
//
const int DIM = 2;

//
// Function for checking results
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c);

//
// Function for printing results
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c);

using launch_policy = RAJA::LaunchPolicy<
#if defined(RAJA_ENABLE_OPENMP)
    RAJA::omp_launch_t
#else
    RAJA::seq_launch_t
#endif
#if defined(RAJA_ENABLE_CUDA)
    ,
    RAJA::cuda_launch_t<false>
#endif
#if defined(RAJA_ENABLE_HIP)
    ,
    RAJA::hip_launch_t<false>
#endif
    >;

/*
 * Define team policies.
 * Up to 3 dimension are supported: x,y,z
 */
using outer0 = RAJA::LoopPolicy<
                                       RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                       ,
                                       RAJA::cuda_block_x_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                       ,
                                       RAJA::hip_block_x_direct
#endif
#if defined(RAJA_ENABLE_SYCL)
                                       ,
                                       RAJA::sycl_group_0_direct
#endif
                                       >;

using outer1 = RAJA::LoopPolicy<
                                       RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                       ,
                                       RAJA::cuda_block_y_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                       ,
                                       RAJA::hip_block_y_direct
#endif
#if defined(RAJA_ENABLE_SYCL)
                                       ,
                                       RAJA::sycl_group_1_direct
#endif
                                       >;
/*
 * Define thread policies.
 * Up to 3 dimension are supported: x,y,z
 */
using inner0 = RAJA::LoopPolicy<
#if defined(RAJA_ENABLE_OPENMP)
                                         RAJA::omp_for_exec
#else
                                         RAJA::loop_exec
#endif

#if defined(RAJA_ENABLE_CUDA)
                                         ,
                                         RAJA::cuda_thread_x_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                         ,
                                         RAJA::hip_thread_x_direct
#endif
#if defined(RAJA_ENABLE_SYCL)
                                        ,
                                         RAJA::sycl_local_0_direct
#endif
                                         >;

using inner1 = RAJA::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_ENABLE_CUDA)
                                         ,
                                         RAJA::cuda_thread_y_direct
#endif
#if defined(RAJA_ENABLE_HIP)
                                         ,
                                         RAJA::hip_thread_y_direct
#endif
#if defined(RAJA_ENABLE_SYCL)
                                        ,
                                         RAJA::sycl_local_1_direct
#endif
                                         >;

template<typename T>
void switch_ptrs(T *A, T *d_A)
{
  T *tmp_ptr;
  tmp_ptr = d_A;
  d_A = A;
  A = tmp_ptr;
}

int main(int argc, char *argv[])
{

  std::cout << "\n\nRAJA matrix transpose example...\n";

  if(argc != 2) {
    RAJA_ABORT_OR_THROW("Usage ./launch_reductions host or ./launch_reductions device");
  }

  //
  // Run time policy section is demonstrated in this example by specifying
  // kernel exection space as a command line argument (host or device).
  // Example usage ./launch_reductions host or ./launch_reductions device
  //
  std::string exec_space = argv[1];
  if(!(exec_space.compare("host") == 0 || exec_space.compare("device") == 0 )){
    RAJA_ABORT_OR_THROW("Usage ./launch_reductions host or ./launch_reductions device");
    return 0;
  }

  RAJA::ExecPlace select_cpu_or_gpu;
  if(exec_space.compare("host") == 0)
    { select_cpu_or_gpu = RAJA::HOST; printf("Running RAJA-Launch reductions example on the host \n"); }
  if(exec_space.compare("device") == 0)
    { select_cpu_or_gpu = RAJA::DEVICE; printf("Running RAJA-Launch reductions example on the device \n"); }



#if defined(RAJA_ENABLE_SYCL)
  memoryManager::sycl_res = new camp::resources::Resource{camp::resources::Sycl()};
  ::RAJA::sycl::detail::setQueue(memoryManager::sycl_res);
#endif

  //
  // Define num rows/cols in matrix, tile dimensions, and number of tiles
  //
  // _mattranspose_localarray_dims_start
  const int N_r = 267;
  const int N_c = 251;

  const int TILE_DIM = 16;

  const int outer_Dimc = (N_c - 1) / TILE_DIM + 1;
  const int outer_Dimr = (N_r - 1) / TILE_DIM + 1;
  // _mattranspose_localarray_dims_end

  //
  // Allocate matrix data
  //
  int *A = memoryManager::allocate<int>(N_r * N_c);
  int *At = memoryManager::allocate<int>(N_r * N_c);

  //
  // In the following implementations of matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  // _mattranspose_localarray_views_start
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_r, N_c);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_c, N_r);
  // _mattranspose_localarray_views_end

  //
  // Initialize matrix data
  //
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      Aview(row, col) = col;
    }
  }
  //printResult<int>(Aview, N_r, N_c);

  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of shared matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  // _mattranspose_localarray_cstyle_start
  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dimr; ++by) {
    for (int bx = 0; bx < outer_Dimc; ++bx) {

      // Stack-allocated local array for data on a tile
      int Tile[TILE_DIM][TILE_DIM];

      //
      // (1) Inner loops to read input matrix tile data into the array
      //
      //     Note: loops are ordered so that input matrix data access
      //           is stride-1.
      //
      for (int ty = 0; ty < TILE_DIM; ++ty) {
        for (int tx = 0; tx < TILE_DIM; ++tx) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Tile[ty][tx] = Aview(row, col);
          }
        }
      }

      //
      // (2) Inner loops to write array data into output array tile
      //
      //     Note: loop order is swapped from above so that output matrix
      //           data access is stride-1.
      //
      for (int tx = 0; tx < TILE_DIM; ++tx) {
        for (int ty = 0; ty < TILE_DIM; ++ty) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Atview(col, row) = Tile[ty][tx];
          }
        }
      }

    }
  }
  // _mattranspose_localarray_cstyle_end

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

  //----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA matrix transpose w/ dynamic shared memory ...\n";

  int *d_A = nullptr, *d_At = nullptr;

#if defined(RAJA_ENABLE_HIP)
  if(select_cpu_or_gpu == RAJA::DEVICE) {
    d_A =  memoryManager::allocate_gpu<int>(N_r * N_c);
    d_At = memoryManager::allocate_gpu<int>(N_r * N_c);

    hipErrchk(hipMemcpy( d_A, A, N_r * N_c * sizeof(int), hipMemcpyHostToDevice ));

    //switch host/device pointers so we can reuse the views
    switch_ptrs(d_A, A);
    switch_ptrs(d_At, At);
  }
#endif


  constexpr size_t dynamic_shared_mem = TILE_DIM * TILE_DIM * sizeof(int);

  RAJA::launch<launch_policy>(select_cpu_or_gpu, dynamic_shared_mem,
    RAJA::Grid(RAJA::Teams(outer_Dimr, outer_Dimc),
                     RAJA::Threads(TILE_DIM, TILE_DIM)),
       [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx)
  {

    RAJA::loop<outer1>(ctx, RAJA::RangeSegment(0, outer_Dimr), [&] (int by){
        RAJA::loop<outer0>(ctx, RAJA::RangeSegment(0, outer_Dimc), [&] (int bx){


            auto Tile_1 = ctx.getSharedMemoryView<int, 2, int, 1>(TILE_DIM*TILE_DIM, TILE_DIM, TILE_DIM);

            RAJA::loop<inner1>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&] (int ty){
              RAJA::loop<inner0>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&] (int tx){

                  int col = bx * TILE_DIM + tx;  // Matrix column index
                  int row = by * TILE_DIM + ty;  // Matrix row index

                  // Bounds check
                  if (row < N_r && col < N_c) {
                    Tile_1(ty,tx) = Aview(row, col);
                  }

                });
              });

            //need a barrier
            ctx.teamSync();

            RAJA::loop<inner1>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&] (int ty){
              RAJA::loop<inner0>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&] (int tx){

                  int col = bx * TILE_DIM + tx;  // Matrix column index
                  int row = by * TILE_DIM + ty;  // Matrix row index

                  // Bounds check
                  if (row < N_r && col < N_c) {
                    Atview(col, row) = Tile_1(ty, tx);
                  }

                });
              });

            ctx.releaseSharedMemory();
          });
      });

  });

#if defined(RAJA_ENABLE_HIP)
  if(select_cpu_or_gpu == RAJA::DEVICE) {
    switch_ptrs(d_At, At);
    switch_ptrs(d_A, A);

    hipErrchk(hipMemcpy( d_At, At, N_r * N_c * sizeof(int), hipMemcpyDeviceToHost ));
  }
#endif


  checkResult<int>(Atview, N_c, N_r);
  //----------------------------------------------------------------------------//

  return 0;
}


//
// Function to check result and report P/F.
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c)
{
  bool match = true;
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      if (Atview(row, col) != row) {
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
// Function to print result.
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_r, int N_c)
{
  std::cout << std::endl;
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      //std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
      //<< std::endl;
      printf("%d ",Atview(row, col));
    }
    std::cout << "" << std::endl;
  }
  std::cout << std::endl;
}
