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
 *  RAJA variants of the example use RAJA local arrays as tile memory.
 *  Furthermore, the tiling pattern is handled by RAJA's tile statements.
 *  For CPU execution, RAJA local arrays are used to improve
 *  performance via cache blocking. For CUDA GPU execution,
 *  RAJA shared memory is mapped to CUDA shared memory which
 *  enables threads in the same thread block to share data.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *       - Multiple lambdas
 *       - Options for specifying lambda arguments
 *       - Tile statement
 *       - ForICount statement
 *       - RAJA local arrays
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


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA shared matrix transpose example...\n";

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

#if defined(RAJA_ENABLE_SYCL)
  std::cout << "\n Running RAJA SYCL matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  int *d_a = memoryManager::allocate_gpu<int>(N_r * N_c);
  int *d_at = memoryManager::allocate_gpu<int>(N_r * N_c);

  memoryManager::sycl_res->memcpy(d_a, A, N_r * N_c * sizeof(int));
  memoryManager::sycl_res->memcpy(d_at, At, N_r * N_c * sizeof(int));

  //Device views
  RAJA::View<int, RAJA::Layout<DIM>> A_(d_a, N_r, N_c);
  RAJA::View<int, RAJA::Layout<DIM>> At_(d_at, N_c, N_r);

  /*
  using launch_policy =
    RAJA::LaunchPolicy<
#if defined(RAJA_DEVICE_ACTIVE)
      RAJA::seq_launch_t,
#endif
      RAJA::sycl_launch_t<false>>;

  using inner0 =
    RAJA::LoopPolicy<
#if defined(RAJA_DEVICE_ACTIVE)
      RAJA::loop_exec,
#endif
      RAJA::sycl_local_0_direct>;

  using inner1 =
    RAJA::LoopPolicy<
#if defined(RAJA_DEVICE_ACTIVE)
      RAJA::loop_exec,
#endif
      RAJA::sycl_local_1_direct>;

  using outer0 =
    RAJA::LoopPolicy<
#if defined(RAJA_DEVICE_ACTIVE)
      RAJA::loop_exec,
#endif
    RAJA::sycl_group_0_direct>;

  using outer1 =
    RAJA::LoopPolicy<
#if defined(RAJA_DEVICE_ACTIVE)
      RAJA::loop_exec,
#endif
      RAJA::sycl_group_1_direct>;
  */

    using launch_policy =
    RAJA::LaunchPolicy<
      RAJA::sycl_launch_t<false>>;

  using inner0 =
    RAJA::LoopPolicy<RAJA::sycl_local_0_direct>;

  using inner1 =
    RAJA::LoopPolicy<RAJA::sycl_local_1_direct>;

  using outer0 =
    RAJA::LoopPolicy<RAJA::sycl_group_0_direct>;

  using outer1 =
    RAJA::LoopPolicy<RAJA::sycl_group_1_direct>;



   //This kernel will require the following amount of shared memory
   const size_t shared_memory_size = 2*TILE_DIM*TILE_DIM*sizeof(int);

   //move shared memory arg to launch 2nd arg
   RAJA::launch<launch_policy>
     (shared_memory_size,
      RAJA::Grid(RAJA::Teams(outer_Dimc, outer_Dimr),
		       RAJA::Threads(TILE_DIM, TILE_DIM)),
      [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx) {

       RAJA::loop<outer1>(ctx, RAJA::RangeSegment(0, outer_Dimr), [&] (int by){
         RAJA::loop<outer0>(ctx, RAJA::RangeSegment(0, outer_Dimc), [&] (int bx){

               //ctx points to a a large chunk of memory
               //getSharedMemory will apply the correct offsetting
	       //Consider templating on size to enable stack allocations on the CPU
               int *tile_1_mem = ctx.getSharedMemory<int>(TILE_DIM*TILE_DIM);
	       int *tile_2_mem = ctx.getSharedMemory<int>(TILE_DIM*TILE_DIM);

	       //consider a getSharedMemoryView method


               //reshape the data
               int (*Tile_2)[TILE_DIM] = (int (*)[TILE_DIM]) (tile_2_mem);
	       //Use RAJA view

               RAJA::loop<inner1>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&] (int ty){
                   RAJA::loop<inner0>(ctx, RAJA::RangeSegment(0, TILE_DIM), [&] (int tx){

                       int col = bx * TILE_DIM + tx;  // Matrix column index
                       int row = by * TILE_DIM + ty;  // Matrix row index

                       // Bounds check
                       if (row < N_r && col < N_c) {
                         Tile_2[ty][tx] = A_(row, col);
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
                         At_(col, row) = Tile_2[ty][tx];
                       }

                     });

                 });

             });
         });

     });

  memoryManager::sycl_res->memcpy(At, d_at, N_c * N_r * sizeof(int));

  checkResult<int>(Atview, N_c, N_r);
//printResult<int>(Atview, N_c, N_r);

#endif


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