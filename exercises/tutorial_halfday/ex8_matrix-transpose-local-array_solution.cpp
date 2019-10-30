//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
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
 *  EXERCISE #8: Matrix Transpose with Local Array
 *
 *  In this exercise, your program will carry out the
 *  transpose of a matrix A using a tiling algorithm and RAJA local array.
 *  Unlike the previous exercise, each tile will be stored within
 *  a RAJA local array. As part of the exercise you will have to provide
 *  the transpose as a second matrix At.
 *
 *  This file contains a C-style variant of the algorithm as well as the
 *  RAJA kernel for a RAJA variant. You will have to
 *  implement the RAJA policy for the sequential, and OpenMP variants.
 *  If you have access to a GPU and a CUDA compiler, try using constructing
 *  the CUDA policy.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *       - Multiple lambdas
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

  //
  // Define num rows/cols in matrix
  //
  const int N_r = 267;
  const int N_c = 251;

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
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_r, N_c);

  //
  //Construct a permuted layout such that the column index has stride 1
  //
  RAJA::Layout<2> perm_layout = RAJA::make_permuted_layout({{N_c, N_r}}, std::array<RAJA::idx_t, 2>{{1, 0}});
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, perm_layout);

  //
  // Define TILE size (TILE_SZ x TILE_SZ)
  //
  const int TILE_SZ = 16;

  // Calculate number of tiles (Needed for C++ version)
  const int outer_Dimc = (N_c - 1) / TILE_SZ + 1;
  const int outer_Dimr = (N_r - 1) / TILE_SZ + 1;

  //
  // Initialize matrix data
  //
  for (int row = 0; row < N_r; ++row) {
    for (int col = 0; col < N_c; ++col) {
      Aview(row, col) = col;
    }
  }
  // printResult<int>(Aview, N_r, N_c);

//----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of shared matrix transpose...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dimr; ++by) {
    for (int bx = 0; bx < outer_Dimc; ++bx) {

      // Stack-allocated local array for data on a tile
      int Tile[TILE_SZ][TILE_SZ];

      //
      // (1) Inner loops to read input matrix tile data into the array
      //
      //     Note: loops are ordered so that input matrix data access
      //           is stride-1.
      //
      for (int trow = 0; trow < TILE_SZ; ++trow) {
        for (int tcol = 0; tcol < TILE_SZ; ++tcol) {

          int col = bx * TILE_SZ + tcol;  // Matrix column index
          int row = by * TILE_SZ + trow;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Tile[trow][tcol] = Aview(row, col);
          }
        }
      }

      //
      // (2) Inner loops to write array data into output array tile
      //
      //     Note: loop order is swapped from above so that output matrix
      //           data access is stride-1.
      //
      for (int tcol = 0; tcol < TILE_SZ; ++tcol) {
        for (int trow = 0; trow < TILE_SZ; ++trow) {

          int col = bx * TILE_SZ + tcol;  // Matrix column index
          int row = by * TILE_SZ + trow;  // Matrix row index

          // Bounds check
          if (row < N_r && col < N_c) {
            Atview(col, row) = Tile[trow][tcol];
          }
        }
      }

    }
  }
  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

//----------------------------------------------------------------------------//

  //
  // The following RAJA variants use the RAJA::Kernel
  // method to carryout the transpose
  //

  // Here we define a RAJA local array type.
  // The array type is templated on
  // 1) Data type
  // 2) Index permutation
  // 3) Dimensions of the array
  //

  using TILE_MEM =
    RAJA::LocalArray<int, RAJA::Perm<0, 1>, RAJA::SizeList<TILE_SZ, TILE_SZ>>;

  // **NOTE** Although the LocalArray is constructed
  // the array memory has not been allocated.

  TILE_MEM RAJA_Tile;

//--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - sequential matrix transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using SEQ_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_SZ>, RAJA::loop_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_SZ>, RAJA::loop_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<2>,

          RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::loop_exec,
            RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            >
          >,

          RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::loop_exec,
            RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >

          >
        >
      >
    >;

  RAJA::kernel_param<SEQ_EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, N_c),
                                                     RAJA::RangeSegment(0, N_r)),

    RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

    [=](int col, int row, int tcol, int trow, TILE_MEM &RAJA_Tile) {

      RAJA_Tile(trow, tcol) = Aview(row, col);

    },

    [=](int col, int row, int tcol, int trow, TILE_MEM &RAJA_Tile) {

      Atview(col, row) = RAJA_Tile(trow, tcol);

  });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

//--------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running RAJA - OpenMP (parallel outer loop) matrix "
               "transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using OPENMP_EXEC_POL =
  RAJA::KernelPolicy<
    RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_SZ>, RAJA::omp_parallel_for_exec,
      RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_SZ>, RAJA::loop_exec,

        RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<2>,

          RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::loop_exec,
            RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::loop_exec,
                                       RAJA::statement::Lambda<0>
            >
          >,

          RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::loop_exec,
            RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::loop_exec,
                                       RAJA::statement::Lambda<1>
            >
          >
        >
      >
    >
   >;

  RAJA::kernel_param<OPENMP_EXEC_POL>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N_c), RAJA::RangeSegment(0, N_r)),
      RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

      [=](int col, int row, int tcol, int trow, TILE_MEM &RAJA_Tile) {

        RAJA_Tile(trow, tcol) = Aview(row, col);

      },

      [=](int col, int row, int tcol, int trow, TILE_MEM &RAJA_Tile) {

        Atview(col, row) = RAJA_Tile(trow, tcol);

      });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
#endif


//--------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running RAJA - CUDA matrix transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using CUDA_EXEC_POL =
  RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_SZ>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_SZ>, RAJA::cuda_block_x_loop,

          RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<2>,

            RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::cuda_thread_y_direct,
              RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::cuda_thread_x_direct,
                                          RAJA::statement::Lambda<0>
              >
            >,

            RAJA::statement::CudaSyncThreads,

            RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::cuda_thread_y_direct,
              RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::cuda_thread_x_direct,
                                            RAJA::statement::Lambda<1>
              >
            >,

            RAJA::statement::CudaSyncThreads
          >
        >
      >
    >
  >;


  RAJA::kernel_param<CUDA_EXEC_POL>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N_c), RAJA::RangeSegment(0, N_r)),
      RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

      [=] RAJA_DEVICE (int col, int row, int tcol, int trow, TILE_MEM &RAJA_Tile) {

        RAJA_Tile(trow, tcol) = Aview(row, col);

      },

      [=] RAJA_DEVICE(int col, int row, int tcol, int trow, TILE_MEM &RAJA_Tile) {

        Atview(col, row) = RAJA_Tile(trow, tcol);

      });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
#endif

  //--------------------------------------------------------------------------//

  //
  // Clean up.
  //
  memoryManager::deallocate(A);
  memoryManager::deallocate(At);

  std::cout << "\n DONE!...\n";

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
        match &= false;
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
      std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
                << std::endl;
    }
    std::cout << "" << std::endl;
  }
  std::cout << std::endl;
}
