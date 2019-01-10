//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
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
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_c, N_r);

  //
  // Define TILE dimensions (TILE_DIM x TILE_DIM)
  //
  const int TILE_DIM = 16;

  // Calculate number of tiles (Needed for C++ version)
  const int outer_Dimc = (N_c - 1) / TILE_DIM + 1;
  const int outer_Dimr = (N_r - 1) / TILE_DIM + 1;

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
    RAJA::LocalArray<int, RAJA::Perm<0, 1>, RAJA::SizeList<TILE_DIM, TILE_DIM>>;

  // **NOTE** Although the LocalArray is constructed
  // the array memory has not been allocated.

  TILE_MEM RAJA_Tile;

  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - sequential matrix transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using SEQ_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

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

    [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {
      RAJA_Tile(ty, tx) = Aview(row, col);
    },

    [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {
      Atview(col, row) = RAJA_Tile(ty, tx);

  });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

#if defined(RAJA_ENABLE_OPENMP)
  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - OpenMP (parallel outer loop) matrix "
               "transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using OPENMP_EXEC_1_POL =
  RAJA::KernelPolicy<
    //
    // (0) Execution policies for outer loops
    //      These loops iterate over the number of
    //      tiles needed to carry out the transpose
    //
    RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::omp_parallel_for_exec,
      RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
        // This statement will initalize local array memory inside a
        // kernel. The cpu_tile_mem policy specifies that memory should be
        // allocated on the stack. The entries in the RAJA::ParamList
        // identify RAJA local arrays in the parameter tuple to intialize.
        RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<2>,
          //
          // (1) Execution policies for the first set of inner
          // loops. These loops copy data from the global matrices
          // to the local tile.
          //
          RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::loop_exec,
            RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::loop_exec,
                                       RAJA::statement::Lambda<0>
            >
          >,
          //
          // (2) Execution policies for the second set of inner
          // loops. These loops copy data from the local tile to
          // the global matrix.
          //     Note: The order of the loops have been
          //     swapped! This enables us to swap which
          //     index has unit stride.
          //
          RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::loop_exec,
            RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::loop_exec,
                                       RAJA::statement::Lambda<1>
            >
          >
        >
      >
    >
   >;

  RAJA::kernel_param<OPENMP_EXEC_1_POL>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N_c), RAJA::RangeSegment(0, N_r)),
      RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

      [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {

        RAJA_Tile(ty, tx) = Aview(row, col);

      },

      [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {

        Atview(col, row) = RAJA_Tile(ty, tx);

      });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);

  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - OpenMP (parallel inner loops) matrix "
               "transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

  using OPENMP_EXEC_2_POL =
  RAJA::KernelPolicy<
    //
    // (0) Execution policies for outer loops
    //      These loops iterate over the number of
    //      tiles needed to carry out the transpose
    //
    RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
      RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
      // This statement will initalize local array memory inside a
      // kernel. The cpu_tile_mem policy specifies that memory should be
      // allocated on the stack. The entries in the RAJA::ParamList
      // identify RAJA local arrays to intialize in the parameter tuple.
        RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<2>,
          //
          // (1) Execution policies for the first set of inner
          // loops. These loops copy data from the global matrices
          // to the local tile.
          //
          RAJA::statement::ForICount<1, RAJA::statement::Param<1>, RAJA::omp_parallel_for_exec,
            RAJA::statement::ForICount<0, RAJA::statement::Param<0>, RAJA::loop_exec,
                                       RAJA::statement::Lambda<0>
             >
          >,
          //
          // (2) Execution policies for the second set of inner
          // loops. These loops copy data from the local tile to
          // the global matrix.
          //     Note: The order of the loops have been
          //     swapped! This enables us to swap which
          //     index has unit stride.
          //
          RAJA::statement::ForICount<0, RAJA::statement::Param<0>, RAJA::loop_exec,
            RAJA::statement::ForICount<1, RAJA::statement::Param<1>, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >
          >
        >
      >
    >
  >;

  RAJA::kernel_param<OPENMP_EXEC_2_POL>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N_c), RAJA::RangeSegment(0, N_r)),
      RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

      [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {

        RAJA_Tile(ty, tx) = Aview(row, col);

      },

      [=](int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {

        Atview(col, row) = RAJA_Tile(ty, tx);

      });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_r, N_c);
#endif

  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - CUDA matrix transpose example ...\n";

  std::memset(At, 0, N_r * N_c * sizeof(int));

#if defined(RAJA_ENABLE_CUDA)
  using CUDA_EXEC_POL =
  RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      //
      // (0) Execution policies for outer loops
      //      These loops iterate over the number of
      //      tiles needed to carry out the transpose
      //
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::cuda_block_x_loop,
          // This statement will initalize local array memory inside a
          // kernel. The cpu_tile_mem policy specifies that memory should be
          // allocated on the stack. The entries in the RAJA::ParamList
          // identify RAJA local arrays to intialize in the parameter tuple.
          RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<2>,
            //
            // (1) Execution policies for the first set of inner
            // loops. These loops copy data from the global matrices
            // to the local tile.
            //
            RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::cuda_thread_y_direct,
              RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::cuda_thread_x_direct,
                                          RAJA::statement::Lambda<0>
              >
            >,
            // Synchronize threads to ensure all loads
            // to the local array are complete
            RAJA::statement::CudaSyncThreads,
            //
            // (2) Execution policies for the second set of inner
            // loops. These loops copy data from the local tile to
            // the global matrix.
            //     Note: The order of the loops have been
            //     swapped! This enables us to swap which
            //     index has unit stride.
            //
            RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::cuda_thread_y_direct,
              RAJA::statement::ForICount<1, RAJA::statement::Param<0>, RAJA::cuda_thread_x_direct,
                                            RAJA::statement::Lambda<1>
              >
            >,
            // Synchronize threads to ensure all reads
            // from the local array are complete
            RAJA::statement::CudaSyncThreads,
          >
        >
      >
    >
  >;


  RAJA::kernel_param<CUDA_EXEC_POL>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N_c), RAJA::RangeSegment(0, N_r)),
      RAJA::make_tuple((int)0, (int)0, RAJA_Tile),

      [=] RAJA_DEVICE (int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {

        RAJA_Tile(ty, tx) = Aview(row, col);

      },

      [=] RAJA_DEVICE(int col, int row, int tx, int ty, TILE_MEM &RAJA_Tile) {

        Atview(col, row) = RAJA_Tile(ty, tx);

      });

  checkResult<int>(Atview, N_c, N_r);
  // printResult<int>(Atview, N_c, N_r);
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
      std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
                << std::endl;
    }
    std::cout << "" << std::endl;
  }
  std::cout << std::endl;
}
