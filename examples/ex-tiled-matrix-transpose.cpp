//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Tiled Matrix Transpose Example
 *
 *  In this example, an input matrix A of dimension N x N is
 *  reconfigured as a second matrix At with the rows of 
 *  matrix A reorganized as the columns of At and the columns
 *  of matrix A becoming be the rows of matrix At. 
 *
 *  This operation is carried out using a tiling algorithm.
 *  The algorithm iterates over tiles of the matrix and 
 *  performs a transpose copy of a single block. 
 *
 *  The algorithm is expressed as a collection of ``outer``
 *  and ``inner`` for loops. Iterations of the inner loop will 
 *  tranpose tile entries; while outer loops will iterate over
 *  the number of tiles needed to carryout the transposition. 
 *  For simplicity we assume the tile size divides the number 
 *  of rows and columns of the matrix.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *       - Multiple lambdas
 *       - Tiling
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
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N);

//
// Function for printing results
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA tiled matrix transpose example...\n";

  //
  // Define num rows/cols in matrix
  //
  const int N = 256;

  //
  // Allocate matrix data
  //
  int *A  = memoryManager::allocate<int>(N * N);
  int *At = memoryManager::allocate<int>(N * N);

  //
  // In the following implementations of tiled matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N, N);

  //
  // Define TILE dimensions
  //
  const int TILE_DIM = 16;

  const int outer_Dim0 = N/TILE_DIM;
  const int outer_Dim1 = N/TILE_DIM;

  //
  // Initialize matrix data
  //
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      Aview(row, col) = col;
    }
  }


  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of tiled matrix transpose...\n";

  std::memset(At, 0, N * N * sizeof(int));

  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dim1; ++by) {
    for (int bx = 0; bx < outer_Dim0; ++bx) {
      //
      // (1) Loops to iterate over tile entries
      //
      for (int ty = 0; ty < TILE_DIM; ++ty) {
        for (int tx = 0; tx < TILE_DIM; ++tx) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index
          Atview(col, row) = Aview(row, col);
        }
      }
      
    }
  }

  checkResult<int>(Atview, N);
  // printResult<int>(At, N);
  //----------------------------------------------------------------------------//

  //
  // The following RAJA variants use the RAJA::kernel method to carryout the
  // transpose. 
  //
  // Here, we define RAJA range segments to establish the iteration spaces.
  // Further partioning of the iteration space is carried out in the 
  // tile_fixed statements. Iterations inside a RAJA loop is given by their
  // global iteration number. 
  //
  RAJA::RangeSegment row_Range(0, N);
  RAJA::RangeSegment col_Range(0, N);

  //----------------------------------------------------------------------------//
  std::cout << "\n Running sequential tiled matrix transpose ...\n";
  std::memset(At, 0, N * N * sizeof(int));

  //
  // The following policy carries out the transpose
  // using sequential loops. The template parameter inside 
  // tile_fixed corresponds to the dimension size of the tile.
  //
  using KERNEL_EXEC_POL = 
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::seq_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::seq_exec,
          RAJA::statement::For<1, RAJA::seq_exec, 
            RAJA::statement::For<0, RAJA::seq_exec,
              RAJA::statement::Lambda<0>
            > //closes For 0
          > //closes For 1
        > // closes Tile 0
      > // closes Tile 1
    >; // closes policy list

  RAJA::kernel<KERNEL_EXEC_POL>(
                         RAJA::make_tuple(col_Range, row_Range),
                         [=](int col, int row) {

    Atview(col, row) = Aview(row, col);

  });

  checkResult<int>(Atview, N);
  //printResult<int>(Atview, N);

//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running openmp tiled matrix transpose -  parallel top inner loop...\n";

  std::memset(At, 0, N * N * sizeof(int));

  //
  // This policy loops over tiles sequentially while exposing parallelism on
  // one of the inner loops.
  //
  using KERNEL_EXEC_POL_OMP = 
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::seq_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::seq_exec,
          RAJA::statement::For<1, RAJA::omp_parallel_for_exec, 
            RAJA::statement::For<0, RAJA::loop_exec,
              RAJA::statement::Lambda<0>
            > //close For 0
          > //closes For 1
        > // closes Tile 0
      > // closes Tile 1
    >; // closes policy list

  RAJA::kernel<KERNEL_EXEC_POL_OMP>(
                          RAJA::make_tuple(col_Range, row_Range), 
                          [=](int col, int row) {

    Atview(col, row) = Aview(row, col);

  });

  checkResult<int>(Atview, N);
  // printResult<int>(Atview, N);
  //----------------------------------------------------------------------------//

  std::cout << "\n Running openmp tiled matrix transpose - collapsed inner loops...\n";
  std::memset(At, 0, N * N * sizeof(int));

  //
  // This policy loops over tiles sequentially while collapsing inner loops
  // into a single OpenMP parallel for loop enabling parallel loads/reads
  // to/from the tile.
  //
  using KERNEL_EXEC_POL_OMP2 = 
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::seq_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::seq_exec,
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
                                    RAJA::statement::Lambda<0>
          > //closes collapse
        > // closes Tile 0
      > // closes Tile 1
    >; // closes policy list
      
  RAJA::kernel<KERNEL_EXEC_POL_OMP2>(
                        RAJA::make_tuple(col_Range, row_Range), 
                        [=](int col, int row) {

    Atview(col, row) = Aview(row, col);

  });

  checkResult<int>(Atview, N);
// printResult<int>(Atview, N);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running cuda tiled matrix transpose ...\n";
  std::memset(At, 0, N * N * sizeof(int));
  
  using KERNEL_EXEC_POL_CUDA = 
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::seq_exec,
          RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::seq_exec,
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::For<0, RAJA::cuda_thread_exec, 
                                      RAJA::statement::Lambda<0> 
              >
            >
          >
        >
      >
    >;

  RAJA::kernel<KERNEL_EXEC_POL_CUDA>(
                           RAJA::make_tuple(col_Range, col_Range), 
                           [=] RAJA_DEVICE (int col, int row) {

    Atview(col, row) = Aview(row, col);
      
  });
  checkResult<int>(Atview, N);
// printResult<int>(Atview, N);
#endif
  //----------------------------------------------------------------------------//


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
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if (Atview(col, row) != col) {
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
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
                << std::endl;
    }
  }
  std::cout << std::endl;
}
