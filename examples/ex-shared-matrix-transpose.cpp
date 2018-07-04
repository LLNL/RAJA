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
 *  Shared Memory Matrix Transpose Example
 *
 *  In this example, an input matrix A of dimension N x N is
 *  reconfigured as a second matrix At with the rows of 
 *  matrix A reorganized as the columns of At and the columns
 *  of matrix A becoming be the rows of matrix At. 
 *
 *  This operation is carried out using a shared memory tiling 
 *  algorithm. The algorithm first loads matrix entries into a 
 *  thread shared tile, a small two-dimensional array, and then 
 *  reads from the tile swapping the row and column indices for 
 *  the output matrix.
 *
 *  The algorithm is expressed as a collection of ``outer``
 *  and ``inner`` for loops. Iterations of the inner loop will load/read
 *  data into the tile; while outer loops will iterate over the number
 *  of tiles needed to carry out the transposition. For simplicity we assume
 *  the tile size divides the number of rows and columns of the matrix.
 *
 *  RAJA variants of the example construct a tile object using a RAJA shared memory
 *  window. For CPU execution, RAJA shared memory windows can be used to improve
 *  performance via cache blocking. For CUDA GPU execution, RAJA shared memory
 *  is mapped to CUDA shared memory.
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *       - Multiple lambdas
 *       - Shared memory tiling windows
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

//#define OMP_EX_1 //Breaks kernel

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

  std::cout << "\n\nRAJA shared matrix transpose example...\n";

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
  // In the following implementations of shared matrix transpose, we
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

  //
  // Define bounds for inner and outer loops
  //
  const int inner_Dim0 = TILE_DIM; 
  const int inner_Dim1 = TILE_DIM; 

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
  std::cout << "\n Running C-version of shared matrix transpose...\n";

  std::memset(At, 0, N * N * sizeof(int));

  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dim1; ++by) {
    for (int bx = 0; bx < outer_Dim0; ++bx) {

      int TILE[TILE_DIM][TILE_DIM];

      //
      // (1) Inner loops to load data into the tile
      //
      for (int ty = 0; ty < inner_Dim1; ++ty) {
        for (int tx = 0; tx < inner_Dim0; ++tx) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index
          TILE[ty][tx] = Aview(row, col);
        }
      }
      //
      // (2) Inner loops to read data from the tile
      //
      for (int ty = 0; ty < inner_Dim1; ++ty) {
        for (int tx = 0; tx < inner_Dim0; ++tx) {

          int col = by * TILE_DIM + tx;  // Transposed matrix column index
          int row = bx * TILE_DIM + ty;  // Transposed matrix row index
          Atview(row, col) = TILE[tx][ty];
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
  // Here, we define RAJA range segments to establish the iteration spaces for
  // the inner and outer loops.
  //
  RAJA::RangeSegment inner_Range0(0, inner_Dim0);
  RAJA::RangeSegment inner_Range1(0, inner_Dim1);
  RAJA::RangeSegment outer_Range0(0, outer_Dim0);
  RAJA::RangeSegment outer_Range1(0, outer_Dim1);

  //
  // Iteration spaces are stored within a RAJA tuple
  //
  auto segments =
      RAJA::make_tuple(inner_Range0, inner_Range1, outer_Range0, outer_Range1);

  //----------------------------------------------------------------------------//
  std::cout << "\n Running sequential shared matrix transpose ...\n";
  std::memset(At, 0, N * N * sizeof(int));

  //
  // Next, we construct a shared memory window object
  // to represent the tile load/read matrix entries.
  // The shared memory object constructor is templated on:
  // 1) RAJA shared memory type
  // 2) Data type
  // 3) List of lambda arguments which will be accessing the data
  // 4) Dimension of the objects
  // 5) The type of the tuple holding the iterations spaces
  //    (for simplicity decltype is used to infer the object type)
  //
  using seq_shmem_t = RAJA::ShmemTile<RAJA::seq_shmem,
                                      int,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::SizeList<TILE_DIM, TILE_DIM>,
                                      decltype(segments)>;
  seq_shmem_t RAJA_SEQ_TILE;

  //
  // The following policy carries out the transpose
  // using sequential loops.
  //

  using KERNEL_EXEC_POL = 
    RAJA::KernelPolicy<
      //
      // (0) Execution policies for outer loops
      //
      RAJA::statement::For<3, RAJA::seq_exec,
        RAJA::statement::For<2, RAJA::seq_exec,
          //
          // Creates a shared memory window for
          // usage within inner loops
          //
          RAJA::statement::SetShmemWindow<
            //
            // (1) Execution policies for the first set of inner
            // loops
            //
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
              >
            >,
            //
            // (2) Execution policies for second set of inner
            // loops
            //
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<1>
              >
            >
          > // closes shared memory window
        > // closes outer loop 2
       > // closes outer loop 3
    >; // closes policy list

  RAJA::kernel_param<KERNEL_EXEC_POL>(

      segments,

      RAJA::make_tuple(RAJA_SEQ_TILE),
      //
      // (1) Lambda for inner loops to load data into the tile
      //
      [=](int tx, int ty, int bx, int by, seq_shmem_t &RAJA_SEQ_TILE) {

        int col = bx * TILE_DIM + tx;  // Matrix column index
        int row = by * TILE_DIM + ty;  // Matrix row index
        RAJA_SEQ_TILE(ty, tx) = Aview(row, col);
      },

      //
      // (2) Lambda for inner loops to read data from the tile
      //
      [=](int tx, int ty, int bx, int by, seq_shmem_t &RAJA_SEQ_TILE) {

        int col = by * TILE_DIM + tx;  // Transposed matrix column index
        int row = bx * TILE_DIM + ty;  // Transposed matrix row index
        Atview(row, col) = RAJA_SEQ_TILE(tx, ty);
      }

      );

  checkResult<int>(Atview, N);
// printResult<int>(Atview, N);

//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_OPENMP)

  //
  // The following code creates an OpenMP shared memory window object to be used
  // by the following two examples.
  //
  using omp_shmem_t = RAJA::ShmemTile<RAJA::omp_shmem,
                                      int,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::SizeList<TILE_DIM, TILE_DIM>,
                                      decltype(segments)>;
  omp_shmem_t RAJA_OMP_TILE;

  //----------------------------------------------------------------------------//
#if defined(OMP_EX_1)
  std::cout << "\n Running openmp shared matrix transpose -  parallel top inner loop...\n";

  std::memset(At, 0, N * N * sizeof(int));

  //
  // This policy loops over tiles sequentially while exposing parallelism on
  // one of the inner loops.
  //
  using KERNEL_EXEC_POL_OMP = 
    RAJA::KernelPolicy<  
      //
      // (0) Execution policies for outer loops
      //
      RAJA::statement::For<3, RAJA::loop_exec,
        RAJA::statement::For<2, RAJA::loop_exec,
          //
          // Creates a shared memory window for
          // usage within inner loops
          //
          RAJA::statement::SetShmemWindow<
            //
            // (1) Lambda for inner loops to load data into the tile
            //
            RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::Lambda<0>
                >
              >,
              //
              // (2) Lambda for inner loops to read data from the tile
              //
            RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<1>
              >
             >
            > // closes shared memory window
          > // closes outer loop 2
         > // closes outer loop 3
    >; // closes policy list

  RAJA::kernel_param<KERNEL_EXEC_POL_OMP> (

        segments,

        RAJA::make_tuple(RAJA_OMP_TILE),
        //
        // (1) Lambda for inner loops to read data from the tile
        //
        [=] (int tx, int ty, int bx, int by, omp_shmem_t &RAJA_OMP_TILE) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index
          RAJA_OMP_TILE(ty, tx) = Aview(row, col);
        },

        //
        // (2) Lambda for inner loops to read data from the tile
        //
        [=] (int tx, int ty, int bx, int by, omp_shmem_t &RAJA_OMP_TILE) {

          int col = by * TILE_DIM + tx;  // Transposed matrix column index
          int row = bx * TILE_DIM + ty;  // Transposed matrix row index
          Atview(row, col) = RAJA_OMP_TILE(tx, ty);
        }

   );

  checkResult<int>(Atview, N);
  // printResult<int>(Atview, N);
#endif
  //----------------------------------------------------------------------------//

  std::cout << "\n Running openmp shared matrix transpose - collapsed inner loops...\n";
  std::memset(At, 0, N * N * sizeof(int));

  //
  // This policy loops over tiles sequentially while collapsing inner loops
  // into a single OpenMP parallel for loop enabling parallel loads/reads
  // to/from the tile.
  //
  using KERNEL_EXEC_POL_OMP2 = 
    RAJA::KernelPolicy<
      //
      // (0) Execution policies for outer loops
      //
      RAJA::statement::For<3, RAJA::loop_exec,
        RAJA::statement::For<2, RAJA::loop_exec,
          //
          // Creates a shared memory window for
          // usage within inner loops
          //
          RAJA::statement::SetShmemWindow<
            //
            // (1) Execution policies for the first set of inner
            //     loops. Loops are collapsed into a single
            //     parallel for loop.
            //
            RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::statement::Lambda<0>
            >,
            //
            // (2) Execution policies for second set of inner
            // loops. Loops are collapsed into a single parallel
            // for loops.
            //
            RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::statement::Lambda<1>
            >
          > // closes shared memory window
        > // closes outer loop 2
      > // closes outer loop 3
    >; // closes policy list

  RAJA::kernel_param<KERNEL_EXEC_POL_OMP2>(

      segments,

      RAJA::make_tuple(RAJA_OMP_TILE),
      //
      // (1) Lambda for inner loops to load data into the tile
      //
      [=](int tx, int ty, int bx, int by, omp_shmem_t &RAJA_OMP_TILE) {

        int col = bx * TILE_DIM + tx;  // Matrix column index
        int row = by * TILE_DIM + ty;  // Matrix row index
        RAJA_OMP_TILE(ty, tx) = Aview(row, col);
      },
      //
      // (2) Lambda for inner loops to read data from the tile
      //
      [=](int tx, int ty, int bx, int by, omp_shmem_t &RAJA_OMP_TILE) {

        int col = by * TILE_DIM + tx;  // Transposed matrix column index
        int row = bx * TILE_DIM + ty;  // Transposed matrix row index
        Atview(row, col) = RAJA_OMP_TILE(tx, ty);
      }

      );

  checkResult<int>(Atview, N);
// printResult<int>(Atview, N);
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running cuda shared matrix transpose ...\n";
  std::memset(At, 0, N * N * sizeof(int));

  //
  // Allocate CUDA shared memory
  //
  using cuda_shmem_t = RAJA::ShmemTile<RAJA::cuda_shmem,
                                       int,
                                       RAJA::ArgList<0, 1>,
                                       RAJA::SizeList<TILE_DIM, TILE_DIM>,
                                       decltype(segments)>;
  cuda_shmem_t RAJA_CUDA_TILE;

  using KERNEL_EXEC_POL_CUDA = 
    RAJA::KernelPolicy<   
      //
      // Collapses policy list into a single CUDA kernel
      //
      RAJA::statement::CudaKernel<
        //
        // (0) Execution policies for outer loops.
        //     Maps iterations from the outer loop
        //     to CUDA thread blocks.
        //
        RAJA::statement::For<3, RAJA::cuda_block_exec,
          RAJA::statement::For<2, RAJA::cuda_block_exec,
            //
            // Creates a shared memory window for
            // usage within threads in a thread block.
            //
            RAJA::statement::SetShmemWindow<
              //
              // (1) Execution policies for the first set of
              // inner loops. Each iteration is assigned to
              // a thread in a CUDA thread block.
              //
              RAJA::statement::For<1, RAJA::cuda_thread_exec,
                RAJA::statement::For<0, RAJA::cuda_thread_exec,
                  RAJA::statement::Lambda<0>
                >
              >,
               //
               // Places a barrier to synchronize CUDA threads
               // within a block. Necessary to ensure data has
               // been loaded into the tile before reading from
               // the tile.
               //
               RAJA::statement::CudaSyncThreads,
               //
               // (2) Execution policies for second set of inner
               //     loops. Each iteration is assigned to a
               //     thread in a CUDA thread block.
               //
               RAJA::statement::For<1, RAJA::cuda_thread_exec,
                 RAJA::statement::For<0, RAJA::cuda_thread_exec,
                   RAJA::statement::Lambda<1>
                 >
               >
            >// closes shared memory window
          > // closes outer loop 2
        > // closes outer loop 3
      > // closes CUDA Kernel
    >; // closes policy list

  RAJA::kernel_param<KERNEL_EXEC_POL_CUDA>(

      segments,

      RAJA::make_tuple(RAJA_CUDA_TILE),
      //
      // (1) Lambda for inner loops to load data into the tile
      //
      [=] RAJA_DEVICE(int tx, int ty, int bx, int by, cuda_shmem_t &RAJA_CUDA_TILE) {

        int col = bx * TILE_DIM + tx;
        int row = by * TILE_DIM + ty;

        RAJA_CUDA_TILE(ty, tx) = Aview(row, col);
      },
      //
      // (2) Lambda for inner loops to read data from the tile
      //
      [=] RAJA_DEVICE(int tx, int ty, int bx, int by, cuda_shmem_t &RAJA_CUDA_TILE) {

        int col = by * TILE_DIM + tx;
        int row = bx * TILE_DIM + ty;

        Atview(row, col) = RAJA_CUDA_TILE(tx, ty);

      }

      );

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
