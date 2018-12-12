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
 *  Matrix Transpose Example
 *
 *  In this example, an input matrix A of dimension N_r x N_c is
 *  reconfigured as a second matrix At with the rows of
 *  matrix A reorganized as the columns of At and the columns
 *  of matrix A as the rows of At.
 *
 *  This operation is carried out using a local memory tiling
 *  algorithm. The algorithm first loads matrix entries into a
 *  thread shared tile, a two-dimensional array, and then
 *  reads from the tile swapping the row and column indices for
 *  the output matrix.
 *
 *  The algorithm is expressed as a collection of ``outer``
 *  and ``inner`` for loops. Iterations of the inner loop will load/read
 *  data into the tile; while outer loops will iterate over the number
 *  of tiles needed to carry out the transposition. For simplicity we assume
 *  the tile size divides the number of rows and columns of the matrix.
 *
 *  RAJA variants of the example use RAJA local arrays as tiles.
 *  For CPU execution, RAJA local arrays are used to improve
 *  performance via cache blocking. For CUDA GPU execution,
 *  RAJA shared memory is mapped to CUDA shared memory which 
 *  enables threads in the same thread block to share data.
 *  
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *       - Multiple lambdas
 *       - TileTCount
 *       - ForICount
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
  // In the following implementations of matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N, N);

  //
  // Define TILE dimensions (TILE_DIM x TILE_DIM)
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

      int Tile[TILE_DIM][TILE_DIM];

      //
      // (1) Inner loops to load data into the tile
      //
      for (int ty = 0; ty < inner_Dim1; ++ty) {
        for (int tx = 0; tx < inner_Dim0; ++tx) {

          int col = bx * TILE_DIM + tx;  // Matrix column index
          int row = by * TILE_DIM + ty;  // Matrix row index
          Tile[ty][tx] = Aview(row, col);
        }
      }
      //
      // (2) Inner loops to read data from the tile
      //
      for (int ty = 0; ty < inner_Dim1; ++ty) {
        for (int tx = 0; tx < inner_Dim0; ++tx) {

          int col = by * TILE_DIM + tx;  // Transposed matrix column index
          int row = bx * TILE_DIM + ty;  // Transposed matrix row index
          Atview(row, col) = Tile[tx][ty];
        }
      }
    }
  }

  checkResult<int>(Atview, N);
  //printResult<int>(Atview, N);
  //----------------------------------------------------------------------------//

  //
  // The following RAJA variants use the RAJA::Kernel method to carryout the
  // transpose.
  //

  // Here we define a RAJA local array type. 
  // The array type is templated on
  // 1) Data type
  // 2) Index permutation
  // 3) Dimensions of the array
  //

  using TILE_MEM = RAJA::LocalArray<int, RAJA::PERM_IJ, RAJA::SizeList<TILE_DIM, TILE_DIM> >;

  // **NOTE** Although the LocalArray is contructed below
  // accessing its memory is only valid inside a RAJA kernel. 
  TILE_MEM RAJA_Tile;

  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - sequential matrix transpose example ...\n";

  std::memset(At, 0, N * N * sizeof(int));

  //
  // The following policy carries out the transpose
  // using a sequential execution policy.
  //
  using SEQ_EXEC_POL =
    RAJA::KernelPolicy<
      //
      // (0) Execution policies for outer loops
      //      These loops iterate over the number of
      //      tiles needed to carry out the transpose
      //
    RAJA::statement::TileTCount<1, RAJA::statement::Param<1>, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
      RAJA::statement::TileTCount<0, RAJA::statement::Param<0>, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

        // This statement will initalize local array memory inside a kernel.
        // The cpu_tile_mem policy specifies that memory should be allocated
        // on the stack. The entries in the RAJA::ParamList identify
        // RAJA local arrays to intialize. 
        RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<4>,

            //
            // (1) Execution policies for the first set of inner
            // loops. These loops copy data from the global matrices
            // to the local tile.
            //
           RAJA::statement::ForICount<1, RAJA::statement::Param<3>, RAJA::loop_exec,
             RAJA::statement::ForICount<0, RAJA::statement::Param<2>, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
           >,

           //
           // (2) Execution policies for the second set of inner
           // loops. These loops copy data from the local tile to 
           // the global matrix.
           //
           RAJA::statement::ForICount<1, RAJA::statement::Param<3>, RAJA::loop_exec,
             RAJA::statement::ForICount<0, RAJA::statement::Param<2>, RAJA::loop_exec,
                RAJA::statement::Lambda<1>
              >
            >
         > //close shared memory scope
       >//closes loop 2
     >//closes loop 3
    >; //closes policy list


  RAJA::kernel_param<SEQ_EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment(0, N), RAJA::RangeSegment(0,N)),
        RAJA::make_tuple((int) 0, (int) 0, (int) 0, (int) 0, RAJA_Tile),

     [=] (int gId0, int gId1, int tId0, int tId1, int id0, int id1,  TILE_MEM &RAJA_Tile) {
          
          //int gId0 = tId0 * TILE_DIM + id0;  // Matrix column index
          //int gId1 = tId1 * TILE_DIM + id1;  // Matrix row index
          
          RAJA_Tile(id1, id0)  = Aview(gId1, gId0);

     },

    [=] (int /*gId0*/, int /*gId1*/, int tId0, int tId1, int id0, int id1,  TILE_MEM &RAJA_Tile) {

       int col = tId1 * TILE_DIM + id0;  // Trasposed Matrix column index
       int row = tId0 * TILE_DIM + id1;  // Transposed Matrix row index
       
       Atview(row, col) = RAJA_Tile(id0,id1);

    });

  checkResult<int>(Atview, N);
  //printResult<int>(Atview, N);

#if defined(RAJA_ENABLE_OPENMP)
  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - OpenMP (parallel outer loop) matrix transpose example ...\n";

  std::memset(At, 0, N * N * sizeof(int));

  //
  // The following policy carries out the transpose
  // using parallel outer loops
  //
  using OPENMP_EXEC_1_POL =
    RAJA::KernelPolicy<
      //
      // (0) Execution policies for outer loops. 
      //      These loops iterate over the number of
      //      tiles needed to carry out the transpose
      //
    RAJA::statement::TileTCount<1, RAJA::statement::Param<1>, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::omp_parallel_for_exec,
      RAJA::statement::TileTCount<0, RAJA::statement::Param<0>, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

        // This statement will initalize local array memory inside a kernel.
        // The cpu_tile_mem policy specifies that memory should be allocated
        // on the stack. The entries in the RAJA::ParamList identify
        // RAJA local arrays to intialize. 
        RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<4>,

            //
            // (1) Execution policies for the first set of inner
            // loops. These loops copy data from the global matrices
            // to the local tile.
            //
           RAJA::statement::ForICount<1, RAJA::statement::Param<3>, RAJA::loop_exec,
             RAJA::statement::ForICount<0, RAJA::statement::Param<2>, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
            >,

            //
            // (2) Execution policies for the second set of inner
            // loops. These loops copy data from the local tile to 
            // the global matrix.
            //
           RAJA::statement::ForICount<1, RAJA::statement::Param<3>, RAJA::loop_exec,
             RAJA::statement::ForICount<0, RAJA::statement::Param<2>, RAJA::loop_exec,
                RAJA::statement::Lambda<1>
              >
            >
         > //close shared memory scope
       >//Tile 0
     >//Tile 1
    >; //closes policy list


  RAJA::kernel_param<OPENMP_EXEC_1_POL>(
    RAJA::make_tuple(RAJA::RangeSegment(0, N), RAJA::RangeSegment(0,N)),
    RAJA::make_tuple((int) 0, (int) 0, (int) 0, (int) 0, RAJA_Tile),

    [=] (int gId0, int gId1, int tId0, int tId1, int id0, int id1,  TILE_MEM &RAJA_Tile) {
          
      //int gId0 = tId0 * TILE_DIM + id0;  // Matrix column index
      //int gId1 = tId1 * TILE_DIM + id1;  // Matrix row index
      
      RAJA_Tile(id1, id0)  = Aview(gId1, gId0);
      
    },

    [=] (int /*gId0*/, int /*gId1*/, int tId0, int tId1, int id0, int id1,  TILE_MEM &RAJA_Tile) {

       int col = tId1 * TILE_DIM + id0;  // Trasposed Matrix column index
       int row = tId0 * TILE_DIM + id1;  // Transposed Matrix row index
      
      Atview(row, col) = RAJA_Tile(id0,id1);
      
   });

  checkResult<int>(Atview, N);
  //printResult<int>(Atview, N);

  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - OpenMP (parallel inner loops) matrix transpose example ...\n";

  std::memset(At, 0, N * N * sizeof(int));

  //
  // The following policy carries out the transpose
  // using parallel inner loops
  //
  using OPENMP_EXEC_2_POL =
    RAJA::KernelPolicy<
      //
      // (0) Execution policies for outer loops
      //      These loops iterate over the number of
      //      tiles needed to carry out the transpose
      //
    RAJA::statement::TileTCount<1, RAJA::statement::Param<1>, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
      RAJA::statement::TileTCount<0, RAJA::statement::Param<0>, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

        // This statement will initalize local array memory inside a kernel.
        // The cpu_tile_mem policy specifies that memory should be allocated
        // on the stack. The entries in the RAJA::ParamList identify
        // RAJA local arrays to intialize. 
        RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<4>,

            //
            // (1) Execution policies for the first set of inner
            // loops. These loops copy data from the global matrices
            // to the local tile.
            //
            RAJA::statement::ForICount<1, RAJA::statement::Param<3>, RAJA::omp_parallel_for_exec,
              RAJA::statement::ForICount<0, RAJA::statement::Param<2>, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
            >,

            //
            // (2) Execution policies for the second set of inner
            // loops. These loops copy data from the local tile to 
            // the global matrix.
            //
            RAJA::statement::ForICount<1, RAJA::statement::Param<3>, RAJA::loop_exec,
              RAJA::statement::ForICount<0, RAJA::statement::Param<2>, RAJA::omp_parallel_for_exec,
                RAJA::statement::Lambda<1>
             >
           >
         > //close shared memory scope
       >//closes loop 2
     >//closes loop 3
    >; //closes policy list


  RAJA::kernel_param<OPENMP_EXEC_2_POL>(
    RAJA::make_tuple(RAJA::RangeSegment(0, N), RAJA::RangeSegment(0,N)),
    RAJA::make_tuple((int) 0, (int) 0, (int) 0, (int) 0, RAJA_Tile),

    [=] (int gId0, int gId1, int tId0, int tId1, int id0, int id1,  TILE_MEM &RAJA_Tile) {
          
      //Calculated automatically
      //int gId0 = tId0 * TILE_DIM + id0; - Matrix column index
      //int gId1 = tId1 * TILE_DIM + id1; - Matrix column index
      
      RAJA_Tile(id1, id0)  = Aview(gId1, gId0);
      
    },

    [=] (int /*gId0*/, int /*gId1*/, int tId0, int tId1, int id0, int id1,  TILE_MEM &RAJA_Tile) {

       //Requires manual calculation
       int col = tId1 * TILE_DIM + id0;  // Trasposed Matrix column index
       int row = tId0 * TILE_DIM + id1;  // Transposed Matrix row index
      
      Atview(row, col) = RAJA_Tile(id0,id1);
      
   });

  checkResult<int>(Atview, N);
  //printResult<int>(Atview, N);  
#endif

  //--------------------------------------------------------------------------//
  std::cout << "\n Running RAJA - CUDA matrix transpose example ...\n";

  std::memset(At, 0, N * N * sizeof(int));

  //
  // The following policy carries out the transpose
  // using a CUDA execution policy
  //
#if defined(RAJA_ENABLE_CUDA)
  using CUDA_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      //
      // (0) Execution policies for outer loops
      //      These loops iterate over the number of
      //      tiles needed to carry out the transpose
      //
    RAJA::statement::TileTCount<1, RAJA::statement::Param<1>, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::cuda_block_y_loop,
      RAJA::statement::TileTCount<0, RAJA::statement::Param<0>, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::cuda_block_x_loop,

        // This statement will initalize local array memory inside a kernel.
        // The cpu_tile_mem policy specifies that memory should be allocated
        // on the stack. The entries in the RAJA::ParamList identify
        // RAJA local arrays to intialize. 
        RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<4>,

            //
            // (1) Execution policies for the first set of inner
            // loops. These loops copy data from the global matrices
            // to the local tile.
            //
           RAJA::statement::ForICount<1, RAJA::statement::Param<3>, RAJA::cuda_thread_y_direct,
             RAJA::statement::ForICount<0, RAJA::statement::Param<2>, RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<0>
              >
            >,
            //Synchronize threads to ensure all loads
            //to the local array are complete
            RAJA::statement::CudaSyncThreads,

           //
           // (2) Execution policies for the second set of inner
           // loops. These loops copy data from the local tile to 
           // the global matrix.
           //
           RAJA::statement::ForICount<1, RAJA::statement::Param<3>, RAJA::cuda_thread_y_direct,
             RAJA::statement::ForICount<0, RAJA::statement::Param<2>, RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<1>
              >
            >,
          //Synchronize threads to ensure all reads
          //from the local array are complete
          RAJA::statement::CudaSyncThreads,

         > //close shared memory scope
       >//closes loop 2
     >//closes loop 3
    >//close kernel policy
   >; //closes policy list



  RAJA::kernel_param<CUDA_EXEC_POL>(
    RAJA::make_tuple(RAJA::RangeSegment(0, N), RAJA::RangeSegment(0,N)),
    RAJA::make_tuple((int) 0, (int) 0, (int) 0, (int) 0, RAJA_Tile),

    [=] RAJA_DEVICE (int gId0, int gId1, int tId0, int tId1, int id0, int id1,  TILE_MEM &RAJA_Tile) {
          
      //int gId0 = tId0 * TILE_DIM + id0;  // Matrix column index
      //int gId1 = tId1 * TILE_DIM + id1;  // Matrix row index
      
      RAJA_Tile(id1, id0)  = Aview(gId1, gId0);
      
    },

    [=] RAJA_DEVICE (int /*gId0*/, int /*gId1*/, int tId0, int tId1, int id0, int id1,  TILE_MEM &RAJA_Tile) {

       int col = tId1 * TILE_DIM + id0;  // Trasposed Matrix column index
       int row = tId0 * TILE_DIM + id1;  // Transposed Matrix row index
      
      Atview(row, col) = RAJA_Tile(id0,id1);
      
   });
  
  checkResult<int>(Atview, N);
  //printResult<int>(Atview, N);
#endif

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
