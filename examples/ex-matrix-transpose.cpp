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

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Tiled Matrix Tranpose Example
 *
 *  Example takes an input matrix A of dimension N x N and
 *  produces a second matrix AT with the rows of matrix A as 
 *  the columns and vice-versa. 
 *
 *  This example caries out the tranpose by using a tiling algorithm.
 *  The algorithm tranposes the matrix by first loading a block
 *  of the matrix into a tile. The algorithm then reads
 *  from the local tile swapping the row and column indicies 
 *  to write into an ouput matrix.
 *
 *  The algorithm may be viewed as a collection of ``inner``
 *  and ``outer`` loops. Iterations of the inner loop will load/read 
 *  data into the tile, while outer loops will iterate over the number
 *  of tiles needed to carry out the transpose. For simplicity we assume
 *  that the tile size divide the number of rows and columns of the matrix.
 * 
 *  For CPU execution, RAJA shared memory windows can be used to improve 
 *  performance via cache blocking. For CUDA GPU execution, CUDA shared 
 *  memory can be used to improve performance. 
 *  
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *    - Basic usage of  RAJA shared memory
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
  int *A = memoryManager::allocate<int>(N * N);
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
//Initialize matrix data
//
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      Aview(row, col) = col;
    }
  }
  //----------------------------------------------------------------------------//
  
//
// Define TILE dimensions
//
  const int TILE_DIM = 16;

//
//Define iteration spaces for inner and outer loops
//
  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = RAJA_DIVIDE_CEILING_INT(N,TILE_DIM);
  const int outer_Dim1 = RAJA_DIVIDE_CEILING_INT(N,TILE_DIM);


//----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of tiled matrix transpose...\n";
  
  std::memset(At, 0, N*N * sizeof(int)); 

  //
  // (0) Outer loop: loops over tiles
  //
  for (int by = 0; by < outer_Dim1; ++by) {
    for (int bx = 0; bx < outer_Dim0; ++bx) {
      
      
      int TILE[TILE_DIM][TILE_DIM];
      
      //
      // (1) Inner loops - load data into tile
      //
      for ( int ty = 0; ty < inner_Dim1; ++ty) {
        for ( int tx = 0; tx < inner_Dim0; ++tx) {
                    
          int col = bx * TILE_DIM + tx; //Matrix column index
          int row = by * TILE_DIM + ty; //Matrix row index
          
          TILE[ty][tx] = Aview(row, col);
        }
      }

      //
      // (2) Inner loops read data from tile
      //
      for ( int ty = 0; ty < inner_Dim1; ++ty) {
        for ( int tx = 0; tx < inner_Dim0; ++tx) {
          
          int col = by * TILE_DIM + tx; //Transposed matrix column index
          int row = bx * TILE_DIM + ty; //Transposed matrix row index
          
          Atview(row, col) = TILE[tx][ty];
        }
      }


    }
  }

  checkResult<int>(Atview, N);
  //printResult<int>(At, N);  
//----------------------------------------------------------------------------//

//
// The following RAJA variants will use the RAJA::kernel method to carryout the
// transpose. The tile array within iterations of the inner loop is represented
// using a RAJA shared memory object. 
//

//
// Here, we define RAJA range segments to establish the iteration spaces for the 
// inner and outer loops.
//
  RAJA::RangeSegment inner_Range0(0, inner_Dim0);
  RAJA::RangeSegment inner_Range1(0, inner_Dim1);
  RAJA::RangeSegment outer_Range0(0, outer_Dim0);
  RAJA::RangeSegment outer_Range1(0, outer_Dim1);

//
// Iteration spaces are stored within a RAJA tuple
//
  auto segments = RAJA::make_tuple(outer_Range1, outer_Range0, inner_Range1, inner_Range0);

//----------------------------------------------------------------------------//  
  std::cout << "\n Running sequential tiled matrix transpose ...\n";
  std::memset(At, 0, N*N * sizeof(int));

  using KERNEL_EXEC_POL = 
    RAJA::KernelPolicy<
     //              
     // (0) Execution policies for outer loops -
     //     Loops over tiles
     //
     RAJA::statement::For<0, RAJA::loop_exec,
       RAJA::statement::For<1, RAJA::loop_exec,
        RAJA::statement::Lambda<0>,
          //
          // Creates a shared memory object for
          // usage within inner loops
          //
          RAJA::statement::SetShmemWindow< 
            //              
            // (1) Execution policies for first set of inner loops -
            //     Loads data into tile
            //
            RAJA::statement::For<2, RAJA::loop_exec, 
              RAJA::statement::For<3, RAJA::loop_exec, 
               RAJA::statement::Lambda<1>
                                   >
                                 >, 
            //              
            // (2) Execution policies for second set of inner loops -
            //     Loads data into tile
            //
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::For<3, RAJA::loop_exec,
               RAJA::statement::Lambda<2>
                                     > 
                                   >
               > //close shared memory
              > //closes outer loop 1
             > //closes outer loop 2
    >; //close policy list

  //
  //Create a shared memory object type. The object is templated on: 
  // 1. shared memory type
  // 2. data type
  // 3. list of lambda arguments which will be accessing the data
  // 4. dimension of the objects
  // 5. the type of the tuple holding the iterations spaces 
  //    (for simplicity decltype is used to infer the type)
  //
  using cpu_shmem_t = RAJA::ShmemTile< RAJA::seq_shmem,double, RAJA::ArgList<2,3>,
                                       RAJA::SizeList<TILE_DIM,TILE_DIM>,decltype(segments)>;
  cpu_shmem_t RAJA_TILE;

  RAJA::kernel_param<KERNEL_EXEC_POL> (
                                     
        segments, RAJA::make_tuple(RAJA_TILE),        
        //
        // (0) Lambda for the outer loops
        //
        [=] (int by, int bx, int ty, int tx, cpu_shmem_t &RAJA_TILE) {

        }, 

        //
        // (1) Lambda for the first set of inner loops which loads data
        //     into the tile from the input matrix
        //
        [=] (int by, int bx, int ty, int tx, cpu_shmem_t &RAJA_TILE) {          
          int x = bx * TILE_DIM + tx;
          int y = by * TILE_DIM + ty;          
          RAJA_TILE(ty,tx) = Aview(y,x);
        }, 

        //
        // (2) Lambda for the second set of inner loops which loads data
        // from the tile to the output matrix
        //
        [=] (int by, int bx, int ty, int tx, cpu_shmem_t &RAJA_TILE) {
          int x = by * TILE_DIM + tx;
          int y = bx * TILE_DIM + ty;
          Atview(y,x) = RAJA_TILE(tx,ty);
        }

  );

  checkResult<int>(Atview, N);
//printResult<int>(Atview, N);

//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running sequential tiled mat-trans openmp ver 1...\n";
  std::memset(At, 0, N*N * sizeof(int));

  using NESTED_EXEC_POL_OMP = 
    RAJA::KernelPolicy<
     RAJA::statement::For<0, RAJA::loop_exec, //outer loop by
       RAJA::statement::For<1, RAJA::loop_exec, //outer loop bx
          RAJA::statement::Lambda<0>,
          RAJA::statement::SetShmemWindow<
            RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                      RAJA::ArgList<2,3>,
                                      RAJA::statement::Lambda<1>
                                      > //closes collapse statement
            ,
            RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                      RAJA::ArgList<2,3>,
                                      RAJA::statement::Lambda<2>
                                   > //closes collapse statement
              > //closes shared memory window
                            > //closes outer loop by
                          > //closes outer loop bx
    >; //policy



  RAJA::kernel_param<NESTED_EXEC_POL_OMP> (
                                     
        segments, RAJA::make_tuple(RAJA_TILE),
        
        [=] (int by, int bx, int ty, int tx, cpu_shmem_t &RAJA_TILE) {

        }, 

        [=] (int by, int bx, int ty, int tx, cpu_shmem_t &RAJA_TILE) {
          
          int x = bx * TILE_DIM + tx;
          int y = by * TILE_DIM + ty;
          
          for ( int j = 0; j < TILE_DIM; j += inner_Dim1) { 
            RAJA_TILE((ty+j),tx) = A[(y+j)*N + x];
          }

        }, 

        [=] (int by, int bx, int ty, int tx, cpu_shmem_t &RAJA_TILE) {

          int x = by * TILE_DIM + tx;
          int y = bx * TILE_DIM + ty;

          for(int j=0; j<TILE_DIM; j+= inner_Dim1){
            At[(y+j)*N + x] = RAJA_TILE(tx,(ty+j));
          }
        }

  );

  checkResult<int>(Atview, N);
//printResult<int>(Atview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential tiled mat-trans openmp ver 2...\n";  
  std::memset(At, 0, N*N * sizeof(int));


  using NESTED_EXEC_POL_OMP2 = 
    RAJA::KernelPolicy<      
     RAJA::statement::For<0, RAJA::loop_exec, //outer loop by
       RAJA::statement::For<1, RAJA::loop_exec, //outer loop bx
          RAJA::statement::Lambda<0>,
            RAJA::statement::SetShmemWindow<
              RAJA::statement::For<2, RAJA::omp_parallel_for_exec, //inner loop
                RAJA::statement::For<3, RAJA::loop_exec, //inner loop
                  RAJA::statement::Lambda<1>
                                     > //closes inner loop ty
                                   > //closes inner loop tx
              ,
            RAJA::statement::For<2, RAJA::omp_parallel_for_exec, //inner loop
              RAJA::statement::For<3, RAJA::loop_exec, //inner loop
               RAJA::statement::Lambda<2> //loop body 2
                                     > //for 2
                                   > //for 3 
              > //closes shared memory window
                            > //closes outer loop by
                          > //closes outer loop bx
    >; //policy


  RAJA::kernel_param<NESTED_EXEC_POL_OMP2> (
                                     
        segments, RAJA::make_tuple(RAJA_TILE),
        
        [=] (int by, int bx, int ty, int tx, cpu_shmem_t &RAJA_TILE) {

        }, 

        [=] (int by, int bx, int ty, int tx, cpu_shmem_t &RAJA_TILE) {
          
          int x = bx * TILE_DIM + tx;
          int y = by * TILE_DIM + ty;
          
          for ( int j = 0; j < TILE_DIM; j += inner_Dim1) { 
            RAJA_TILE((ty+j),tx) = A[(y+j)*N + x];
          }

        }, 

        [=] (int by, int bx, int ty, int tx, cpu_shmem_t &RAJA_TILE) {

          int x = by * TILE_DIM + tx;
          int y = bx * TILE_DIM + ty;

          for(int j=0; j<TILE_DIM; j+= inner_Dim1){
            At[(y+j)*N + x] = RAJA_TILE(tx,(ty+j));
          }
        }

  );

  checkResult<int>(Atview, N);
//printResult<int>(Atview, N);
#endif

//----------------------------------------------------------------------------//
#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running cuda tiled matrix transpose ...\n";
  std::memset(At, 0, N*N * sizeof(int));

  using NESTED_EXEC_POL_CUDA = 
    RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
     RAJA::statement::For<0, RAJA::cuda_block_exec, //outer loop by
       RAJA::statement::For<1, RAJA::cuda_block_exec, //outer loop bx
          RAJA::statement::Lambda<0>,
          RAJA::statement::SetShmemWindow<
              RAJA::statement::For<2, RAJA::cuda_thread_exec, //inner loop
                RAJA::statement::For<3, RAJA::cuda_thread_exec, //inner loop
                  RAJA::statement::Lambda<1>
                                     > //closes inner loop ty
                                   > //closes inner loop tx
            ,RAJA::statement::CudaSyncThreads, //synchronize threads
            RAJA::statement::For<2, RAJA::cuda_thread_exec, //inner loop
              RAJA::statement::For<3, RAJA::cuda_thread_exec, //inner loop
               RAJA::statement::Lambda<2> //loop body 2
                                     > //for 2
                                   > //for 3 
              > //closes shared memory window
                            > //closes outer loop by
                          > //closes outer loop bx
      >//close CUDA Kernel
    >; //policy

  
  //
  //Allocate CUDA shared memory
  //
  using cuda_shmem_t = RAJA::ShmemTile<RAJA::cuda_shmem, double,RAJA::ArgList<2,3>,RAJA::SizeList<TILE_DIM,TILE_DIM>, decltype(segments)>;
  cuda_shmem_t cuda_shmem; 
  

  RAJA::kernel_param<NESTED_EXEC_POL_CUDA> (
                                     
        segments, RAJA::make_tuple(cuda_shmem),
        
        [=] RAJA_DEVICE (int by, int bx, int ty, int tx, cuda_shmem_t &cuda_shmem) {

        }, 

        [=] RAJA_DEVICE (int by, int bx, int ty, int tx, cuda_shmem_t &cuda_shmem) {
     
          int x = bx * TILE_DIM + tx;
          int y = by * TILE_DIM + ty;
          
          for ( int j = 0; j < TILE_DIM; j += inner_Dim1) { 
            cuda_shmem((ty+j),tx) = A[(y+j)*N + x];
          }

        }, 

        [=] RAJA_DEVICE (int by, int bx, int ty, int tx, cuda_shmem_t &cuda_shmem) {

          int x = by * TILE_DIM + tx;
          int y = bx * TILE_DIM + ty;

          for(int j=0; j<TILE_DIM; j+= inner_Dim1){
            At[(y+j)*N + x] = cuda_shmem(tx,(ty+j));
          }
        }

  );

  checkResult<int>(Atview, N);
//printResult<int>(Atview, N);
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
// Functions to check result and report P/F.
//
template <typename T>
void checkResult(T* At, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( At(col, row) != col ) { 
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
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( Atview(col, row) != col ) { 
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
