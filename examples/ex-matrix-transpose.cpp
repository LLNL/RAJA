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
 *  Example takes an input matrix A and produces a second matrix 
 *  AT with the row and column indicies swaped. This example
 *  caries out the tranpose by using a 'tiled' approach. 
 *
 *  RAJA features shown:
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *    - Basic usage of  RAJA shared memory
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
#endif


//
// Define dimensionality of matrices.
//
const int DIM = 2;

//
// Define macros to simplify row-col indexing (non-RAJA implementations only)
//
#define A(r, c) A[c + N * r]
#define At(r, c) At[c + N * r]


//
// Functions for checking results
//
template <typename T>
void checkResult(T *At, int N);

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N);

//
// Functions for printing results
//
template <typename T>
void printResult(T *At, int N);

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
// Allocate and initialize matrix data.
//
  int *A = memoryManager::allocate<int>(N * N);
  int *At = memoryManager::allocate<int>(N * N);

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A(row, col) = col;
    }
  }

//----------------------------------------------------------------------------//

//Define TILE dimensions
  const int TILE_DIM = 16;

//
//Define iteration spaces for outer and inner loop
//
  const int innerDIM_1 = 8;
  const int innerDIM_0 = 16;
  const int outerDIM_0 = (N-1)/TILE_DIM + 1;
  const int outerDIM_1 = (N-1)/TILE_DIM + 1;


//----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of tiled matrix transpose...\n";
  
  std::memset(At, 0, N*N * sizeof(int)); 

  //
  //Outer loop: loops over tiles - kernel 
  //
  for (int by = 0; by < outerDIM_1; ++by) {
    for (int bx = 0; bx < outerDIM_0; ++bx) {
      
      
      int TILE[TILE_DIM][TILE_DIM];
      
      //
      //Inner loop: load data into tile
      //
      for ( int ty = 0; ty < innerDIM_1; ++ty) {
        for ( int tx = 0; tx < innerDIM_0; ++tx) {
          
          int x = bx * TILE_DIM + tx;
          int y = by * TILE_DIM + ty;
          
          for ( int j = 0; j < TILE_DIM; j += innerDIM_1) {
            TILE[ty + j][tx] = A((y+j), x);
          }
          
        }
      }

      //
      //Inner loop: read data from tile
      //
      for ( int ty = 0; ty < innerDIM_1; ++ty) {
        for ( int tx = 0; tx < innerDIM_0; ++tx) {
          
          int x = by * TILE_DIM + tx;
          int y = bx * TILE_DIM + ty;
          
          for(int j = 0; j < TILE_DIM; j += innerDIM_1) {
            At((y+j), x) = TILE[tx][ty+j];
          }

        }
      }


    }
  }

  checkResult<int>(At, N);
//printResult<int>(At, N);  


//----------------------------------------------------------------------------//

//
// In the following RAJA implementations of tiled matrix transpose, we 
// use RAJA 'View' objects to access the matrix data. A RAJA view
// holds a pointer to a data array and enables multi-dimensional indexing 
// into that data, similar to the macros we defined above.
//
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N, N);

//----------------------------------------------------------------------------//

//
// Here, we define RAJA range segments to define the ranges of
// row and column indices
//
  RAJA::RangeSegment inner_Range0(0, innerDIM_0);
  RAJA::RangeSegment inner_Range1(0, innerDIM_1);
  RAJA::RangeSegment outer_Range0(0, outerDIM_0);
  RAJA::RangeSegment outer_Range1(0, outerDIM_1);

//----------------------------------------------------------------------------//

  //
  //Create a tuple iteration spaces
  //
  auto segments = RAJA::make_tuple(outer_Range1, outer_Range0, inner_Range1, inner_Range0);

  //
  //Create shared memory object 
  //
  using shmem_t = RAJA::ShmemTile< RAJA::seq_shmem,double, RAJA::ArgList<2,3>,
                                   RAJA::SizeList<TILE_DIM,TILE_DIM>,decltype(segments)>;
  shmem_t shared_data;


//----------------------------------------------------------------------------//  
  std::cout << "\n Running sequential tiled matrix transpose ...\n";
  std::memset(At, 0, N*N * sizeof(int));

  using NESTED_EXEC_POL = 
    RAJA::KernelPolicy<
     RAJA::statement::For<0, RAJA::loop_exec, //outer loop by
       RAJA::statement::For<1, RAJA::loop_exec, //outer loop bx
          RAJA::statement::Lambda<0>,
          RAJA::statement::SetShmemWindow<
              RAJA::statement::For<2, RAJA::loop_exec, //inner loop
                RAJA::statement::For<3, RAJA::loop_exec, //inner loop
                  RAJA::statement::Lambda<1>
                                     > //closes inner loop ty
                                   > //closes inner loop tx
              ,
            RAJA::statement::For<2, RAJA::loop_exec, //inner loop
              RAJA::statement::For<3, RAJA::loop_exec, //inner loop
               RAJA::statement::Lambda<2> //loop body 2
                                     > //for 2
                                   > //for 3 
              > //closes shared memory window
                            > //closes outer loop by
                          > //closes outer loop bx
    >; //policy



  RAJA::kernel_param<NESTED_EXEC_POL> (
                                     
        segments, RAJA::make_tuple(shared_data),
        
        [=] (int by, int bx, int ty, int tx, shmem_t &shared_data) {

        }, 

        [=] (int by, int bx, int ty, int tx, shmem_t &shared_data) {
          
          int x = bx * TILE_DIM + tx;
          int y = by * TILE_DIM + ty;
          
          for ( int j = 0; j < TILE_DIM; j += innerDIM_1) { 
            shared_data((ty+j),tx) = A[(y+j)*N + x];
          }

        }, 

        [=] (int by, int bx, int ty, int tx, shmem_t &shared_data) {

          int x = by * TILE_DIM + tx;
          int y = bx * TILE_DIM + ty;

          for(int j=0; j<TILE_DIM; j+= innerDIM_1){
            At[(y+j)*N + x] = shared_data(tx,(ty+j));
          }
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
                                     
        segments, RAJA::make_tuple(shared_data),
        
        [=] (int by, int bx, int ty, int tx, shmem_t &shared_data) {

        }, 

        [=] (int by, int bx, int ty, int tx, shmem_t &shared_data) {
          
          int x = bx * TILE_DIM + tx;
          int y = by * TILE_DIM + ty;
          
          for ( int j = 0; j < TILE_DIM; j += innerDIM_1) { 
            shared_data((ty+j),tx) = A[(y+j)*N + x];
          }

        }, 

        [=] (int by, int bx, int ty, int tx, shmem_t &shared_data) {

          int x = by * TILE_DIM + tx;
          int y = bx * TILE_DIM + ty;

          for(int j=0; j<TILE_DIM; j+= innerDIM_1){
            At[(y+j)*N + x] = shared_data(tx,(ty+j));
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
                                     
        segments, RAJA::make_tuple(shared_data),
        
        [=] (int by, int bx, int ty, int tx, shmem_t &shared_data) {

        }, 

        [=] (int by, int bx, int ty, int tx, shmem_t &shared_data) {
          
          int x = bx * TILE_DIM + tx;
          int y = by * TILE_DIM + ty;
          
          for ( int j = 0; j < TILE_DIM; j += innerDIM_1) { 
            shared_data((ty+j),tx) = A[(y+j)*N + x];
          }

        }, 

        [=] (int by, int bx, int ty, int tx, shmem_t &shared_data) {

          int x = by * TILE_DIM + tx;
          int y = bx * TILE_DIM + ty;

          for(int j=0; j<TILE_DIM; j+= innerDIM_1){
            At[(y+j)*N + x] = shared_data(tx,(ty+j));
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
          
          for ( int j = 0; j < TILE_DIM; j += innerDIM_1) { 
            cuda_shmem((ty+j),tx) = A[(y+j)*N + x];
          }

        }, 

        [=] RAJA_DEVICE (int by, int bx, int ty, int tx, cuda_shmem_t &cuda_shmem) {

          int x = by * TILE_DIM + tx;
          int y = bx * TILE_DIM + ty;

          for(int j=0; j<TILE_DIM; j+= innerDIM_1){
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
