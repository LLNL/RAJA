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
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

//
// Define dimensionality of matrices
//
const int DIM = 2;

//
// Define num rows/cols in matrix
//
const int N = 4;



// Define TILE dimensions
//
const int TILE_DIM = 2;

//
// Define bounds for inner and outer loops
//
const int inner_Dim0 = TILE_DIM; 
const int inner_Dim1 = TILE_DIM; 

const int outer_Dim0 = N/TILE_DIM;
const int outer_Dim1 = N/TILE_DIM;


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

  std::cout << "\n\nRAJA vector addition example...\n";

  //
  // Allocate matrix data
  //
  int *A  = memoryManager::allocate<int>(N * N);
  int *At = memoryManager::allocate<int>(N * N);

  int *B  = memoryManager::allocate<int>(N * N);
  int *Bt = memoryManager::allocate<int>(N * N);

  //
  // In the following implementations of shared matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N, N);

  RAJA::View<int, RAJA::Layout<DIM>> Bview(B, N, N);
  RAJA::View<int, RAJA::Layout<DIM>> Btview(Bt, N, N);


  //
  // Initialize matrix data
  //
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = col;
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
          //std::cout<<Aview(row,col)<<std::endl;
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
  //printResult<int>(Atview, N);
  //----------------------------------------------------------------------------//


  //----------------------------------------------------------------------------//
  //RAJA Shared memory version... 
  printf("Run RAJA shared memory version \n");
  
  std::memset(At, 0, N * N * sizeof(int));

  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                     RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1));


  using SharedTile = RAJA::SharedMem<int, TILE_DIM, TILE_DIM>;

  //Create shared memory object
  using mySharedMemory = RAJA::SharedMemWrapper<SharedTile>;
  mySharedMemory myTile;
  mySharedMemory myTile2;


  using KERNEL_EXEC_POL = 
    RAJA::KernelPolicy<
      RAJA::statement::For<3, RAJA::loop_exec,
        RAJA::statement::For<2, RAJA::loop_exec,
                                                          
                             RAJA::statement::CreateShmem,

         RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                   RAJA::ArgList<0, 1>,
                                   RAJA::statement::Lambda<0>
                                   >,

         RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                   RAJA::ArgList<0, 1>,
                                   RAJA::statement::Lambda<1>
                                   >
        >//for 2
       >//for 3
      >; //close policy list

  RAJA::kernel_param<KERNEL_EXEC_POL>(iSpace, 
                                      RAJA::make_tuple(myTile, myTile2),

      [=] (int tx, int ty, int bx, int by, mySharedMemory &myTile,  mySharedMemory &myTile2) {
         
           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
           (*myTile.SharedMem)(ty,tx)  = Aview(row, col);
           (*myTile2.SharedMem)(ty,tx) = Bview(row, col);
        },

      //read from shared mem
       [=] (int tx, int ty, int bx, int by, mySharedMemory &myTile, mySharedMemory &myTile2) {
           
           int col = by * TILE_DIM + tx;  // Transposed matrix column index
           int row = bx * TILE_DIM + ty;  // Transposed matrix row index
           Atview(row, col) = (*myTile.SharedMem)(tx,ty);
           Btview(row, col) = (*myTile2.SharedMem)(tx,ty);
        });                                         




    checkResult<int>(Atview, N);
    checkResult<int>(Btview, N);
    //printResult<int>(Atview, N);
    //printResult<int>(Btview, N);
  //----------------------------------------------------------------------------//


  printf("using existing RAJA shared memory... \n");

  //Goal is to used this... 
  using seq_shmem_t = RAJA::ShmemTile<RAJA::seq_shmem,
                                      int,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::SizeList<TILE_DIM, TILE_DIM>,
                                      decltype(iSpace)>; 
  //seq_shmem_t RAJA_Shmem;
  using RAJAMemory = RAJA::SharedMemWrapper<seq_shmem_t>;
  RAJAMemory rajaTile;
      
  RAJA::kernel_param<KERNEL_EXEC_POL>(iSpace, 
                                      RAJA::make_tuple(rajaTile),


      [=] (int tx, int ty, int bx, int by, RAJAMemory &rajaTile) {
         
           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
           (*rajaTile.SharedMem)(ty,tx)  = Aview(row, col);
        },

      //read from shared mem
       [=] (int tx, int ty, int bx, int by, RAJAMemory &rajaTile) {
           
           int col = by * TILE_DIM + tx;  // Transposed matrix column index
           int row = bx * TILE_DIM + ty;  // Transposed matrix row index
           Atview(row, col) = (*rajaTile.SharedMem)(tx,ty);
        });                                         


    checkResult<int>(Atview, N);



  //
  // Clean up.
  //
  memoryManager::deallocate(A);
  memoryManager::deallocate(At);

  memoryManager::deallocate(B);
  memoryManager::deallocate(Bt);

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
}


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
