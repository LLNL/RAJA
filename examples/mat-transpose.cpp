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
const int N_rows = 12;
const int N_cols = 12;



// Define TILE dimensions
//
const int TILE_DIM = 4;

//
// Define bounds for inner and outer loops
//
const int inner_Dim0 = TILE_DIM;
const int inner_Dim1 = TILE_DIM;

const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;


//
// Function for checking results
//
template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_rows, int N_cols);

//
// Function for printing results
//
template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_rows, int N_cols);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector addition example...\n";

  //
  // Allocate matrix data
  //
  int *A  = memoryManager::allocate<int>(N_rows * N_cols);
  int *At = memoryManager::allocate<int>(N_rows * N_cols);

  int *B  = memoryManager::allocate<int>(N_rows * N_cols);
  int *Bt = memoryManager::allocate<int>(N_rows * N_cols);

  //
  // In the following implementations of shared matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  RAJA::View<int, RAJA::Layout<DIM>> Bview(B, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Btview(Bt, N_cols, N_rows);

  //
  // Initialize matrix data
  //
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = col;
    }
  }

  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of shared matrix transpose...\n";

  std::memset(At, 0, N_rows * N_cols * sizeof(int));
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
          if(row < N_rows && col < N_cols){
            TILE[ty][tx] = Aview(row, col);
          }
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
          if(row < N_cols && col < N_rows){
            Atview(row, col) = TILE[tx][ty];
          }

        }
      }
    }
  }

  checkResult<int>(Atview, N_rows, N_cols);
  //printResult<int>(Atview, N_cols, N_rows);
  //----------------------------------------------------------------------------//


  //----------------------------------------------------------------------------//
  //RAJA Shared memory version...
  printf("Run RAJA shared memory version \n");

  std::memset(At, 0, N_rows * N_cols * sizeof(int));

  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                     RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1));


  //Toy shared memory object - For proof of concept.
  using SharedTile = RAJA::SharedMem<int, TILE_DIM, TILE_DIM>;
  using mySharedMemory = RAJA::SharedMemWrapper<SharedTile>;
  mySharedMemory myTile;
  mySharedMemory myTile2;

  using KERNEL_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::For<3, RAJA::loop_exec,
        RAJA::statement::For<2, RAJA::loop_exec,

                             RAJA::statement::CreateShmem<

         RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                   RAJA::ArgList<0, 1>,
                                   RAJA::statement::Lambda<0>
                                   >,

         RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                   RAJA::ArgList<0, 1>,
                                   RAJA::statement::Lambda<1>
                                   >
                               > //Close shared memory scope
        >//for 2
       >//for 3
      >; //close policy list

  RAJA::kernel_param<KERNEL_EXEC_POL>(iSpace,
                                      RAJA::make_tuple(myTile, myTile2),

      [=] (int tx, int ty, int bx, int by, mySharedMemory &myTile,  mySharedMemory &myTile2) {

           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
          if(row < N_rows && col < N_cols){
            (*myTile.SharedMem)(ty,tx)  = Aview(row, col);
            (*myTile2.SharedMem)(ty,tx) = Bview(row, col);
          }
        },

      //read from shared mem
       [=] (int tx, int ty, int bx, int by, mySharedMemory &myTile, mySharedMemory &myTile2) {

           int col = by * TILE_DIM + tx;  // Transposed matrix column index
           int row = bx * TILE_DIM + ty;  // Transposed matrix row index
           if(row < N_cols && col < N_rows){
             Atview(row, col) = (*myTile.SharedMem)(tx,ty);
             Btview(row, col) = (*myTile2.SharedMem)(tx,ty);
           }
        });




    checkResult<int>(Atview, N_rows,N_cols);
    checkResult<int>(Btview, N_rows,N_cols);
    //printResult<int>(Atview, N);
    //printResult<int>(Btview, N);
  //----------------------------------------------------------------------------//


  printf("using existing RAJA shared memory... \n");


  //RAJA Shared Memory
  using cpu_shmem_t = RAJA::ShmemTile<RAJA::cpu_shmem,
                                      int,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::SizeList<TILE_DIM, TILE_DIM>,
                                      decltype(iSpace)>;
  using RAJAMemory = RAJA::SharedMemWrapper<cpu_shmem_t>;
  RAJAMemory rajaTile;

  RAJA::kernel_param<KERNEL_EXEC_POL>(iSpace,
                                      RAJA::make_tuple(rajaTile),


      //Load shared memory
      [=] (int tx, int ty, int bx, int by, RAJAMemory &rajaTile) {

           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
           if(row < N_rows && col < N_cols){
             (*rajaTile.SharedMem)(ty,tx)  = Aview(row, col);
           }
        },

      //Read from shared mem
       [=] (int tx, int ty, int bx, int by, RAJAMemory &rajaTile) {

           int col = by * TILE_DIM + tx;  // Transposed matrix column index
           int row = bx * TILE_DIM + ty;  // Transposed matrix row index
          if(row < N_cols && col < N_rows){
            Atview(row, col) = (*rajaTile.SharedMem)(tx,ty);
          }
        });

  checkResult<int>(Atview, N_rows, N_cols);


  //----------------------------------------------------------------------------//
  //New shared memory object version
  printf("Run new RAJA shared memory object version \n");
  std::memset(At, 0, N_rows * N_cols * sizeof(int));


  using SharedTile2 = RAJA::SharedMem2<int, RAJA::SizeList<TILE_DIM,TILE_DIM>>;
  SharedTile2::layout_t::print();

  //Need a wrapper:
  using SharedWrapperV2 = RAJA::SharedMemWrapper<SharedTile2>;
  SharedWrapperV2 shmemMem;


  using KERNEL_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::For<3, RAJA::loop_exec,
        RAJA::statement::For<2, RAJA::loop_exec,

                             RAJA::statement::CreateShmem<

         RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                   RAJA::ArgList<0, 1>,
                                   RAJA::statement::Lambda<0>
                                   >,

         RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                   RAJA::ArgList<0, 1>,
                                   RAJA::statement::Lambda<1>
                                   >
                               > //Close shared memory scope
        >//for 2
       >//for 3
      >; //close policy list

  RAJA::kernel_param<KERNEL_EXEC_POL>(iSpace,
                                      RAJA::make_tuple(shmemMem),

    [=] (int tx, int ty, int bx, int by, SharedWrapperV2 &shmemMem) {

           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
         if(row < N_rows && col < N_cols){
           shmemMem(ty,tx)  = Aview(row, col);
          }
        },

      //read from shared mem
    [=] (int tx, int ty, int bx, int by, SharedWrapperV2 &shmemMem) {

           int col = by * TILE_DIM + tx;  // Transposed matrix column index
           int row = bx * TILE_DIM + ty;  // Transposed matrix row index
           if(row < N_cols && col < N_rows){
             Atview(row, col) = shmemMem(tx,ty);
           }
        });




    checkResult<int>(Atview, N_rows,N_cols);
    checkResult<int>(Btview, N_rows,N_cols);
    //printResult<int>(Atview, N);
    //printResult<int>(Btview, N);
  //----------------------------------------------------------------------------//



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
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_rows, int N_cols)
{
  bool match = true;
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
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
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Atview, int N_rows, int N_cols)
{
  std::cout << std::endl;
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      std::cout << "At(" << row << "," << col << ") = " << Atview(row, col)
                << std::endl;
    }
  }
  std::cout << std::endl;
}
