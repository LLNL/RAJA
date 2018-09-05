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


  //
  // Define TILE dimensions
  //
//  const int TILE_DIM = 2;

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

//using RAJA::int;
template<typename LAMBDA>
void forall(int beg, int end, LAMBDA&& body)
{

  for(int i=beg; i<end; ++i){
    body(i);
  }

} 

template<typename LAMBDA>
void forall(int beg0, int end0,
            int beg1, int end1,
            LAMBDA&& body)
{
#pragma omp parallel for
  for(int by=beg1; by<end1; ++by){
    for(int bx=beg0; bx<end0; ++bx){
      body(bx,by);
    }
  }

} 


template<typename LAMBDA0, typename LAMBDA1, typename LAMBDA2>
void forall(int obeg, int oend,
            int ibeg, int iend,
            LAMBDA0&& body0, LAMBDA1&& body1, LAMBDA2&& body2)
{

  std::cout<<"Running Art's forall"<<std::endl;

  for(int by=obeg; by<oend; ++by){
    for(int bx=obeg; bx<oend; ++bx){

      body0(bx,by);
 
      int TILE[TILE_DIM][TILE_DIM];

#pragma omp parallel for     
      for(int ty=ibeg; ty<iend; ++ty){
        for(int tx=ibeg; tx<iend; ++tx){
          body1(tx,ty, bx, by, TILE);
        }
      }

#pragma omp parallel for     
      for(int ty=ibeg; ty<iend; ++ty){
        for(int tx=ibeg; tx<iend; ++tx){
          body2(tx,ty, bx, by, TILE); 
        }
      }

    }
  }

} 

/*
struct sharedMem{
  int array[TILE_DIM][TILE_DIM];    
  sharedMem(){ printf("Created shared memory \n");};
  int &operator()(int row, int col)
  { return array[row][col];};
};
*/



int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector addition example...\n";

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

  std::memset(At, 0, N * N * sizeof(int));

  forall(0, outer_Dim0, 0, outer_Dim1,         
         [=] (int bx, int by) {
          
          int TILE[TILE_DIM][TILE_DIM];

          forall(0, inner_Dim1, [=, &TILE] (int ty) {
              forall(0, inner_Dim0, [=, &TILE] (int tx) {
                  
                  int col = bx * TILE_DIM + tx;  // Matrix column index
                  int row = by * TILE_DIM + ty;  // Matrix row index
                  TILE[ty][tx] = Aview(row, col);

                });
            });
          
          forall(0, inner_Dim1, [=, &TILE] (int ty) {
              forall(0, inner_Dim0, [=, &TILE] (int tx) {
      
                  int col = by * TILE_DIM + tx;  // Transposed matrix column index
                  int row = bx * TILE_DIM + ty;  // Transposed matrix row index
                  Atview(row, col) = TILE[tx][ty];

                });
            });
    });


    checkResult<int>(Atview, N);
  //printResult<int>(Atview, N);
  //----------------------------------------------------------------------------//

  std::memset(At, 0, N * N * sizeof(int));

  forall(0, outer_Dim0,
         0, inner_Dim0,

         //nothing happens here
         [=] (int bx, int by) {
          
         }, 

         //Create TILE HERE
         
         //Load data into shared memory
         [=] (int tx, int ty, int bx, int by, int TILE[TILE_DIM][TILE_DIM] ) {

           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
           TILE[ty][tx] = Aview(row, col);
        },

         //Read data from shared memory
         [=] (int tx, int ty, int bx, int by, int TILE[TILE_DIM][TILE_DIM]) {
           
           int col = by * TILE_DIM + tx;  // Transposed matrix column index
           int row = bx * TILE_DIM + ty;  // Transposed matrix row index
           Atview(row, col) = TILE[tx][ty];

        });   

    checkResult<int>(Atview, N);
  //printResult<int>(Atview, N);
  //----------------------------------------------------------------------------//


  std::memset(At, 0, N * N * sizeof(int));

  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                     RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1));

  using seq_shmem_t = RAJA::ShmemTile<RAJA::seq_shmem,
                                      int,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::SizeList<TILE_DIM, TILE_DIM>,
                                      decltype(iSpace)>; 
  seq_shmem_t RAJA_Shmem;

  //int TILE 
  using KERNEL_EXEC_POL = 
    RAJA::KernelPolicy<      
      RAJA::statement::For<3, RAJA::loop_exec,
        RAJA::statement::For<2, RAJA::loop_exec,
                             RAJA::statement::Lambda<0>,

                                                          
         RAJA::statement::CreateShmem,

           RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
             RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
               RAJA::statement::Lambda<1>
           > //0
          >,//1


           RAJA::statement::For<1, RAJA::loop_exec,
            RAJA::statement::For<0, RAJA::loop_exec,
             RAJA::statement::Lambda<2>
          >// 0
         >//1

        >//for 2
       >//for 3
      >; //close policy list


  PtrWrapper<sharedMem> myTile;

  std::cout<<"launching raja kernel"<<std::endl;
  RAJA::kernel_param<KERNEL_EXEC_POL>(iSpace, 
                                      RAJA::make_tuple(myTile),

      [=] (int , int , int , int , PtrWrapper<sharedMem> myTile ) {
         printf("Create shared memory after here! \n");
        },

                                      
     [=] (int tx, int ty, int bx, int by, PtrWrapper<sharedMem> myTile ) {
         
           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
           (*myTile.myData)(ty,tx) = Aview(row, col);
        },

      //read from shared mem
       [=] (int tx, int ty, int bx, int by, PtrWrapper<sharedMem> myTile ) {
           
           int col = by * TILE_DIM + tx;  // Transposed matrix column index
           int row = bx * TILE_DIM + ty;  // Transposed matrix row index
           //Atview(row, col) = myTile.myData(tx,ty);           
           Atview(row, col) = (*myTile.myData)(tx,ty);
        });                                         




    checkResult<int>(Atview, N);
  //printResult<int>(Atview, N);
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
