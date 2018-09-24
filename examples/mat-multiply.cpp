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

//Matrix A size : N x M
//Matrix B size : M x P
//Matrix C size : N x P

const int M = 100;
const int N = 120;
const int P = 15;

// Define TILE dimensions
//
const int TILE_DIM = 16;

//
// Define bounds for inner and outer loops
//
const int inner_Dim0 = TILE_DIM;
const int inner_Dim1 = TILE_DIM;

const int windowIter = (M-1)/TILE_DIM+1;

const int outer_Dim0 = (P-1)/TILE_DIM+1;
const int outer_Dim1 = (N-1)/TILE_DIM+1;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA shared memory matrix multiplication example...\n";

  //
  // Allocate matrix data
  //
  int *A      = memoryManager::allocate<int>(N * M);
  int *B      = memoryManager::allocate<int>(M * P);
  int *C      = memoryManager::allocate<int>(N * P);
  int *C_sol  = memoryManager::allocate<int>(N * P);

  //
  // In the following implementations of shared matrix transpose, we
  // use RAJA 'View' objects to access the matrix data. A RAJA view
  // holds a pointer to a data array and enables multi-dimensional indexing
  // into the data.
  //
  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N, M);
  RAJA::View<int, RAJA::Layout<DIM>> Bview(B, M, P);
  RAJA::View<int, RAJA::Layout<DIM>> Cview(C, N, P);
  RAJA::View<int, RAJA::Layout<DIM>> C_solview(C, N, P);

  //
  // Initialize matrix data
  //
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < M; ++col) {
      Aview(row, col) = col;
    }
  }

  for(int row = 0; row < M; ++row) {
    for(int col = 0; col < P; ++col) {
      Bview(row, col) = col;
    }
  }

  //----------------------------------------------------------------------------//
  std::cout << "\n Running C-version of shared matrix multiplication with window...\n";
  //
  // (0) Outer loops to iterate over tiles
  //
  for (int by = 0; by < outer_Dim1; ++by) {
    for (int bx = 0; bx < outer_Dim0; ++bx) {

      int aShared[TILE_DIM][TILE_DIM]; // shared memory
      int bShared[TILE_DIM][TILE_DIM]; // shared memory
      int pValue[TILE_DIM][TILE_DIM]; //thread private value

      //
      // (1) Initialize thread private value
      //
      for(int ty = 0; ty < inner_Dim1; ++ty){
        for(int tx = 0; tx < inner_Dim0; ++tx){
          pValue[ty][tx] = 0.0;
        }
      }

      //Loop to slide window across the matrix
      for(int i = 0; i < windowIter; ++i) {

        //
        // (2) Inner loops to load data into the tile
        //
        for (int ty = 0; ty < inner_Dim1; ++ty) {
          for (int tx = 0; tx < inner_Dim0; ++tx) {

            int col = bx * TILE_DIM + tx;  // Matrix column index
            int row = by * TILE_DIM + ty;  // Matrix row index

            if(row < N && ((i*TILE_DIM + tx) < M)){
              aShared[ty][tx] = Aview(row, ((i*TILE_DIM+tx) ));
            }else{
              aShared[ty][tx] = 0.0;
            }

            if( col < P && ((i*TILE_DIM + ty) < M) ){
              bShared[ty][tx] = Bview((i*TILE_DIM + ty), col);
            }else{
              bShared[ty][tx] = 0.0;
            }

          }
        }
        //Syncthreads

        //
        // (3) Matrix mutiply
        //
        for (int ty = 0; ty < inner_Dim1; ++ty) {
          for (int tx = 0; tx < inner_Dim0; ++tx) {

            for(int j=0; j<TILE_DIM; ++j){
              pValue[ty][tx] += aShared[ty][j]*bShared[j][tx];
            }

          }
        }

      }//loop to slide window across matrix

      //
      // (4) Write out to global matrix
      //
      for (int ty = 0; ty < inner_Dim1; ++ty) {
        for (int tx = 0; tx < inner_Dim0; ++tx) {

          int row = by * TILE_DIM + ty;  // Matrix row index
          int col = bx * TILE_DIM + tx;  // Matrix column index

          if(row < N && col < P)
            Cview(row, col) = pValue[ty][tx];
        }
      }

    }
  }

  //----------------------------------------------------------------------------//
  //RAJA Shared memory version...
  printf("  Running RAJA shared memory version ... \n");

  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                     RAJA::RangeSegment(0, windowIter),
                     RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1));

  using SharedTile = RAJA::SharedMem<int, TILE_DIM, TILE_DIM>;
  using Shmem = RAJA::SharedMemWrapper<SharedTile>;
  using threadPriv = RAJA::SharedMemWrapper<SharedTile>;

  Shmem aShared, bShared; //memory to be shared between threads
  threadPriv pVal; //thread private value

  using KERNEL_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,

          RAJA::statement::CreateShmem<

            //Initalize thread private value
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                                   RAJA::statement::Lambda<0> > >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::loop_exec,

               //Load matrix into tile
               RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<0, 1>,
                                     RAJA::statement::Lambda<1>
                                     >,

             //perform matrix multiplcation
             RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::statement::Lambda<2>
                                      >
            >, //sliding window

            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                                   RAJA::statement::Lambda<3> > >
         > //Create shared memory
        >//For 3
       >//For 4
      >; //close policy list

  RAJA::kernel_param<KERNEL_EXEC_POL>(iSpace,
                                      RAJA::make_tuple(aShared, bShared, pVal),

    //
    // (1) Intialize thread private value
    //
    [=] (int tx, int ty, int , int , int , Shmem &,  Shmem &, threadPriv &pVal) {

       (*pVal.SharedMem)(ty,tx) = 0.0;

     },

   //
   // (2) Inner loops to load data into the tile
   //
  [=] (int tx, int ty, int i, int bx, int by, Shmem &aShared,  Shmem &bShared, threadPriv &) {

   int row = by * TILE_DIM + ty;  // Matrix row index
   int col = bx * TILE_DIM + tx;  // Matrix column index

   //Load tile for A
   if( row < N && ((i*TILE_DIM + tx) < M) ){
     (*aShared.SharedMem)(ty,tx) = Aview(row, (i*TILE_DIM+tx)); //A[row*M + i*TILE_DIM + tx];
   }else{
     (*aShared.SharedMem)(ty,tx) = 0.0;
   }

   //Load tile for B
   if( col < P && ((i*TILE_DIM + ty) < M) ){
     (*bShared.SharedMem)(ty, tx) = Bview((i*TILE_DIM + ty), col);
   }else{
     (*bShared.SharedMem)(ty, tx) = 0.0;
   }

  },

  //
  // (3) Matrix mutiply
  //
  [=] (int tx, int ty, int , int , int , Shmem &aShared,  Shmem &bShared, threadPriv & pVal) {

    for(int j=0; j<TILE_DIM; j++){
      (*pVal.SharedMem)(ty,tx) += (*aShared.SharedMem)(ty,j) * (*bShared.SharedMem)(j, tx);
    }

  },

 //
 // (4) Write out to global matrix
 //
 [=] (int tx, int ty, int , int bx, int by, Shmem &, Shmem &, threadPriv &pValue) {

   int row = by * TILE_DIM + ty;  // Matrix row index
   int col = bx * TILE_DIM + tx;  // Matrix column index

   if(row < N && col < P)
     Cview(row,col) = (*pValue.SharedMem)(ty,tx);

  });


  //Check result
  bool pass = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < P; ++col) {

      if(Cview(row, col) != C_solview(row, col)) pass = false;
    }
  }

  if(pass){
    printf("Pass! \n");
  }else{
    printf("Fail \n");
  }

  //
  // Clean up.
  //
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);
  memoryManager::deallocate(C_sol);

  std::cout << "\n DONE!...\n";

  return 0;
}
