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

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cassert>

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

using namespace RAJA;
using namespace RAJA::statement;

#if defined(RAJA_ENABLE_OPENMP)
TEST(Shared, MatrixMultiplication){

  const int DIM = 2;

  //Matrix A size: N x M
  //Matrix B size: M x P
  //Result C size: N x P

  const int N = 9;
  const int M = 12;
  const int P = 15;

  const int TILE_DIM = 4;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int windowIter = (M-1)/TILE_DIM+1;
  const int outer_Dim0 = (P-1)/TILE_DIM+1;
  const int outer_Dim1 = (N-1)/TILE_DIM+1;

  int *A  = new int[N * M];
  int *B  = new int[M * P];
  int *C  = new int[N * P];
  int *C_sol  = new int[N * P];


  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N, M);
  RAJA::View<int, RAJA::Layout<DIM>> Bview(B, M, P);
  RAJA::View<int, RAJA::Layout<DIM>> Cview(C, N, P);
  RAJA::View<int, RAJA::Layout<DIM>> C_solView(C_sol, N, P);

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < M; ++col) {
      Aview(row, col) = col;
    }
  }

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < P; ++col) {
      Bview(row, col) = col;
    }
  }

  for(int r=0; r<N; ++r){
    for(int c=0; c<P; ++c){
      int dot = 0.0;
      for(int k=0; k<M; ++k){
        dot += Aview(r,k)*Bview(k,c);
      }
      C_solView(r,c) = dot;
    }
  }


  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                     RAJA::RangeSegment(0, windowIter),
                     RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1));

  //Toy shared memory object - For proof of concept.
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

  [=] (int tx, int ty, int , int , int , Shmem &,  Shmem &, threadPriv &pVal) {

       //I would like this to behave like a thread private variable
       (*pVal.SharedMem)(ty,tx) = 0.0;

     },

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

  //read from shared mem
  [=] (int tx, int ty, int , int , int , Shmem &aShared,  Shmem &bShared, threadPriv & pVal) {

    //Matrix multiply
    for(int j=0; j<TILE_DIM; j++){
      (*pVal.SharedMem)(ty,tx) += (*aShared.SharedMem)(ty,j) * (*bShared.SharedMem)(j, tx);
    }

  },

 //If in range write out
 [=] (int tx, int ty, int , int bx, int by, Shmem &, Shmem &, threadPriv &pValue) {

   int row = by * TILE_DIM + ty;  // Matrix row index
   int col = bx * TILE_DIM + tx;  // Matrix column index

   if(row < N && col < P)
     Cview(row,col) = (*pValue.SharedMem)(ty,tx);

  });

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < P; ++col) {
      ASSERT_FLOAT_EQ(Cview(row,col), C_solView(row,col));
    }
  }

  delete [] A;
  delete [] B;
  delete [] C;
  delete [] C_sol;
}


TEST(Shared, MatrixTranposeUserSharedInnerCollapsed){

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;
  const int TILE_DIM = 16;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  int *A  = new int[N_rows * N_cols];
  int *At = new int[N_rows * N_cols];

  int *B  = new int[N_rows * N_cols];
  int *Bt = new int[N_rows * N_cols];


  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  RAJA::View<int, RAJA::Layout<DIM>> Bview(B, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Btview(Bt, N_cols, N_rows);


  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = col;
    }
  }


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
                               >
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


  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(Atview(col,row), col);
      ASSERT_FLOAT_EQ(Btview(col,row), col);
    }
  }

  delete [] A;
  delete [] At;
  delete [] B;
  delete [] Bt;

}

TEST(Shared, MatrixTranposeUserSharedInner){

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;
  const int TILE_DIM = 16;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  int *A  = new int[N_rows * N_cols];
  int *At = new int[N_rows * N_cols];

  int *B  = new int[N_rows * N_cols];
  int *Bt = new int[N_rows * N_cols];


  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  RAJA::View<int, RAJA::Layout<DIM>> Bview(B, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Btview(Bt, N_cols, N_rows);


  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = col;
    }
  }

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

            RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
             >,
            RAJA::statement::For<1, RAJA::loop_exec,
           RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
                                RAJA::statement::Lambda<1>
           >
          >
         > //close shared mem window
        > //2
       >//3
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


  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(Atview(col,row), col);
      ASSERT_FLOAT_EQ(Btview(col,row), col);
    }
  }

  delete [] A;
  delete [] At;
  delete [] B;
  delete [] Bt;

}


TEST(Shared, MatrixTranposeUserSharedOuter){

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;
  const int TILE_DIM = 16;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  int *A  = new int[N_rows * N_cols];
  int *At = new int[N_rows * N_cols];

  int *B  = new int[N_rows * N_cols];
  int *Bt = new int[N_rows * N_cols];


  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  RAJA::View<int, RAJA::Layout<DIM>> Bview(B, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Btview(Bt, N_cols, N_rows);


  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = col;
    }
  }


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
      RAJA::statement::For<3, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<2, RAJA::loop_exec,

          RAJA::statement::CreateShmem<

            RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
             >,
            RAJA::statement::For<1, RAJA::loop_exec,
           RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
                                RAJA::statement::Lambda<1>
           >
          >
         > //close shared mem window
        > //2
       >//3
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


  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(Atview(col,row), col);
      ASSERT_FLOAT_EQ(Btview(col,row), col);
    }
  }

  delete [] A;
  delete [] At;
  delete [] B;
  delete [] Bt;
}


TEST(Shared, MatrixTranposeRAJAShared){

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;
  const int TILE_DIM = 16;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  int *A  = new int[N_rows * N_cols];
  int *At = new int[N_rows * N_cols];

  int *B  = new int[N_rows * N_cols];
  int *Bt = new int[N_rows * N_cols];


  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  RAJA::View<int, RAJA::Layout<DIM>> Bview(B, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Btview(Bt, N_cols, N_rows);


  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = col;
    }
  }

  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                     RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1));


  using seq_shmem_t = RAJA::ShmemTile<RAJA::seq_shmem,
                                      int,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::SizeList<TILE_DIM, TILE_DIM>,
                                      decltype(iSpace)>;
  using RAJAMemory = RAJA::SharedMemWrapper<seq_shmem_t>;
  RAJAMemory rajaTile;


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
                               > //close shared memory scope
        >//for 2
       >//for 3
      >; //close policy list

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


  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(Atview(col,row), col);
    }
  }

  delete [] A;
  delete [] At;
}


#if defined(RAJA_ENABLE_CUDA)
//Exisiting version
CUDA_TEST(Shared, MatrixTranposeCUDARAJAShared){
  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;
  const int TILE_DIM = 16;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  int *A;
  int *At;

  cudaMallocManaged(&A,  sizeof(int) * N_rows * N_cols);
  cudaMallocManaged(&At, sizeof(int) * N_rows * N_cols);

  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
    }
  }

  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                     RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1));

  using RAJAMemory = RAJA::ShmemTile<RAJA::cuda_shmem, int, RAJA::ArgList<0, 1>, RAJA::SizeList<TILE_DIM, TILE_DIM>,decltype(iSpace)>;

  RAJAMemory rajaTile;

  using KERNEL_EXEC_POL =
    RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::For<3, RAJA::cuda_block_exec,
        RAJA::statement::For<2, RAJA::cuda_block_exec,

          RAJA::statement::SetShmemWindow<

            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::For<0, RAJA::cuda_thread_exec,
                RAJA::statement::Lambda<0>
                                   >
                                 >,
            RAJA::statement::CudaSyncThreads,
              RAJA::statement::For<1, RAJA::cuda_thread_exec,
                RAJA::statement::For<0, RAJA::cuda_thread_exec,
                                   RAJA::statement::Lambda<1> > >
              > //close shared memory scope
            >//for 2
        >//for 3
        > //CudaKernel
      >; //close policy list

  RAJA::kernel_param<KERNEL_EXEC_POL>(iSpace,
                                      RAJA::make_tuple(rajaTile),

      //Load shared memory
    [=] RAJA_DEVICE (int tx, int ty, int bx, int by, RAJAMemory &rajaTile) {

           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
           if(row < N_rows && col < N_cols){
              rajaTile(ty,tx)  = Aview(row, col);
           }
     }
      //Read from shared mem
   ,[=] RAJA_DEVICE (int tx, int ty, int bx, int by, RAJAMemory &rajaTile) {

       int col = by * TILE_DIM + tx;  // Transposed matrix column index
       int row = bx * TILE_DIM + ty;  // Transposed matrix row index
       if(row < N_cols && col < N_rows){
         Atview(row, col) = rajaTile(tx,ty);
       }
     }
	);


  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(Atview(col,row), col);
    }
  }
  cudaFree(A);
  cudaFree(At);
}



CUDA_TEST(Shared, MatrixTranposeMyCUDAShared){

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;
  const int TILE_DIM = 16;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  int *A, *B;
  int *At, *Bt;

  cudaMallocManaged(&A,  sizeof(int) * N_rows * N_cols);
  cudaMallocManaged(&At, sizeof(int) * N_rows * N_cols);

  cudaMallocManaged(&B,  sizeof(int) * N_rows * N_cols);
  cudaMallocManaged(&Bt, sizeof(int) * N_rows * N_cols);

  RAJA::View<int, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  RAJA::View<int, RAJA::Layout<DIM>> Bview(B, N_rows, N_cols);
  RAJA::View<int, RAJA::Layout<DIM>> Btview(Bt, N_cols, N_rows);

  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = row;
    }
  }

  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                     RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1));

  //using RAJAMemory = RAJA::ShmemTile<RAJA::cuda_shmem, int, RAJA::ArgList<0, 1>, RAJA::SizeList<TILE_DIM, TILE_DIM>,decltype(iSpace)>;

  using SharedTile = RAJA::SharedMem<int,TILE_DIM,TILE_DIM>;
  using RAJAMemory = RAJA::SharedMemWrapper<SharedTile>;
  RAJAMemory rajaTile, rajaTile2;


  using KERNEL_EXEC_POL =
    RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::For<3, RAJA::cuda_block_exec,
        RAJA::statement::For<2, RAJA::cuda_block_exec,


          RAJA::statement::CreateShmem<

            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::For<0, RAJA::cuda_thread_exec,
                RAJA::statement::Lambda<0>
                                   >
                                 >,
            RAJA::statement::CudaSyncThreads,
              RAJA::statement::For<1, RAJA::cuda_thread_exec,
                RAJA::statement::For<0, RAJA::cuda_thread_exec,
                                   RAJA::statement::Lambda<1> > >
              > //close shared memory scope
            >//for 2
        >//for 3
        > //CudaKernel
      >; //close policy list

  RAJA::kernel_param<KERNEL_EXEC_POL>(iSpace,
                                      RAJA::make_tuple(rajaTile, rajaTile2),

      //Load shared memory
    [=] RAJA_DEVICE (int tx, int ty, int bx, int by, RAJAMemory &rajaTile, RAJAMemory &rajaTile2) {

           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
           if(row < N_rows && col < N_cols){
             (*rajaTile.SharedMem)(ty,tx)  = Aview(row, col);
             (*rajaTile2.SharedMem)(ty,tx) = Bview(row, col);
           }
     }
      //Read from shared mem
    ,[=] RAJA_DEVICE (int tx, int ty, int bx, int by, RAJAMemory &rajaTile, RAJAMemory &rajaTile2) {

       int col = by * TILE_DIM + tx;  // Transposed matrix column index
       int row = bx * TILE_DIM + ty;  // Transposed matrix row index
       if(row < N_cols && col < N_rows){
         Atview(row, col) = (*rajaTile.SharedMem)(tx,ty);
         Btview(row, col) = (*rajaTile2.SharedMem)(tx,ty);
       }
     }
	);


  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(Atview(col,row), col);
      ASSERT_FLOAT_EQ(Btview(col,row), row);
    }
  }
  cudaFree(A);
  cudaFree(At);
  cudaFree(B);
  cudaFree(Bt);
}

#endif//CUDA bounds


#endif
