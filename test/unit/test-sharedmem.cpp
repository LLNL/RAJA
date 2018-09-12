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


using namespace RAJA;
using namespace RAJA::statement;


#if defined(RAJA_ENABLE_OPENMP)
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


#endif
