//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

#include "camp/camp.hpp"
#include "camp/concepts.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

using namespace RAJA;
using namespace RAJA::statement;

//Define tile size ( TILE_DIM x TILE_DIM )
//Matrix transpose and matrix multiplication
//are carried out via tiling algorithms
RAJA_INDEX_VALUE(TX, "TX");
RAJA_INDEX_VALUE(TY, "TY");

const int TILE_DIM = 16;

template <typename NestedPolicy>
class TypedLocalMem : public ::testing::Test
{

  virtual void SetUp() {}
  virtual void TearDown() {}
};
TYPED_TEST_CASE_P(TypedLocalMem);

CUDA_TYPED_TEST_P(TypedLocalMem, Basic)
{
  using Pol = at_v<TypeParam, 0>;

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  double *A, *B;
#if defined(RAJA_ENABLE_CUDA)
  size_t Arr_sz = N_rows * N_cols;
  cudaMallocManaged(&A,  sizeof(double) * Arr_sz);
  cudaMallocManaged(&B, sizeof(double)  * Arr_sz);
#else
  A  = new double[N_rows * N_cols];
  B  = new double[N_rows * N_cols];
#endif

  RAJA::TypedView<double, RAJA::Layout<DIM>, TY, TX> Aview(A, N_rows, N_cols);
  RAJA::TypedView<double, RAJA::Layout<DIM>, TY, TX> Bview(B, N_rows, N_cols);

  for (int row = 0; row < N_rows; ++row) {
    for (int col= 0 ; col < N_cols; ++col) {
      A[col + N_cols*row] = col;
    }
  }

  using SharedTile = TypedLocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<TILE_DIM,TILE_DIM>, TY, TX>;
  SharedTile myTile, myTile2;

  const TX TX_TILE_DIM(16);
  const TY TY_TILE_DIM(16);

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::TypedRangeSegment<TX>(0, inner_Dim0), RAJA::TypedRangeSegment<TY>(0,inner_Dim1),
                                           RAJA::TypedRangeSegment<TX>(0, outer_Dim0), RAJA::TypedRangeSegment<TY>(0,outer_Dim1)),
                          RAJA::make_tuple(myTile, myTile2),

  //Load data into shared memory
  [=] RAJA_HOST_DEVICE (TX tx, TY ty, TX bx, TY by, SharedTile &myTile, SharedTile &) {

    TX col = bx * TX_TILE_DIM + tx;  // Matrix column index
    TY row = by * TY_TILE_DIM + ty;  // Matrix row index

    if(row < N_rows && col < N_cols){
      myTile(ty,tx)   = Aview(row, col);
    }

  },

  //read from shared mem
  [=] RAJA_HOST_DEVICE (TX tx, TY ty, TX bx, TY by, SharedTile &myTile, SharedTile &) {

    TX col = bx * TX_TILE_DIM + tx;  // Matrix column index
    TY row = by * TY_TILE_DIM + ty;  // Matrix row index

    if(row < N_rows && col < N_cols){
      Bview(row, col) = myTile(ty, tx);
    }

  });

  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(B[col + row*N_cols], A[col + row*N_cols]);
    }
  }

#if defined(RAJA_ENABLE_CUDA)
  cudaFree(A);
  cudaFree(B);
#else
  delete [] A;
  delete [] B;
#endif
}

REGISTER_TYPED_TEST_CASE_P(TypedLocalMem, Basic);

//
//Matrix transpose example - test all variants
//
template <typename NestedPolicy>
class MatTranspose : public ::testing::Test
{

  virtual void SetUp() {}
  virtual void TearDown() {}
};
TYPED_TEST_CASE_P(MatTranspose);

CUDA_TYPED_TEST_P(MatTranspose, Basic)
{

  using Pol = at_v<TypeParam, 0>;

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;
  const int TILE_DIM = 16;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  double *A, *At, *B, *Bt;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged(&A,  sizeof(double) * N_rows * N_cols);
  cudaMallocManaged(&At, sizeof(double) * N_rows * N_cols);
  cudaMallocManaged(&B,  sizeof(double) * N_rows * N_cols);
  cudaMallocManaged(&Bt, sizeof(double) * N_rows * N_cols);
#else
  A  = new double[N_rows * N_cols];
  At = new double[N_rows * N_cols];
  B  = new double[N_rows * N_cols];
  Bt = new double[N_rows * N_cols];
#endif

  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N_rows, N_cols);
  RAJA::View<double, RAJA::Layout<DIM>> Atview(At, N_cols, N_rows);

  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N_rows, N_cols);
  RAJA::View<double, RAJA::Layout<DIM>> Btview(Bt, N_cols, N_rows);


  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      Aview(row, col) = col;
      Bview(row, col) = col;
    }
  }


  using SharedTile = LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<TILE_DIM,TILE_DIM>>;

  SharedTile myTile, myTile2;

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                                           RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1)),
                          RAJA::make_tuple(myTile, myTile2),

  //Load data into shared memory
  [=] RAJA_HOST_DEVICE (int tx, int ty, int bx, int by, SharedTile &myTile, SharedTile &myTile2) {

    int col = bx * TILE_DIM + tx;  // Matrix column index
    int row = by * TILE_DIM + ty;  // Matrix row index

    if(row < N_rows && col < N_cols){
      myTile(ty,tx)  = Aview(row, col);
      myTile2(ty,tx) = Bview(row, col);
    }

  },

  //read from shared mem
  [=] RAJA_HOST_DEVICE (int tx, int ty, int bx, int by, SharedTile &myTile, SharedTile &myTile2) {

    int col = by * TILE_DIM + tx;  // Transposed matrix column index
    int row = bx * TILE_DIM + ty;  // Transposed matrix row index

    if(row < N_cols && col < N_rows){
      Atview(row, col) = myTile(tx,ty);
      Btview(row, col) = myTile2(tx,ty);
    }

  });

  //Check result
  for (int row = 0; row < N_rows; ++row) {
    for (int col = 0; col < N_cols; ++col) {
      ASSERT_FLOAT_EQ(Atview(col,row), col);
      ASSERT_FLOAT_EQ(Btview(col,row), col);
    }
  }


#if defined(RAJA_ENABLE_CUDA)
  cudaFree(A);
  cudaFree(At);
  cudaFree(B);
  cudaFree(Bt);
#else
  delete [] A;
  delete [] At;
  delete [] B;
  delete [] Bt;
#endif
}

REGISTER_TYPED_TEST_CASE_P(MatTranspose, Basic);


using SeqTypes =
  ::testing::Types<
  RAJA::list<
    RAJA::KernelPolicy<
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::For<2, RAJA::loop_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::loop_exec,
                RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::Lambda<0>
                                   >
                                 >,

                //Read data from shared memory
                RAJA::statement::For<1, RAJA::loop_exec,
                  RAJA::statement::For<0, RAJA::loop_exec,
                    RAJA::statement::Lambda<1> > >

              > //close shared memory scope
            >//for 2
        >//for 3
      > //kernel policy
    > //list
  >; //types
INSTANTIATE_TYPED_TEST_CASE_P(Seq, MatTranspose, SeqTypes);
INSTANTIATE_TYPED_TEST_CASE_P(Seq, TypedLocalMem, SeqTypes);


#if defined(RAJA_ENABLE_OPENMP)
using TestTypes =
  ::testing::Types<
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::For<3, RAJA::loop_exec,
        RAJA::statement::For<2, RAJA::loop_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1>,

           //Load data into shared memory
           RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<0, 1>,
                                     RAJA::statement::Lambda<0>
                                     >,

           //Read data from shared memory
           RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<0, 1>,
                                     RAJA::statement::Lambda<1>
                                     >
                                 >
        >//for 2
       >//for 3
       > //close policy
     > //close list

  ,RAJA::list<
      RAJA::KernelPolicy<
      RAJA::statement::For<3, RAJA::loop_exec,
        RAJA::statement::For<2, RAJA::loop_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1>,

           //Load data into shared memory
            RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
             >,

           //Read data from shared memory
            RAJA::statement::For<1, RAJA::loop_exec,
           RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
                                RAJA::statement::Lambda<1>
           >
          >
         > //close shared mem window
        > //2
       >//3
     >//close policy
    > //close list
  ,RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::For<3, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<2, RAJA::loop_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1>,

           //Load data into shared memory
           RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
             >,

           //Read data from shared memory
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<1>
           >
          >
         > //close shared mem window
        > //2
       >//3
      > //close policy list
     > //close list
  ,RAJA::list<
    RAJA::KernelPolicy<
           RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<2, 3>,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem,RAJA::ParamList<0,1>,

           //Load data into shared memory
           RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
             >,

           //Read data from shared memory
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<1>
           >
          >
         > //close shared mem window
       >//outer collapsed
      > //close policy list
     > //close list
   >;


INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, MatTranspose, TestTypes);
INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, TypedLocalMem, TestTypes);
#endif

#if defined(RAJA_ENABLE_CUDA)

using CUDATypes =
  ::testing::Types<
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<3, RAJA::cuda_block_y_loop,
          RAJA::statement::For<2, RAJA::cuda_block_x_loop,

           RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<0,1>,

             //Load data into shared memory
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<0>
                                   >
                                 >,
              RAJA::statement::CudaSyncThreads,

                //Read data from shared memory
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                    RAJA::statement::Lambda<1> > >
               ,RAJA::statement::CudaSyncThreads,
              > //close shared memory scope
            >//for 2
        >//for 3
        > //CudaKernel
      > //kernel policy
    > //list
  >; //types
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, MatTranspose, CUDATypes);
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, TypedLocalMem, CUDATypes);

#endif

template <typename NestedPolicy>
class MatMultiply : public ::testing::Test
{
  virtual void SetUp(){}
  virtual void TearDown(){}
};

TYPED_TEST_CASE_P(MatMultiply);

CUDA_TYPED_TEST_P(MatMultiply, shmem)
{

  using Tile_size0 = at_v<TypeParam, 0>;
  using Tile_size1 = at_v<TypeParam, 1>;
  using Pol = at_v<TypeParam, 2>;

  const int DIM = 2;

  //Matrix A size: N x M
  //Matrix B size: M x P
  //Result C size: N x P

  const int N = 150;
  const int M = 25;
  const int P = 95;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int windowIter = (M-1)/TILE_DIM+1;
  const int outer_Dim0 = (P-1)/TILE_DIM+1;
  const int outer_Dim1 = (N-1)/TILE_DIM+1;

  double *A, *B, *C, *C_sol;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged(&A,  sizeof(double) * N * M);
  cudaMallocManaged(&B,  sizeof(double) * M * P);
  cudaMallocManaged(&C,  sizeof(double) * N * P);
  cudaMallocManaged(&C_sol,  sizeof(double) * N * P);
#else
  A  = new double[N * M];
  B  = new double[M * P];
  C  = new double[N * P];
  C_sol  = new double[N * P];
#endif

  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, M);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, M, P);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, P);
  RAJA::View<double, RAJA::Layout<DIM>> C_solView(C_sol, N, P);

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


  using Shmem      = RAJA::LocalArray<double, RAJA::PERM_IJ, Tile_size0>;
  using ThreadPriv = RAJA::LocalArray<double, RAJA::PERM_IJ, Tile_size1>;

  Shmem aShared, bShared; //memory to be shared between threads
  ThreadPriv pVal; //iteration dependent data

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                                           RAJA::RangeSegment(0, windowIter),
                                           RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1)),
                          RAJA::make_tuple(aShared, bShared, pVal),

  [=] RAJA_HOST_DEVICE (int tx, int ty, int , int , int , Shmem &,  Shmem &, ThreadPriv &pVal) {

   pVal(ty,tx) = 0.0;

  },

  [=] RAJA_HOST_DEVICE (int tx, int ty, int i, int bx, int by, Shmem &aShared,  Shmem &bShared, ThreadPriv &) {

   int row = by * TILE_DIM + ty;  // Matrix row index
   int col = bx * TILE_DIM + tx;  // Matrix column index


   //Load a tile of A
   if( row < N && ((i*TILE_DIM + tx) < M) ){
     aShared(ty,tx) = Aview(row, (i*TILE_DIM+tx)); //A[row*M + i*TILE_DIM + tx];
   }else{
     aShared(ty,tx) = 0.0;
   }

   //Load a tile of B
   if( col < P && ((i*TILE_DIM + ty) < M) ){
     bShared(ty, tx) = Bview((i*TILE_DIM + ty), col);
   }else{
     bShared(ty, tx) = 0.0;
   }

  },

  //read from shared mem
  [=] RAJA_HOST_DEVICE (int tx, int ty, int , int , int , Shmem &aShared,  Shmem &bShared, ThreadPriv & pVal) {

    //Matrix multiply
    for(int j=0; j<TILE_DIM; j++){
      pVal(ty,tx) += aShared(ty,j) * bShared(j, tx);
    }

  },

 //If in range write out
 [=] RAJA_HOST_DEVICE (int tx, int ty, int , int bx, int by, Shmem &, Shmem &, ThreadPriv &pValue) {

   int row = by * TILE_DIM + ty;  // Matrix row index
   int col = bx * TILE_DIM + tx;  // Matrix column index

   if(row < N && col < P){
     Cview(row,col) = pValue(ty,tx);
    }

  });

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < P; ++col) {
      ASSERT_FLOAT_EQ(Cview(row,col), C_solView(row,col));
    }
  }


#if defined(RAJA_ENABLE_CUDA)
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C_sol);
#else
  delete [] A;
  delete [] B;
  delete [] C;
  delete [] C_sol;
#endif

}

REGISTER_TYPED_TEST_CASE_P(MatMultiply, shmem);


//
//Matrix multiplication with a scalar for accumulating the dot product
//Illustrates how to go between CPU and GPU code by change order of lambdas
//

template <typename NestedPolicy>
class MatMultiplyScalar : public ::testing::Test
{
  virtual void SetUp(){}
  virtual void TearDown(){}
};

TYPED_TEST_CASE_P(MatMultiplyScalar);

CUDA_TYPED_TEST_P(MatMultiplyScalar, shmem)
{

  using Tile_size0 = at_v<TypeParam, 0>;
  using Pol = at_v<TypeParam, 3>;

  const int DIM = 2;

  //Matrix A size: N x M
  //Matrix B size: M x P
  //Result C size: N x P

  const int N = 150;
  const int M = 25;
  const int P = 95;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int windowIter = (M-1)/TILE_DIM+1;
  const int outer_Dim0 = (P-1)/TILE_DIM+1;
  const int outer_Dim1 = (N-1)/TILE_DIM+1;

  double *A, *B, *C, *C_sol;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged(&A,  sizeof(double) * N * M);
  cudaMallocManaged(&B,  sizeof(double) * M * P);
  cudaMallocManaged(&C,  sizeof(double) * N * P);
  cudaMallocManaged(&C_sol,  sizeof(double) * N * P);
#else
  A  = new double[N * M];
  B  = new double[M * P];
  C  = new double[N * P];
  C_sol  = new double[N * P];
#endif

  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, M);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, M, P);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, P);
  RAJA::View<double, RAJA::Layout<DIM>> C_solView(C_sol, N, P);

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
      Cview(r,c) = 0.0;
      C_solView(r,c) = dot;
    }
  }

  using Shmem = RAJA::LocalArray<double, RAJA::PERM_IJ, Tile_size0>;

  Shmem aShared, bShared; //memory to be shared between threads

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                                           RAJA::RangeSegment(0, windowIter),
                                           RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1)),
                          RAJA::make_tuple(aShared, bShared, 0.0),

  [=] RAJA_HOST_DEVICE (int, int, int, int, int, Shmem &,  Shmem &, double & pVal) {

   pVal = 0.0;

  },

  [=] RAJA_HOST_DEVICE (int tx, int ty, int i, int bx, int by, Shmem &aShared,  Shmem &bShared, double &) {

   int row = by * TILE_DIM + ty;  // Matrix row index
   int col = bx * TILE_DIM + tx;  // Matrix column index


   //Load tile for A
   if( row < N && ((i*TILE_DIM + tx) < M) ){
     aShared(ty,tx) = Aview(row, (i*TILE_DIM+tx)); //A[row*M + i*TILE_DIM + tx];
   }else{
     aShared(ty,tx) = 0.0;
   }

   //Load tile for B
   if( col < P && ((i*TILE_DIM + ty) < M) ){
     bShared(ty, tx) = Bview((i*TILE_DIM + ty), col);
   }else{
     bShared(ty, tx) = 0.0;
   }

  },

  //read from shared mem
  [=] RAJA_HOST_DEVICE (int tx, int ty, int , int , int, Shmem &aShared,  Shmem &bShared, double & pVal) {

    //Matrix multiply
    for(int j=0; j<TILE_DIM; j++){
      pVal += aShared(ty,j) * bShared(j, tx);
    }

  },

  //write out solution
  [=] RAJA_HOST_DEVICE (int tx, int ty, int , int bx , int by, Shmem &,  Shmem &, double & pVal) {

    int row = by * TILE_DIM + ty;  // Matrix row index
    int col = bx * TILE_DIM + tx;  // Matrix column index
    if(row < N && col < P){
      Cview(row, col) += pVal;
    }

  });

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < P; ++col) {
      ASSERT_FLOAT_EQ(Cview(row,col), C_solView(row,col));
    }
  }


#if defined(RAJA_ENABLE_CUDA)
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C_sol);
#else
  delete [] A;
  delete [] B;
  delete [] C;
  delete [] C_sol;
#endif

}

REGISTER_TYPED_TEST_CASE_P(MatMultiplyScalar, shmem);


using SeqTypes2 =
  ::testing::Types<
  RAJA::list<
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<2,1,0>,

            //Initalize thread private value
           RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                                   RAJA::statement::Lambda<0> > >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::loop_exec,

               //Load matrix into tile
               RAJA::statement::For<1, RAJA::loop_exec,
                 RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::Lambda<1>
                >
               >,
               //Partial multiplication
               RAJA::statement::For<1, RAJA::loop_exec,
                 RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::Lambda<2>
                >
               >
            >, //sliding window

            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                                   RAJA::statement::Lambda<3> > >
         > //Create shared memory
        >//For 3
       >//For 4
      >, //close kernel policy
    //
    //Policy for matrix multiplication using a scalar as accumulator
    //
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<1,0>,
            //Initalize thread private value

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::loop_exec,

               //Load matrix into tile
               RAJA::statement::For<1, RAJA::loop_exec,
                 RAJA::statement::For<0, RAJA::loop_exec,
                                      RAJA::statement::Lambda<1>
                >
               >,
               //Partial multiplication
               RAJA::statement::For<1, RAJA::loop_exec,
                 RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::Lambda<0>, //set pVal = 0
                  RAJA::statement::Lambda<2>, //dot product
                  RAJA::statement::Lambda<3> //write partial product out
                >
               >
            > //sliding window

         > //Create shared memory
        >//For 3
       >//For 4
      > //close kernel policy
    > //close list
  >;//close types

INSTANTIATE_TYPED_TEST_CASE_P(Seq, MatMultiply, SeqTypes2);
INSTANTIATE_TYPED_TEST_CASE_P(Seq, MatMultiplyScalar, SeqTypes2);

#if defined(RAJA_ENABLE_OPENMP)
using OmpTypes2 =
  ::testing::Types<
  RAJA::list<
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<2,1,0>,
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
      >, //close kernel policy
    //Policy for matrix multiply with a scalar as accumulator
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<1,0>,

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
                                       RAJA::statement::Lambda<0>,
                                       RAJA::statement::Lambda<2>,
                                       RAJA::statement::Lambda<3>
                                      >
            > //sliding window

         > //Create shared memory
        >//For 3
       >//For 4
      > //close kernel policy
    > //close list
  >;//close types

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, MatMultiply, OmpTypes2);
INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, MatMultiplyScalar, OmpTypes2);
#endif


#if defined(RAJA_ENABLE_CUDA)
using CudaTypes2 =
  ::testing::Types<
  RAJA::list<
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::For<4, RAJA::cuda_block_y_loop,
        RAJA::statement::For<3, RAJA::cuda_block_x_loop,
          RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<2,1,0>,
            //Initalize thread private value
            RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                                   RAJA::statement::Lambda<0> > >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::seq_exec,

               //Load matrix into tile
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<1>
                                   >
                                 >,
             //perform matrix multiplcation
              RAJA::statement::CudaSyncThreads,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                    RAJA::statement::Lambda<2> > >
                     ,RAJA::statement::CudaSyncThreads,
            >, //sliding window
            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                                   RAJA::statement::Lambda<3> > >
         > //Create shared memory
        >//For 3
       >//For 4
        > //CudaKernel
      >, //close kernel policy
    //Policy for Matrix multiply with a scalar as accumulator
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::For<4, RAJA::cuda_block_y_loop,
        RAJA::statement::For<3, RAJA::cuda_block_x_loop,
          RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<1,0>,

            //Intialize thread private value to zero
            RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                                   RAJA::statement::Lambda<0> > >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::seq_exec,

               //Load matrix into tile
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<1>
                                   >
                                 >,
             //perform matrix multiplcation
              RAJA::statement::CudaSyncThreads,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                    RAJA::statement::Lambda<2> > >
                     ,RAJA::statement::CudaSyncThreads,
           >, //sliding window
            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                                   RAJA::statement::Lambda<3> > >
         > //Create shared memory
        >//For 3
       >//For 4
      > //CudaKernel
     > //close kernel policy
    > //close list
  >;//close types

INSTANTIATE_TYPED_TEST_CASE_P(CUDAShmem, MatMultiply, CudaTypes2);
INSTANTIATE_TYPED_TEST_CASE_P(CUDAShmem, MatMultiplyScalar, CudaTypes2);

using CudaTypes3 =
  ::testing::Types<
  RAJA::list<
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::SizeList<0,0>,
    //Policy for Matrix multiply with a scalar
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::For<4, RAJA::cuda_block_y_loop,
        RAJA::statement::For<3, RAJA::cuda_block_x_loop,
          RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<0,1>,
            RAJA::statement::InitLocalMem<RAJA::cuda_thread_mem, RAJA::ParamList<2>,
            //Initalize thread private value
            RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                                   RAJA::statement::Lambda<0> > >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::seq_exec,

               //Load matrix into tile
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<1>
                                   >
                                 >,
             //perform matrix multiplcation
              RAJA::statement::CudaSyncThreads,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                    RAJA::statement::Lambda<2> > >
                     ,RAJA::statement::CudaSyncThreads,
            >, //sliding window
            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                                   RAJA::statement::Lambda<3> > >
         > //Create private memory
        > //Create shared memory
        >//For 3
       >//For 4
       > //CudaKernel
      > //close kernel policy
    > //close list
  >;//close types

INSTANTIATE_TYPED_TEST_CASE_P(CUDAShmemPriv, MatMultiply, CudaTypes3);
#endif
