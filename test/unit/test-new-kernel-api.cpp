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

  RAJA::kernel_param<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, N_cols),
                                           RAJA::RangeSegment(0, N_rows)),
                          RAJA::make_tuple(myTile, myTile2),

  //Load data into shared memory
  [=] RAJA_HOST_DEVICE (int col, int row, int tx, int ty, SharedTile &myTile, SharedTile &myTile2) {

      myTile(ty,tx)  = Aview(row, col);
      myTile2(ty,tx) = Bview(row, col);

  },

  //read from shared mem
  [=] RAJA_HOST_DEVICE (int col, int row, int tx, int ty, SharedTile &myTile, SharedTile &myTile2) {

     Atview(col, row) = myTile(ty,tx);
     Btview(col, row) = myTile2(ty,tx);
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
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::InitList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::loop_exec,
                RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::Lambda<0, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1>>
                                   >
                                 >,

                //Read data from shared memory
                RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::For<1, RAJA::loop_exec,
                  RAJA::statement::Lambda<1, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1>>
                     >
                  >

              > //close shared memory scope
            >//for 2
        >//for 3
      > //kernel policy
    > //list
  >; //types
INSTANTIATE_TYPED_TEST_CASE_P(Seq, MatTranspose, SeqTypes);


#if defined(RAJA_ENABLE_OPENMP)
using TestTypes =
  ::testing::Types<
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
       RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::InitList<0,1>,

           //Load data into shared memory
           RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<0, 1>,
                                     RAJA::statement::Lambda<0, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1>>
                                     >,

           //Read data from shared memory
           RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<0, 1>,
                                     RAJA::statement::Lambda<1, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1>>
                                     >
                                 >
        >//for 2
       >//for 3
       > //close policy
    >, //close list
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::InitList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
                RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::Lambda<0, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1>>
                                   >
                                 >,

                //Read data from shared memory
                RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
                  RAJA::statement::Lambda<1, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1>>
                     >
                  >

              > //close shared memory scope
            >//for 2
        >//for 3
      > //kernel policy
    > //close list
  ,RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::omp_parallel_for_exec,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::loop_exec,

          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::InitList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::loop_exec,
                RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::Lambda<0, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1>>
                                   >
                                 >,

                //Read data from shared memory
                RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::For<1, RAJA::loop_exec,
                  RAJA::statement::Lambda<1, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1>>
                     >
                  >

              > //close shared memory scope
            >//for 2
        >//for 3
      > //kernel policy
     > //close list
   >;


INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, MatTranspose, TestTypes);
#endif

#if defined(RAJA_ENABLE_CUDA)
using CUDATypes =
  ::testing::Types<
  RAJA::list<
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::Tile<1, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::cuda_block_y_loop,
        RAJA::statement::Tile<0, RAJA::statement::tile_fixed<TILE_DIM>, RAJA::cuda_block_x_loop,

            RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::InitList<0,1>,

              //Load data into shared memory
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<0, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1> >
                 >
               >,
              RAJA::statement::CudaSyncThreads,

                //Read data from shared memory
                RAJA::statement::For<0, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<1, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<1, Segs<0>, Segs<1>, OffSets<0>, OffSets<1>, Params<0,1> >
                  >
                 >,
                RAJA::statement::CudaSyncThreads
              > //close shared memory scope
            >//for 2
          >//for 3
        > //CudaKernel
      > //kernel policy
    > //list
  >; //types
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, MatTranspose, CUDATypes);
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

  [=] RAJA_HOST_DEVICE (int tx, int ty, ThreadPriv &pVal) {

   pVal(ty,tx) = 0.0;

  },

  [=] RAJA_HOST_DEVICE (int tx, int ty, int i, int bx, int by, Shmem &aShared,  Shmem &bShared) {

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
  [=] RAJA_HOST_DEVICE (int tx, int ty, Shmem &aShared,  Shmem &bShared, ThreadPriv & pVal) {

    //Matrix multiply
    for(int j=0; j<TILE_DIM; j++){
      pVal(ty,tx) += aShared(ty,j) * bShared(j, tx);
    }

  },

 //If in range write out
 [=] RAJA_HOST_DEVICE (int tx, int ty, int bx, int by, ThreadPriv &pValue) {

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


using SeqTypes2 =
  ::testing::Types<
  RAJA::list<
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::InitList<2,1,0>,

            //Initalize thread private value
           RAJA::statement::For<1, RAJA::loop_exec,
             RAJA::statement::For<0, RAJA::loop_exec,
               RAJA::statement::Lambda<0, Segs<0,1>, Params<2> >
              >
            >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::loop_exec,

               //Load matrix into tile
               RAJA::statement::For<1, RAJA::loop_exec,
                 RAJA::statement::For<0, RAJA::loop_exec,
                   RAJA::statement::Lambda<1, Segs<0,1,2,3,4>, Params<0,1> >
                >
               >,
               //Partial multiplication
               RAJA::statement::For<1, RAJA::loop_exec,
                 RAJA::statement::For<0, RAJA::loop_exec,
                   RAJA::statement::Lambda<2, Segs<0,1>, Params<0,1,2> >
                >
               >
            >, //sliding window

            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<3, Segs<0,1, 3, 4>, Params<2> >
              >
            >
         > //Create shared memory
        >//For 3
       >//For 4
      > //close kernel policy
    > //close list
  >;//close types

INSTANTIATE_TYPED_TEST_CASE_P(Seq, MatMultiply, SeqTypes2);


#if defined(RAJA_ENABLE_OPENMP)
using OmpTypes2 =
  ::testing::Types<
  RAJA::list<
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::InitList<2,1,0>,
            //Initalize thread private value
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                                   RAJA::statement::Lambda<0, Segs<0,1>, Params<2> >
              >
            >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::loop_exec,

               //Load matrix into tile
               RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                     RAJA::ArgList<0, 1>,
                   RAJA::statement::Lambda<1, Segs<0,1,2,3,4>, Params<0,1> >
                                     >,

             //perform matrix multiplcation
             RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                      RAJA::ArgList<0, 1>,
                   RAJA::statement::Lambda<2, Segs<0,1>, Params<0,1,2> >
                                      >
            >, //sliding window

            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<0, RAJA::loop_exec,
                RAJA::statement::Lambda<3, Segs<0,1, 3, 4>, Params<2> >
              >
             >
         > //Create shared memory
        >//For 3
       >//For 4
      > //close kernel policy
    > //close list
  >;//close types

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, MatMultiply, OmpTypes2);
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
          RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::InitList<2,1,0>,
            //Initalize thread private value
            RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<0, Segs<0,1>, Params<2> >
              >
             >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::seq_exec,

              //Load matrix into tile
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                   RAJA::statement::Lambda<1, Segs<0,1,2,3,4>, Params<0,1> >
                >
              >,
              //perform matrix multiplcation
              RAJA::statement::CudaSyncThreads,
                RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                  RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                   RAJA::statement::Lambda<2, Segs<0,1>, Params<0,1,2> >
                >
              >,
              RAJA::statement::CudaSyncThreads
            >, //sliding window

            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
              RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                RAJA::statement::Lambda<3, Segs<0,1, 3, 4>, Params<2> >
              >
             >
         > //Create shared memory
        >//For 3
       >//For 4
        > //CudaKernel
      > //close kernel policy
    > //close list
  >;//close types

INSTANTIATE_TYPED_TEST_CASE_P(CUDAShmem, MatMultiply, CudaTypes2);
#endif
