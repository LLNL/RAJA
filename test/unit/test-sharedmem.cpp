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

#include "camp/camp.hpp"
#include "camp/concepts.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

using namespace RAJA;
using namespace RAJA::statement;

//Define Tile size ( TILE_DIM x TILE_DIM )
const int TILE_DIM = 16;

//
//Matrix tranpose example
//
/*
template <typename NestedPolicy>
class Kernel : public ::testing::Test
{

  virtual void SetUp() {}
  virtual void TearDown() {}

};
TYPED_TEST_CASE_P(Kernel);

CUDA_TYPED_TEST_P(Kernel, Basic)
{

  using Tile_pol = at_v<TypeParam, 0>;
  using Pol = at_v<TypeParam, 1>;

  const int DIM = 2;
  const int N_rows = 144;
  const int N_cols = 255;
  const int TILE_DIM = 16;

  const int inner_Dim0 = TILE_DIM;
  const int inner_Dim1 = TILE_DIM;

  const int outer_Dim0 = (N_cols-1)/TILE_DIM+1;
  const int outer_Dim1 = (N_rows-1)/TILE_DIM+1;

  int *A, *At, *B, *Bt;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged(&A,  sizeof(int) * N_rows * N_cols);
  cudaMallocManaged(&At, sizeof(int) * N_rows * N_cols);
  cudaMallocManaged(&B,  sizeof(int) * N_rows * N_cols);
  cudaMallocManaged(&Bt, sizeof(int) * N_rows * N_cols);
#else
  A  = new int[N_rows * N_cols];
  At = new int[N_rows * N_cols];
  B  = new int[N_rows * N_cols];
  Bt = new int[N_rows * N_cols];
#endif

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
  ///using memObj = RAJA::MemObj<int, RAJA::SizeList<TILE_DIM, TILE_DIM>>;

  //using mySharedMemory = RAJA::MemWrapper<Tile_pol, memObj>;
  //mySharedMemory myTile;
  //mySharedMemory myTile2;

#if 0
  RAJA::kernel_param<Pol>(iSpace,
                          RAJA::make_tuple(myTile, myTile2),

  [=] RAJA_HOST_DEVICE (int tx, int ty, int bx, int by, mySharedMemory &myTile,  mySharedMemory &myTile2) {

           int col = bx * TILE_DIM + tx;  // Matrix column index
           int row = by * TILE_DIM + ty;  // Matrix row index
           if(row < N_rows && col < N_cols){
             myTile(ty,tx)  = Aview(row, col);
             myTile2(ty,tx) = Bview(row, col);
           }
        },

      //read from shared mem
       [=] RAJA_HOST_DEVICE (int tx, int ty, int bx, int by, mySharedMemory &myTile, mySharedMemory &myTile2) {

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
#endif
}

REGISTER_TYPED_TEST_CASE_P(Kernel, Basic);
*/

/*
using SeqTypes =
  ::testing::Types<
  RAJA::list<RAJA::cpu_tile_mem,
    RAJA::KernelPolicy<
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::For<2, RAJA::loop_exec,
                               //RAJA::statement::CreateShmem<camp::idx_seq<1,0>,
            RAJA::statement::CreateShmem<camp::idx_seq<0>,
              RAJA::statement::For<1, RAJA::loop_exec,
                RAJA::statement::For<0, RAJA::loop_exec,
                  RAJA::statement::Lambda<0>
                                   >
                                 >,
                RAJA::statement::For<1, RAJA::loop_exec,
                  RAJA::statement::For<0, RAJA::loop_exec,
                    RAJA::statement::Lambda<1> > >
              > //close shared memory scope
            >//for 2
        >//for 3
      > //kernel policy
    > //list 
  >; //types
INSTANTIATE_TYPED_TEST_CASE_P(Seq, Kernel, SeqTypes);
*/

/*
#if defined(RAJA_ENABLE_OPENMP)
using TestTypes =
  ::testing::Types<
  RAJA::list<RAJA::cpu_tile_mem,
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
       > //close policy
     > //close list

  ,RAJA::list<RAJA::cpu_tile_mem,
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
     >//close policy
    > //close list
  ,RAJA::list<RAJA::cpu_tile_mem,
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
      > //close policy list
     >
   >;


INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, Kernel, TestTypes);
#endif

#if defined(RAJA_ENABLE_CUDA)

using CUDATypes =
  ::testing::Types<
  RAJA::list<RAJA::cuda_shared_mem,
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
      > //kernel policy
    > //list 
  >; //types
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, Kernel, CUDATypes);
#endif
*/

template <typename NestedPolicy>
class MatMultiply : public ::testing::Test
{
  virtual void SetUp(){}  
  virtual void TearDown(){}
};

TYPED_TEST_CASE_P(MatMultiply);

CUDA_TYPED_TEST_P(MatMultiply, shmem)
{

  using Tile_pol0 = at_v<TypeParam, 0>;
  using Tile_size0 = at_v<TypeParam, 1>;

  using Tile_pol1 = at_v<TypeParam, 2>;
  using Tile_size1 = at_v<TypeParam, 3>;
  using Pol = at_v<TypeParam, 4>;

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

  int *A, *B, *C, *C_sol;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged(&A,  sizeof(int) * N * M);
  cudaMallocManaged(&B,  sizeof(int) * M * P);
  cudaMallocManaged(&C,  sizeof(int) * N * P);
  cudaMallocManaged(&C_sol,  sizeof(int) * N * P);
#else
  A  = new int[N * M];
  B  = new int[M * P];
  C  = new int[N * P];
  C_sol  = new int[N * P];
#endif

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

  using memObj0 = RAJA::MemObj<double, Tile_size0>;
  using memObj1 = RAJA::MemObj<double, Tile_size1>;
  //using memObj = RAJA::MemObj<double, RAJA::SizeList<TILE_DIM, TILE_DIM>>;
  //using memObj = RAJA::MemObj<double, RAJA::SizeList<TILE_DIM,TILE_DIM>>;
  
  using Shmem      = RAJA::MemWrapper<Tile_pol0, memObj0>;
  using threadPriv = RAJA::MemWrapper<Tile_pol1, memObj1>;
  
  Shmem aShared, bShared; //memory to be shared between threads
  threadPriv pVal; //thread private value
 
  RAJA::kernel_param<Pol>(iSpace,
                          RAJA::make_tuple(aShared, bShared, pVal),

  [=] RAJA_HOST_DEVICE (int tx, int ty, int , int , int , Shmem &,  Shmem &, threadPriv &pVal) {

   //I would like this to behave like a thread private variable
   pVal(ty,tx) = 0.0;

  },

  [=] RAJA_HOST_DEVICE (int tx, int ty, int i, int bx, int by, Shmem &aShared,  Shmem &bShared, threadPriv &) {

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
  [=] RAJA_HOST_DEVICE (int tx, int ty, int , int , int , Shmem &aShared,  Shmem &bShared, threadPriv & pVal) {

    //Matrix multiply
    for(int j=0; j<TILE_DIM; j++){
      pVal(ty,tx) += aShared(ty,j) * bShared(j, tx);
    }

  },

 //If in range write out
 [=] RAJA_HOST_DEVICE (int tx, int ty, int , int bx, int by, Shmem &, Shmem &, threadPriv &pValue) {

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


template <typename NestedPolicy>
class MatMultiplyScalar : public ::testing::Test
{
  virtual void SetUp(){}  
  virtual void TearDown(){}
};

TYPED_TEST_CASE_P(MatMultiplyScalar);

CUDA_TYPED_TEST_P(MatMultiplyScalar, shmem)
{

  using Tile_pol0 = at_v<TypeParam, 0>;
  using Tile_size0 = at_v<TypeParam, 1>;

  using Pol = at_v<TypeParam, 5>; 

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

  int *A, *B, *C, *C_sol;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged(&A,  sizeof(int) * N * M);
  cudaMallocManaged(&B,  sizeof(int) * M * P);
  cudaMallocManaged(&C,  sizeof(int) * N * P);
  cudaMallocManaged(&C_sol,  sizeof(int) * N * P);
#else
  A  = new int[N * M];
  B  = new int[M * P];
  C  = new int[N * P];
  C_sol  = new int[N * P];
#endif

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
      Cview(r,c) = 0.0;
      C_solView(r,c) = dot;
    }
  }

  auto iSpace =
    RAJA::make_tuple(RAJA::RangeSegment(0, inner_Dim0), RAJA::RangeSegment(0,inner_Dim1),
                     RAJA::RangeSegment(0, windowIter),
                     RAJA::RangeSegment(0, outer_Dim0), RAJA::RangeSegment(0,outer_Dim1));

  using memObj0 = RAJA::MemObj<double, Tile_size0>;
  using Shmem      = RAJA::MemWrapper<Tile_pol0, memObj0>;
  
  Shmem aShared, bShared; //memory to be shared between threads

  RAJA::kernel_param<Pol>(iSpace,
                          RAJA::make_tuple(aShared, bShared, 0.0),

  //Set pVal to zero
  [=] RAJA_HOST_DEVICE (int tx, int ty, int i, int bx, int by, Shmem &aShared,  Shmem &bShared, double & pVal) {

  pVal = 0.0;
  },

  //Loop across window
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
  [=] RAJA_HOST_DEVICE (int tx, int ty, int , int bx , int by, Shmem &aShared,  Shmem &bShared, double & pVal) {

    //Matrix multiply
    for(int j=0; j<TILE_DIM; j++){
      pVal += aShared(ty,j) * bShared(j, tx);
    }

  },

  //write out solution
  [=] RAJA_HOST_DEVICE (int tx, int ty, int , int bx , int by, Shmem &aShared,  Shmem &bShared, double & pVal) {

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
    RAJA::cpu_tile_mem, RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::cpu_tile_mem, RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
         RAJA::statement::CreateShmem<camp::idx_seq<2,1,0>,
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
    //Policy for matrix multiplication using a scalar
    //
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::CreateShmem<camp::idx_seq<1,0>,
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
                  RAJA::statement::Lambda<3>, //write partial product out
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
    RAJA::cpu_tile_mem, RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::cpu_tile_mem, RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::CreateShmem<camp::idx_seq<2,1,0>,
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
    //Policy for matrix multiply with a scalar
    RAJA::KernelPolicy<
      RAJA::statement::For<4, RAJA::loop_exec,
        RAJA::statement::For<3, RAJA::loop_exec,
          RAJA::statement::CreateShmem<camp::idx_seq<1,0>,

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
    RAJA::cuda_shared_mem, RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::cuda_shared_mem, RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::For<4, RAJA::cuda_block_exec,
        RAJA::statement::For<3, RAJA::cuda_block_exec,
          RAJA::statement::CreateShmem<camp::idx_seq<2,1,0>,
            //Initalize thread private value
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::For<0, RAJA::cuda_thread_exec,
                                   RAJA::statement::Lambda<0> > >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::seq_exec,

               //Load matrix into tile
              RAJA::statement::For<1, RAJA::cuda_thread_exec,
                RAJA::statement::For<0, RAJA::cuda_thread_exec,
                  RAJA::statement::Lambda<1>
                                   >
                                 >,
             //perform matrix multiplcation
              RAJA::statement::CudaSyncThreads,
                RAJA::statement::For<1, RAJA::cuda_thread_exec,
                  RAJA::statement::For<0, RAJA::cuda_thread_exec,
                    RAJA::statement::Lambda<2> > >
                     ,RAJA::statement::CudaSyncThreads,
            >, //sliding window
            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::For<0, RAJA::cuda_thread_exec,
                                   RAJA::statement::Lambda<3> > >
         > //Create shared memory
        >//For 3
       >//For 4
        > //CudaKernel
      >, //close kernel policy
    //Policy for Matrix multiply with a scalar
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::For<4, RAJA::cuda_block_exec,
        RAJA::statement::For<3, RAJA::cuda_block_exec,
          RAJA::statement::CreateShmem<camp::idx_seq<1,0>,

            //Intialize thread private value to zero
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::For<0, RAJA::cuda_thread_exec,
                                   RAJA::statement::Lambda<0> > >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::seq_exec,

               //Load matrix into tile
              RAJA::statement::For<1, RAJA::cuda_thread_exec,
                RAJA::statement::For<0, RAJA::cuda_thread_exec,
                  RAJA::statement::Lambda<1>
                                   >
                                 >,
             //perform matrix multiplcation
              RAJA::statement::CudaSyncThreads,
                RAJA::statement::For<1, RAJA::cuda_thread_exec,
                  RAJA::statement::For<0, RAJA::cuda_thread_exec,
                    RAJA::statement::Lambda<2> > >
                     ,RAJA::statement::CudaSyncThreads,
           >, //sliding window
            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::For<0, RAJA::cuda_thread_exec,
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
    RAJA::cuda_shared_mem, RAJA::SizeList<TILE_DIM, TILE_DIM>,
    RAJA::cuda_priv_mem, RAJA::SizeList<1>,
    //Policy for Matrix multiply with a scalar
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::For<4, RAJA::cuda_block_exec,
        RAJA::statement::For<3, RAJA::cuda_block_exec,
          RAJA::statement::CreateShmem<camp::idx_seq<2,1,0>,
            //Initalize thread private value
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::For<0, RAJA::cuda_thread_exec,
                                   RAJA::statement::Lambda<0> > >,

            //Slide window across matrix
             RAJA::statement::For<2, RAJA::seq_exec,

               //Load matrix into tile
              RAJA::statement::For<1, RAJA::cuda_thread_exec,
                RAJA::statement::For<0, RAJA::cuda_thread_exec,
                  RAJA::statement::Lambda<1>
                                   >
                                 >,
             //perform matrix multiplcation
              RAJA::statement::CudaSyncThreads,
                RAJA::statement::For<1, RAJA::cuda_thread_exec,
                  RAJA::statement::For<0, RAJA::cuda_thread_exec,
                    RAJA::statement::Lambda<2> > >
                     ,RAJA::statement::CudaSyncThreads,
            >, //sliding window
            //Write memory out to global matrix
            RAJA::statement::For<1, RAJA::cuda_thread_exec,
              RAJA::statement::For<0, RAJA::cuda_thread_exec,
                                   RAJA::statement::Lambda<3> > >
         > //Create shared memory
        >//For 3
       >//For 4
       > //CudaKernel
      > //close kernel policy
    > //close list
  >;//close types

INSTANTIATE_TYPED_TEST_CASE_P(CUDAShmemPriv, MatMultiply, CudaTypes3);

#endif


//
//Example with RAJA shared memory objects
//
/*
#if defined(RAJA_ENABLE_OPENMP)
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


  using cpu_shmem_t = RAJA::ShmemTile<RAJA::cpu_shmem,
                                      int,
                                      RAJA::ArgList<0, 1>,
                                      RAJA::SizeList<TILE_DIM, TILE_DIM>,
                                      decltype(iSpace)>;
  using RAJAMemory = RAJA::SharedMemWrapper<cpu_shmem_t>;
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
*/
