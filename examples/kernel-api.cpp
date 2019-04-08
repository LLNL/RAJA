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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>

#include "memoryManager.hpp"
#include "RAJA/RAJA.hpp"

const int DIM = 2;

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Testing existing kernel lambda statements \n"<<std::endl;
  // Create kernel policy
    using KERNEL_POLICY =
    RAJA::KernelPolicy<
      RAJA::statement::For<0,RAJA::loop_exec,
        RAJA::statement::Lambda<0>
      >,
      RAJA::statement::For<1,RAJA::loop_exec,
        RAJA::statement::Lambda<1>
      >,
     RAJA::statement::For<1, RAJA::loop_exec,
       RAJA::statement::For<2,RAJA::loop_exec,
         RAJA::statement::Lambda<2>
       >
      >
    >;

  //Existing kernel API
  RAJA::kernel<KERNEL_POLICY>(
    RAJA::make_tuple(RAJA::RangeSegment(0, 3),  // segment tuple...
                     RAJA::RangeSegment(5, 8),
                     RAJA::RangeSegment(20, 23)),
    [=](int i, int , int ) {
      printf("i = %d \n", i);
    },

    [=](int, int j, int) {
      printf("j = %d \n", j);
    },
    [=](int, int j, int k) {
      printf("j, k = %d  %d \n",j, k);
    });


  std::cout<<"----------------------------------------------------\n \n"<<std::endl;
  std::cout<<"Testing new kernel lambda statement types \n"<<std::endl;

  //Lambda statement format : lambda idx, segment indices, parameter indices (to be tested...)
  using NEW_POLICY =
    RAJA::KernelPolicy<
      RAJA::statement::For<0,RAJA::loop_exec,
        RAJA::statement::tLambda<0, camp::idx_seq<0>, camp::idx_seq<>>
      >,
      RAJA::statement::For<1,RAJA::loop_exec,
        RAJA::statement::tLambda<1, camp::idx_seq<1>, camp::idx_seq<>>
      >,
     RAJA::statement::For<1, RAJA::loop_exec,
       RAJA::statement::For<2,RAJA::loop_exec,
         RAJA::statement::tLambda<2, camp::idx_seq<1,2>, camp::idx_seq<>>
       >
      >
    >;
  // New Kernel API ...
  RAJA::kernel<NEW_POLICY>(
    RAJA::make_tuple(RAJA::RangeSegment(0, 3),  // segment tuple...
                     RAJA::RangeSegment(5, 8),
                     RAJA::RangeSegment(20, 23)),
    [=](int i) {
      printf("i = %d \n", i);
    },

    [=](int j) {
      printf("j = %d \n", j);
    },

    [=](int j, int k) {
      printf("j, k = %d  %d \n",j, k);
    });



  //-----------------------------------------------
  printf("\n Testing matrix multiplication kernel with new Lambda API ...\n");
  int N = 10;
  double *A = memoryManager::allocate<double>(N * N);
  double *B = memoryManager::allocate<double>(N * N);
  double *C = memoryManager::allocate<double>(N * N);

  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, N);

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      Aview(row, col) = row;
      Bview(row, col) = col;
    }
  }

  using EXEC_POL6 =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::loop_exec,
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::tLambda<0,camp::idx_seq<>,camp::idx_seq<0>>,  // dot = 0.0
          RAJA::statement::For<2, RAJA::loop_exec,
            RAJA::statement::tLambda<1, camp::idx_seq<0,1,2>, camp::idx_seq<0>> // inner loop: dot += ...
          >,
          RAJA::statement::tLambda<2, camp::idx_seq<0,1>, camp::idx_seq<0>>   // set C(row, col) = dot
        >
      >
    >;

  RAJA::kernel_param<EXEC_POL6>(
    RAJA::make_tuple(RAJA::RangeSegment(0, N),
                     RAJA::RangeSegment(0, N),
                     RAJA::RangeSegment(0, N)),
    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },

    // lambda 2
    [=] (int col, int row, double& dot) {
       Cview(row, col) = dot;
    }

  );

  checkResult<double>(Cview, N);

  return 0;
}


template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( std::abs( Cview(row, col) - row * col * N ) > 10e-12 ) {
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
