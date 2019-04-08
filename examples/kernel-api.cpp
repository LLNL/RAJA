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

#include "RAJA/RAJA.hpp"

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Testing existing kernel lambda statement types \n"<<std::endl;
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

  return 0;
}
