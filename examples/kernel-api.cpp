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

RAJA_INDEX_VALUE(IIDX, "IIDX");
RAJA_INDEX_VALUE(JIDX, "JIDX");

using RAJA::statement::Seg;
using RAJA::statement::SegList;
using RAJA::statement::Param;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

#if 0
  std::cout<<"\n ----------------------------------------------------"<<std::endl;
  std::cout<<"Testing existing kernel lambda statements"<<std::endl;
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


  std::cout<<"\n \n----------------------------------------------------"<<std::endl;
  std::cout<<"New Kernel API"<<std::endl;

  //Lambda statement format : lambda idx, segment indices, parameter indices (to be tested...)
  using NEW_POLICY =
    RAJA::KernelPolicy<
      RAJA::statement::For<0,RAJA::loop_exec,
        RAJA::statement::Lambda<0, Seg<0>>
      >,
      RAJA::statement::For<1,RAJA::loop_exec,
        RAJA::statement::Lambda<1, Seg<1>>
      >,
      RAJA::statement::For<1, RAJA::loop_exec,
        RAJA::statement::For<2,RAJA::loop_exec,
         RAJA::statement::Lambda<2, Seg<1>, Seg<2>>
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



  //-------------------------------------------------------------------------------
  std::cout<<"\n \n----------------------------------------------------"<<std::endl;
  std::cout<<"New Kernel API"<<std::endl;

  using POLICY_V2 =
    RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::loop_exec,
                         RAJA::statement::Lambda<0, Param<0>, Seg<0>>
                         >,
    RAJA::statement::For<1,RAJA::loop_exec,
                         RAJA::statement::Lambda<1, Seg<1>>
    >
    >;

  RAJA::TypedRangeSegment<IIDX> IRange(0, 5);
  RAJA::TypedRangeSegment<JIDX> JRange(7, 10);

  RAJA::kernel_param<POLICY_V2>
    (RAJA::make_tuple(IRange, JRange),
     RAJA::make_tuple((double)55.2),
     [=](double &dot, IIDX i) {
      printf("invoke kernel 1 :  %f , iter = %d \n", dot, (int)(*i));
    },
     [=](JIDX j) {
      printf("invoke kernel 2 : iter = %d \n", (int)(*j));
    });

#endif

  //-------------------------------------------------------------------------------
  std::cout<<"\n \n----------------------------------------------------"<<std::endl;
  std::cout<<"New Kernel API with SegList"<<std::endl;

  using POLICY_V3 =
    RAJA::KernelPolicy<
    RAJA::statement::For<0, RAJA::loop_exec,
                           //RAJA::statement::Lambda<0, Seg<0>>
                           RAJA::statement::Lambda<0, SegList<0>>
      >
    >;

  RAJA::RangeSegment iSegment(0, 1);
  RAJA::kernel<POLICY_V3>
    (RAJA::make_tuple(iSegment), [=] (int i) {

      printf("i = %d \n", i);
    });





  return 0;
}
