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
using RAJA::statement::Param;
using RAJA::statement::OffSet;

using RAJA::statement::SegList;
using RAJA::statement::ParamList;


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  using POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0,RAJA::loop_exec,
        RAJA::statement::Lambda<0, Seg<0>>
      >,
      RAJA::statement::For<1,RAJA::loop_exec,
        RAJA::statement::Lambda<1, Seg<1>>
      >,
      RAJA::statement::For<1, RAJA::loop_exec,
        RAJA::statement::For<2,RAJA::loop_exec,
          RAJA::statement::Lambda<2, SegList<1,2>>
      >
     >
    >;

  //kernel policy 1
  RAJA::kernel<POL1>(
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
  using POL2 =
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

  RAJA::kernel_param<POL2>
    (RAJA::make_tuple(IRange, JRange),
     RAJA::make_tuple((double)55.2),
     [=](double &dot, IIDX i) {
      printf("invoke kernel 1 :  %f , iter = %d \n", dot, (int)(*i));
    },
     [=](JIDX j) {
       printf("invoke kernel 2 : iter = %d \n", (int)(*j));
     });


  //-------------------------------------------------------------------------------
  std::cout<<"\n \n----------------------------------------------------"<<std::endl;
#if defined(RAJA_ENABLE_CUDA)
  using POL3 =
    RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::Tile<0, RAJA::statement::tile_fixed<1>, RAJA::cuda_block_x_loop,
        RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                             RAJA::statement::Lambda<0, Seg<0>, OffSet<0> >
                             >
                            >
      >//cuda
    >; //kernel

  RAJA::RangeSegment rangeSeg(0, 1);

  RAJA::kernel<POL3>
    (RAJA::make_tuple(rangeSeg),
    [=] RAJA_HOST_DEVICE (int i, int j) {
     printf("Segment/offset %d  %d \n", i, j);
  });
#endif

  return 0;
}
