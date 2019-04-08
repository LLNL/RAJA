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

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector addition example...\n";

//
// Define vector length
//
  int a_N = 3; 
  int b_N = 5; 

  int *a = memoryManager::allocate<int>(a_N);
  int *b = memoryManager::allocate<int>(b_N);

#if 0
  auto myLambda = [=] (int i) { printf("lambda test \n"); };
  myLambda(2,4);
#endif

  using NEW_POLICY = RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::loop_exec,
        RAJA::statement::tLambda<0, camp::idx_seq<0>, camp::idx_seq<>>
      >
    >;
  
  //Create kernel policy
  using KERNEL_EXEC_POL = 
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::loop_exec,
        RAJA::statement::Lambda<0>
      >
    >;  

  //RAJA::kernel<KERNEL_EXEC_POL>
  RAJA::kernel<NEW_POLICY>
    (RAJA::make_tuple(RAJA::RangeSegment(0,3), //segment tuple...
                      RAJA::RangeSegment(5,8),
                      RAJA::RangeSegment(10,12)
                      ),
     [=](int i) {
      printf("i = %d \n",i);
      assert( 0 && "invoking first lambda \n");
    });
     


//
// Clean up.
//
  memoryManager::deallocate(a);
  memoryManager::deallocate(b);

  std::cout << "\n DONE!...\n";

  return 0;
}
