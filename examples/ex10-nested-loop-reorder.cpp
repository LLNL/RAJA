//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

//
//Define in function scope for the RAJA CUDA variant
//

RAJA_INDEX_VALUE(ID, "ID");
RAJA_INDEX_VALUE(IZ, "IZ"); 

//
//Loop Reordering example
//
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{


  using myPol = RAJA::nested::Policy<   
  RAJA::nested::TypedFor<0, RAJA::loop_exec, ID>,
  RAJA::nested::TypedFor<1, RAJA::loop_exec,IZ>   
    >;

  RAJA::RangeSegment Range0(0,4);
  RAJA::RangeSegment Range1(0,1);

  printf("Loop Format #1 \n ");
  RAJA::nested::forall(myPol{},
                       RAJA::make_tuple(Range0, Range1),                       
                       [=] (ID i0, IZ i1) {
                         printf("%ld, %ld \n", (long int)*i0, (long int)*i1);
                     });


  printf("\n Loop Format #2 \n ");
  using myPol2 = RAJA::nested::Policy<   
  RAJA::nested::TypedFor<1, RAJA::loop_exec,IZ>,
    RAJA::nested::TypedFor<0, RAJA::loop_exec, ID>
    >;

  RAJA::nested::forall(myPol2{},
                       RAJA::make_tuple(Range0, Range1),                       
                       [=] (ID i0, IZ i1) {
                         printf("%ld, %ld \n", (long int)*i0, (long int)*i1);
                       });

  return 0;
}

