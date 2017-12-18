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

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

/*
  Example 4: Reductions
  This example code shows how to use multiple RAJA
  reduction types to perform different reductions
  in a RAJA forall loop.
*/
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{


  int N  = 10;
  int *x = new int[N];

  x[0] = 1000; x[1] = 2; x[2] = 3; x[3] = 4; x[4] = 5;
  x[5] = 6; x[6] = 7; x[7] = 8; x[8] = 9; x[9] = 1;
  
  typedef RAJA::seq_reduce reduce_policy;
  typedef RAJA::seq_exec execute_policy;
  
  RAJA::ReduceSum<reduce_policy, int> sum(0);

  RAJA::ReduceMin<reduce_policy, int> min(0);
  RAJA::ReduceMinLoc<reduce_policy, int> minLocation(0);
  
  RAJA::ReduceMax<reduce_policy, int> max(0);
  RAJA::ReduceMaxLoc<reduce_policy, int> maxLocation(-1,-1);

  RAJA::forall<execute_policy>(RAJA::RangeSegment(0,N), [=] (RAJA::Index_type i){
      
      sum += x[i];

      min.min(x[i]);
      minLocation.minloc(x[i],i);
      minLocation.minloc(i,x[i]);

      max.max(x[i]);
      maxLocation.maxloc(x[i],i);
    });

  printf("sum = %d \n", sum.get());
  printf("min = %d, min location = %d \n",min.get(), int(minLocation.getLoc()));
  printf("max = %d, max location = %d \n",max.get(), int(maxLocation.getLoc()));


  delete [] x; 

  return 0;
}
