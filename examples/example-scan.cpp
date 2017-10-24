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

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{


  int N = 6; 
  int *x  = (int*) malloc(N*sizeof(int));
  int *y  = (int*) malloc(N*sizeof(int));
  
  x[0] = 3; x[1] = 1; x[2] = 7;
  x[3] = 0; x[4] = 6; x[5] = 3;
    
  typedef RAJA::seq_exec execute_policy;
  
  printf("Performing exclusive scan \n"); 
  RAJA::exclusive_scan<execute_policy>(x,x+N,y);
  for(int i=0; i<N; ++i){
    printf("%d ",y[i]);
  }
  printf("\n \n");

  printf("Performing inclusive scan \n"); 
  RAJA::inclusive_scan<execute_policy>(x,x+N,y);
  for(int i=0; i<N; ++i){
    printf("%d ",y[i]);
  }
  printf("\n");

  free(x);
  free(y);

  return 0;
}
