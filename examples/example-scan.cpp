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


  int N = 8; 
  int *input  = (int*) malloc(N*sizeof(int));
  int *output = (int*) malloc(N*sizeof(int));

  input[0] = 3; input[1] = 1; input[2] = 7;
  input[3] = 0; input[4] = 4; input[5] = 1;
  input[6] = 6; input[7] = 3;
  
  typedef RAJA::seq_exec execute_policy;
  
  printf("Performing exclusive scan \n"); 
  RAJA::exclusive_scan<execute_policy>(input,input+N,output);
  for(int i=0; i<N; ++i){
    printf("%d ",output[i]);
  }
  printf("\n \n");

  printf("Performing inclusive scan \n"); 
  RAJA::inclusive_scan<execute_policy>(input,input+N,output);
  for(int i=0; i<N; ++i){
    printf("%d ",output[i]);
  }
  printf("\n");

  free(input);
  free(output);

  return 0;
}
