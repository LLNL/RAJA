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

/*
  Example 9: Vertex-sum with Index Coloring.

  ----[Details]---------------------

  -----[RAJA Concepts]---------------

  ----[Kernel Variants and RAJA Features]------------

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

/*
  Function to verify correctness
*/
void checkSolution();

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  printf("Example 9: ...\n \n");

  const int N = 1000;

  return 0;
}
