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
  Example 3: Dot product
  Placeholder file for example showing how to perform a
  vector dot-product using RAJA sum reduction type.
*/

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  double* a = new double[1000];
  double* b = new double[1000];
  
  for (int i = 0; i < 1000; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }

  delete[] a; 
  delete[] b;
  
  return 0;
}

