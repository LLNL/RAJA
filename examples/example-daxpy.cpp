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
  Example 0: DAXPY
  This Double-Precision-A*X Plus Y example code 
  illustrates the similarities between a C++ style 
  for loop and a RAJA forall loop. 

  Details about this code may be found in the RAJA documentation
  http://raja.readthedocs.io/en/feature-user-docs/getting_started.html
*/

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  double* a = new double[1000];
  double* b = new double[1000];
  
  double c = 3.14159;
  
  for (int i = 0; i < 1000; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }

  RAJA::forall<RAJA::seq_exec>(0, 1000, [=] (int i) {
      a[i] += b[i] * c;
    });

  delete[] a; 
  delete[] b;
  
  return 0;
}

