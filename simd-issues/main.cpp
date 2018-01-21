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
#include <chrono>
#include <ctime>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

#include "vec_fun.hpp"

using arr_type = double; 

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  
  //-----Initialize variables
  int nIter = 100;
  const RAJA::Index_type arrLen = 8000;  
  arr_type *A = new arr_type[arrLen];   
  RAJA::View<arr_type, RAJA::Layout<1> > Aview(A,arrLen);

  
  //-----
  auto start = std::chrono::system_clock::now();

  for(int i=0; i<nIter; ++i){
  setArray<RAJA::View<arr_type, RAJA::Layout<1> >, double, RAJA::seq_exec >(Aview, 0, arrLen);
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "SEQ elapsed time: " << elapsed_seconds.count() << "s\n";
  //---------------------------------------------------------------


  //-----
  start = std::chrono::system_clock::now();

  for(int i=0; i<nIter; ++i){
  setArray<RAJA::View<arr_type, RAJA::Layout<1> >, double, RAJA::loop_exec >(Aview, 0, arrLen);
  }

  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  std::cout << "LOOP elapsed time: " << elapsed_seconds.count() << "s\n";
  //---------------------------------------------------------------


  //-----
  start = std::chrono::system_clock::now();

  for(int i=0; i<nIter; ++i){
  setArray<RAJA::View<arr_type, RAJA::Layout<1> >, double, RAJA::simd_exec >(Aview, 0, arrLen);
  }

  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  std::cout << "SIMD elapsed time: " << elapsed_seconds.count() << "s\n";
  //---------------------------------------------------------------



  delete[] A;

  return 0;
}
