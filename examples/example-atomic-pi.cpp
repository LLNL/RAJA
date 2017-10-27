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
  typedef RAJA::seq_reduce reduce_policy;
  typedef RAJA::seq_exec execute_policy;
  typedef RAJA::atomic::seq_atomic atomic_policy; 

  RAJA::Index_type begin = 0;
  RAJA::Index_type numBins = 512 * 512;
  
  double *atomicPiSum = new double[1]; *atomicPiSum = 0; 
  RAJA::ReduceSum<reduce_policy, double> piSum(0.0);

  //Computing PI using reduction
  RAJA::forall<execute_policy>(begin,
                               numBins,
                               [=](int i) {
                                 double x = (double(i) + 0.5) / numBins;
                                 piSum += 4.0 / (1.0 + x * x);
                               });

  std::cout << "Reduction PI is ~ " << double(piSum) / numBins << std::endl;

  //Compute PI using atomics
  RAJA::forall<execute_policy>(begin,
                               numBins,
                               [=](int i) {
                                 double x = (double(i) + 0.5) / numBins;
                                 double addToSum = 4.0 / (1.0 + x * x);
                                 RAJA::atomic::atomicAdd<atomic_policy>(atomicPiSum,addToSum);
                               });

  std::cout << "Atomic PI is ~ " << (*atomicPiSum)/numBins<< std::endl;

  return 0;
}
