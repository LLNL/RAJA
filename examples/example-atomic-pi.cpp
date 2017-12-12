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
  Example a : Atomic Pi
  This example illustrates presents an two ways to compute Pi.
  The first approach makes use of the RAJA reduction variables.
  The second approach employs RAJA atomic variables.  
*/
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  
  RAJA::Index_type begin = 0;
  RAJA::Index_type numBins = 512 * 512;
  typedef RAJA::seq_reduce reduce_policy;
  typedef RAJA::seq_exec execute_policy;  

  // Computing PI using reduction
  RAJA::ReduceSum<reduce_policy, double> piSum(0.0);
  
  RAJA::forall<execute_policy>(begin, numBins, [=](int i) {
    double x = (double(i) + 0.5) / numBins;
    piSum += 4.0 / (1.0 + x * x);
  });

  std::cout << "Reduction PI is ~ " << double(piSum) / numBins << std::endl;
  
  // Compute PI using atomic operations
  typedef RAJA::atomic::seq_atomic atomic_policy; 
  double *atomicPiSum = memoryManager::allocate<double>(1);
  *atomicPiSum = 0; 
 
  RAJA::forall<execute_policy>(begin, numBins, [=](int i) {
    double x = (double(i) + 0.5) / numBins;
    double addToSum = 4.0 / (1.0 + x * x);
    RAJA::atomic::atomicAdd<atomic_policy>(atomicPiSum,addToSum);
  });

  std::cout << "Atomic PI is ~ " << (*atomicPiSum)/numBins<< std::endl;

  memoryManager::deallocate(atomicPiSum);

  return 0;
}
