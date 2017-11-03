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

  RAJA::Index_type begin = 0;
  RAJA::Index_type numBins = 512 * 512;

  RAJA::ReduceSum<reduce_policy, double> piSum(0.0);

  RAJA::forall<execute_policy>(begin,
                               numBins,
                               [=](int i) {
                                 double x = (double(i) + 0.5) / numBins;
                                 piSum += 4.0 / (1.0 + x * x);
                               });

  std::cout << "PI<seq> is ~ " << double(piSum) / numBins << std::endl;

  #if defined(RAJA_ENABLE_TARGET_OPENMP)


  {
    typedef RAJA::policy::omp::omp_target_reduce<64> reduce_policy;
    typedef RAJA::policy::omp::omp_target_parallel_for_exec<64> execute_policy;
    int numBins = 512 * 512;

    RAJA::ReduceSum<reduce_policy, double> piSum(0.0);
    RAJA::ReduceMin<reduce_policy, double> piMin(std::numeric_limits<double>::max());
    RAJA::ReduceMax<reduce_policy, double> piMax(std::numeric_limits<double>::min());
    RAJA::ReduceMinLoc<reduce_policy, double> piMinLoc(std::numeric_limits<double>::max(),std::numeric_limits<RAJA::Index_type>::max());
    RAJA::ReduceMaxLoc<reduce_policy, double> piMaxLoc(std::numeric_limits<double>::min(),std::numeric_limits<RAJA::Index_type>::min());


    RAJA::forall<execute_policy>(0, numBins, [=](int i) {
      double x = (double(i) + 0.5) / numBins;
      piSum += 4.0 / (1.0 + x * x);
      piMin.min(-(double)i);
      piMax.max((double)i);
      RAJA::Index_type j = i;
      piMinLoc.minloc(-(double)i,j);
      piMaxLoc.maxloc((double)i,j);
    });
    std::cout << "PI<omp target> is ~ " << double(piSum) / numBins << std::endl;
    std::cout << "PImin<omp target> is ~ " << double(piMin) / numBins << std::endl;
    std::cout << "PImax<omp target> is ~ " << double(piMax) / numBins << std::endl;
    std::cout << "PIminloc<omp target> is ~ " << double(piMinLoc)/numBins << " at " << piMinLoc.getLoc() << std::endl;
    std::cout << "PImaxloc<omp target> is ~ " << double(piMaxLoc)/numBins << " at " << piMaxLoc.getLoc() << std::endl;
  }

#endif

  return 0;
}
