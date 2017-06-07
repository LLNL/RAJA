#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
  constexpr const size_t Teams = 64;
  using reduce_policy = RAJA::omp_target_reduce<Teams>;
  using execute_policy = RAJA::omp_target_parallel_for_exec<Teams>;

  RAJA::Index_type begin = 0;
  RAJA::Index_type numBins = 512 * 512;

  RAJA::ReduceSum<reduce_policy, double> piSum(0.0);
  RAJA::ReduceMin<reduce_policy, double> piMin(std::numeric_limits<double>::max());
  RAJA::ReduceMax<reduce_policy, double> piMax(std::numeric_limits<double>::min());
  RAJA::ReduceMinLoc<reduce_policy, double> piMinLoc(std::numeric_limits<double>::max(),std::numeric_limits<RAJA::Index_type>::max());
  RAJA::ReduceMaxLoc<reduce_policy, double> piMaxLoc(std::numeric_limits<double>::min(),std::numeric_limits<RAJA::Index_type>::min());

  RAJA::forall<execute_policy>(begin, numBins, [=](int i) {
    double x = (double(i) + 0.5) / numBins;
    piSum += 4.0 / (1.0 + x * x);
    piMin.min(-(double)i);
    piMax.max((double)i);
    RAJA::Index_type j = i;
    piMinLoc.minloc(-(double)i,j);
    piMaxLoc.maxloc((double)i,j);
  });

  std::cout << "PI is ~ " << double(piSum) / numBins << std::endl;
  std::cout << "PImin is ~ " << double(piMin) / numBins << std::endl;
  std::cout << "PImax is ~ " << double(piMax) / numBins << std::endl;
  std::cout << "PIminloc is ~ " << double(piMinLoc)/numBins << " at " << piMinLoc.getLoc() << std::endl;
  std::cout << "PImaxloc is ~ " << double(piMaxLoc)/numBins << " at " << piMaxLoc.getLoc() << std::endl;

  return 0;
}
