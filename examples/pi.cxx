#include <cstdlib>
#include <iostream>

#include "RAJA/RAJA.hxx"
#include "RAJA/internal/defines.hxx"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
  typedef RAJA::seq_reduce reduce_policy;
  typedef RAJA::seq_exec execute_policy;

  RAJA::Index_type begin = 0;
  RAJA::Index_type numBins = 512 * 512;

  RAJA::ReduceSum<reduce_policy, double> piSum(0.0);

  RAJA::forall<execute_policy>(begin, numBins, [=](int i) {
    double x = (double(i) + 0.5) / numBins;
    piSum += 4.0 / (1.0 + x * x);
  });

  std::cout << "PI is ~ " << double(piSum) / numBins << std::endl;

  return 0;
}
