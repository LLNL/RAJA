#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"
#include <cstdlib>

TEST(PiTest, CheckValue) {
  typedef RAJA::omp_reduce             reduce_policy ;
  typedef RAJA::omp_parallel_for_exec  execute_policy ;
 
  int numBins = 1024*1024 ;

  RAJA::ReduceSum<reduce_policy, double> piSum(0.0) ;

  RAJA::forall<execute_policy>( 0, numBins, [=] RAJA_DEVICE (int i) {
      double x = (double(i) + 0.5)/numBins ;
      piSum += 4.0/(1.0 + x*x) ;
  } ) ;

  ASSERT_NEAR(double(piSum)/numBins, 3.14, 0.1);
}
