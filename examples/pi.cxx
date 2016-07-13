#include "RAJA/RAJA.hxx"
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
  typedef RAJA::seq_reduce reduce_policy ;
  typedef RAJA::seq_exec   execute_policy ;
 
  int numBins = 512*512 ;

  RAJA::ReduceSum<reduce_policy, double> piSum(0.0) ;

  RAJA::forall<execute_policy>( 0, numBins, [=] RAJA_DEVICE (int i) {
      double x = (double(i) + 0.5)/numBins ;
      piSum += 4.0/(1.0 + x*x) ;
  } ) ;

  std::cout << "PI is ~ " << double(piSum)/numBins << std::endl;

  return 0;
}
