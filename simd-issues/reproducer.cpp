#include <cstdlib>
#include <iostream>
#include <chrono>

#include <time.h>       /* time */
#include "Layout.hpp"
#include "View.hpp"
#include "nested.hpp"
#include "camp/camp.hpp"
#include "RangeSegment.hpp"

#include "seq_policy.hpp"
#include "simd_policy.hpp"
#include "omp_policy.hpp"


using arr_type = double;

int main(int argc, char *argv[])
{

  srand (time(NULL));
  Index_type stride = rand() % 50 + 1;
  Index_type arrLen = stride*stride*stride;
  arr_type *A = new arr_type[arrLen];
  arr_type *B = new arr_type[arrLen];
  arr_type *C = new arr_type[arrLen];

  for(int i=0; i<arrLen; ++i){
    A[i] = 0.5;
    B[i] = 2;
  }

  RAJA::View<arr_type, RAJA::Layout<3> > Aview(A,stride,stride,stride);
  RAJA::View<arr_type, RAJA::Layout<3> > Bview(B,stride,stride,stride);
  RAJA::View<arr_type, RAJA::Layout<3> > Cview(C,stride,stride,stride);

  RAJA::RangeSegment myStride(0,stride);

  //using Pol = RAJA::nested::Policy<RAJA::nested::For<0,RAJA::simd_exec>>;
  
  using Pol = RAJA::nested::Policy<RAJA::nested::For<2,RAJA::omp_parallel_for_exec>,
                                   RAJA::nested::For<1,RAJA::seq_exec>,
                                   RAJA::nested::For<0,RAJA::simd_exec> >;

  
  
  return 0;
}


