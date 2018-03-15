#include <cstdlib>
#include <iostream>
#include <chrono>

#include <time.h>       /* time */
#include "Layout.hpp"
#include "View.hpp"
#include "RangeSegment.hpp"
#include "camp/camp.hpp"

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



  
  
  return 0;
}


