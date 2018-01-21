#ifndef __VEC_FUN__
#define __VEC_FUN__

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <ctime>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

template<typename T, typename K, typename Policy>
void setArray(T A, K val, RAJA::Index_type arrLen){
    
  RAJA::forall<Policy>
    (RAJA::RangeSegment(0,arrLen), [=] (RAJA::Index_type i){
      A(i) = val;
    });

}
  
#endif
