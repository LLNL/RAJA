#include <iostream>
#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"


#define arr_type double
int main(int argc, char * argv[])
{

  srand (time(NULL));
  int tCount = 64;
  int stride = 64;
  RAJA::Index_type arrLen = stride*tCount;
  arr_type *A = new arr_type[arrLen];
  arr_type *B = new arr_type[tCount]; 
  RAJA::View<arr_type, RAJA::Layout<2> > Aview(A,tCount,stride);
  RAJA::View<arr_type, RAJA::Layout<1> > Bview(B,tCount);


  for(int i=0; i<arrLen; ++i){
    A[i]  = 1;
  }

  for(int i=0; i<tCount; ++i)
    {
      B[i] = 0; 
    }
  
   
  RAJA::RangeSegment longStride(0,stride);
  RAJA::RangeSegment simdStride(0,tCount);

#if 1
  using Pol = RAJA::nested::Policy<RAJA::nested::For<1,RAJA::omp_parallel_for_exec>, RAJA::nested::For<0,RAJA::simd_exec> >;
  RAJA::nested::forall(Pol{},
                       RAJA::make_tuple(simdStride,longStride),
                       [=] (RAJA::Index_type j, RAJA::Index_type i){                        
                         int id = i + j*tCount; 
                         B[i] += A[id];
                       });
#else  

  RAJA::forall<RAJA::omp_parallel_for_exec>(longStride, [=] (int i) {      

      double dot = 0.0;      
#pragma simd reduction(+:dot)
      for(int j=0; j<tCount; ++j){
        int id = i + j*tCount;
        dot += A[id];
      }
      B[i] = dot;
    });
#endif


  bool pass = 0; 
  for(int i=0; i<stride; ++i)
    {
      if(B[i] !=tCount)
        {
          pass = !pass; 
        }
    }

  if(pass){
    std::cout<<"Pass --- "<<std::endl;
  }else{
    std::cout<<"Fail --- "<<std::endl;
  }


  return 0;
}
