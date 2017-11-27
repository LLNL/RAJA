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

  const long int n1 = 10;
  const long int n2 = 10; 
  const long int n3 = 8;
  const long int arrLen = n1*n2*n3;

  long int *A = new long int[arrLen]; 
  long int *B = new long int[arrLen];
  long int *C = new long int[arrLen];

  for(auto i=0; i<arrLen; ++i){
    A[i] = i; B[i] = -i; C[i] = 0;
  }
  
  RAJA::View<long int, RAJA::Layout<3> > Aview(A,n1,n2,n3);
  RAJA::View<long int, RAJA::Layout<3> > Bview(B,n1,n2,n3);
  RAJA::View<long int, RAJA::Layout<3> > Cview(C,n1,n2,n3);


  RAJA::forall<RAJA::seq_exec>
    (RAJA::RangeSegment(0,n1), [=] (RAJA::Index_type i){

      RAJA::forall<RAJA::seq_exec>
        (RAJA::RangeSegment(0,n2), [=] (RAJA::Index_type j){

          RAJA::forall<RAJA::simd_exec>
            (RAJA::RangeSegment(0,n3), [=] (RAJA::Index_type k){
              Aview(i,j,k) = 0.0;

        });
      });
    });
  
  


  




  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
