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

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

//
//Loop Reordering example
//
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  int DIM = 5; 
  int N0 = 8;
  int N1 = 20;
  int N2 = 10;
  int N3 = 40;
  int N4 = 30;

  size_t arrayLen = N0*N1*N2*N3*N4;

  double *P_ptr    = memoryManager::allocate<double>(arrayLen);
  double *Pout_ptr = memoryManager::allocate<double>(arrayLen);
  
  //Populate array with data
          

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0,arrayLen), [=] (RAJA::Index_type ii){
      P_prt[ii] = 1; 
    });
    
  //Wrap into views
  RAJA::View<double,RAJA::Layout<DIM> > P(P_prt, N0, N1, N2, N3, N4);
  RAJA::View<double,RAJA::Layout<DIM> > Pout(P_out, N0, N1, N2, N3, N4);

  using Pol = RAJA::nested::Policy< 
    RAJA::nested::For<4, RAJA::seq_exec>, 
      RAJA::nested::For<3, RAJA::seq_exec>,
      RAJA::nested::For<2, RAJA::seq_exec>,
      RAJA::nested::For<1, RAJA::seq_exec>,
      RAJA::nested::For<0, RAJA::seq_exec> >;

  




  
  memoryManager::deallocate(P);
  memoryManager::deallocate(Pold);
  
  return 0;
}

