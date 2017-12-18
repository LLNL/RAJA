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
//Define in function scope for the RAJA CUDA variant
//

RAJA_INDEX_VALUE(ID, "ID");
RAJA_INDEX_VALUE(IZ, "IZ");
RAJA_INDEX_VALUE(IG, "IG");
RAJA_INDEX_VALUE(IM, "IM");

//
//Loop Reordering example
//
int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{


  int N0 = 8;
  int N1 = 20;
  int N2 = 10;
  int N3 = 40;
  const int DIM = 4; 

  size_t arrayLen = N0*N1*N2*N3;
  
  double *Ptr    = memoryManager::allocate<double>(arrayLen);
  double *Pouttr = memoryManager::allocate<double>(arrayLen);
  
  //Populate array with data          
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0,arrayLen), [=] (RAJA::Index_type ii){
      Ptr[ii]    = 1; 
      Pouttr[ii] = 0; 
    });

#if 0
  using myPol = RAJA::nested::Policy<   
  RAJA::nested::TypedFor<0, RAJA::loop_exec, ID>,
  RAJA::nested::TypedFor<1, RAJA::loop_exec,IZ>   
    >;
#else 
  using myPol = RAJA::nested::Policy<   
  RAJA::nested::TypedFor<1, RAJA::loop_exec,IZ>,
  RAJA::nested::TypedFor<0, RAJA::loop_exec, ID>
    >;
#endif

 //RAJA::nested::For<2, RAJA::loop_exec, IG>,
    //RAJA::nested::For<2, RAJA::loop_exec, IM> >
  ///RAJA::nested::For<1, RAJA::loop_exec,RAJA::Index_type tz> >;

  RAJA::RangeSegment Range0(0,4);
  RAJA::RangeSegment Range1(0,1);

  RAJA::nested::forall(myPol{},
                     camp::make_tuple(Range0, Range1),
                     //[=] (RAJA::Index_type i1, RAJA::Index_type i0){
                     [=] (ID i0, IZ i1) {
                       printf("%ld, %ld \n",(long int)*i0, (long int)*i1);
                     });



#if 0
  RAJA_INDEX_VALUE ID; 
  RAJA_INDEX_VALUE IZ;
  RAJA_INDEX_VALUE IG;
  RAJA_INDEX_VALUE IM;
  
  //Wrap into views
  RAJA::View<double,RAJA::Layout<DIM> > P(Ptr, ID, IZ, IG, IM);
  RAJA::View<double,RAJA::Layout<DIM> > Pout(Pouttr, ID, IZ, IG, IM);
  //RAJA::View<double,RAJA::Layout<DIM> > P(Ptr, N3, N2, N1, N0);
  //RAJA::View<double,RAJA::Layout<DIM> > Pout(Pouttr, N3, N2, N1, N0);


  //Setup execution policies
  using Pol = RAJA::nested::Policy< 
  RAJA::nested::For<3, RAJA::loop_exec, IZ>,
    RAJA::nested::For<2, RAJA::loop_exec, IG>,
    RAJA::nested::For<1, RAJA::loop_exec, IM>,
    RAJA::nested::For<0, RAJA::loop_exec, ID> > ;


  //Setup range segments
  RAJA::RangeSegment IZRange(0,N0);
  RAJA::RangeSegment IGRange(0,N1);
  RAJA::RangeSegment IMRange(0,N2);
  RAJA::RangeSegment IDRange(0,N3);

  RAJA::nested::forall(Pol{}, 
                       camp::make_tuple(RangeN3, RangeN2, RangeN1, RangeN0),
                       [=] (IZ i0, IG i1, IM i2, ID i3) { 
                         //RAJA::Index_type i3, RAJA::Index_type i2, RAJA::Index_type i1, RAJA::Index_type i0) {
                         Pout(i3,i2,i1,i0) =  5*Pout(i3,i2,i1,i0);                         
                       }); 
#endif

#if (RAJA_ENABLE_CUDA)
cudaDeviceSynchronize();
#endif


memoryManager::deallocate(Ptr);
memoryManager::deallocate(Pouttr);
  
  return 0;
}

