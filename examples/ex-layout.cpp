//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

/*
 *  Layout Example
 *
 *  Populates a 4-dimensional array
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  basic usage of the layout object
 */

//
// Function to check result
//
template<typename T>
void checkResult(T A, T B, int len);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

//
//Define dimension
//
  const int dim = 4;

//
// Define stride length
//  
  const int Istride = 2;
  const int Jstride = 3;
  const int Mstride = 4;
  const int Nstride = 5;

//
//Define range segment
//
  RAJA::RangeSegment Irange(0,Istride);
  RAJA::RangeSegment Jrange(0,Jstride);
  RAJA::RangeSegment Mrange(0,Mstride);
  RAJA::RangeSegment Nrange(0,Nstride);

//
// Allocate 1D arrays
//
  int array_len = Istride*Jstride*Mstride*Nstride;

  int* A = new int[array_len];
  int* B = new int[array_len];

  //Wrap A in a view
  RAJA::View<int, RAJA::Layout<dim> > A_IJMN_view(A, Istride, Jstride, Mstride, Nstride);

  //Create layout objects assuming an IJMN indexing
  RAJA::Layout<dim> IJMN_layout = RAJA::make_permuted_layout({Istride,Jstride,Mstride,Nstride}, RAJA::as_array<RAJA::Perm<0,1,2,3>>::get());
  //Equivalent to : RAJA::Layout<dim> IJMN_Layout(Istride,JStride,MStride,Nstride);

//----------------------------------------------------------------------------//
//
// In the following example we show how a layout
// object may be used to convert a 1D index to a 4D index.
//----------------------------------------------------------------------------//   
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, array_len), [=] (int id) {
      
      //Map from 1D index to 4D index
      int i,j,m,n;
      IJMN_layout.toIndices(id,i,j,m,n);
      
      A_IJMN_view(i,j,m,n) = id%dim;
  });


//----------------------------------------------------------------------------//
//
// In the following example we show how a layout
// object may be used to convert from a 4D index 
// to a 1D index.
//----------------------------------------------------------------------------//
  using IJMN_EXEC_POL =
    RAJA::nested::Policy< 
    RAJA::nested::For<3, RAJA::seq_exec>,
    RAJA::nested::For<2, RAJA::seq_exec>,
    RAJA::nested::For<1, RAJA::seq_exec>,                
    RAJA::nested::For<0, RAJA::seq_exec> >;

  
  RAJA::nested::forall(IJMN_EXEC_POL{},
                       RAJA::make_tuple(Irange, Jrange, Mrange, Nrange),
                       [=](int i, int j, int m, int n) {
                         
       int id = IJMN_layout(i,j,m,n);
       B[id] = id%dim;
  });

//
// Check result
//
  checkResult(A, B, array_len);



//============================================================================//
//  
//As a second example we can permute the the layout object
//============================================================================//

  //Create a view with permuted strides
  RAJA::View<int, RAJA::Layout<dim> > A_NMJI_view(A, Nstride, Mstride, Jstride, Istride);
  
  //Permute the layout
  RAJA::Layout<dim> NMJI_layout = RAJA::make_permuted_layout({Istride,Jstride,Mstride,Nstride}, RAJA::as_array<RAJA::Perm<3,2,1,0>>::get());
  
//----------------------------------------------------------------------------//
//
//Conversion of 1D index to 4D index with an NMJI layout
//----------------------------------------------------------------------------//
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, array_len), [=] (int id) {
      
      //Map from 1D index to 4D index
      int m,n,i,j;
      NMJI_layout.toIndices(id,i,j,m,n);
      A_NMJI_view(n,m,j,i) = id/dim;

    });
  
//----------------------------------------------------------------------------//
//
// Conversion on from a 4D index with permuted layout (NMJI) to
// to a 1D index.
//----------------------------------------------------------------------------//
    using NMJI_EXEC_POL =
    RAJA::nested::Policy< 
    RAJA::nested::For<3, RAJA::seq_exec>, 
    RAJA::nested::For<2, RAJA::seq_exec>,
    RAJA::nested::For<1, RAJA::seq_exec>,
    RAJA::nested::For<0, RAJA::seq_exec>>;


  RAJA::nested::forall(NMJI_EXEC_POL{},
                       RAJA::make_tuple(Irange, Jrange, Mrange, Nrange),
                       [=] (int i, int j, int m, int n) {
                         
                         int id = NMJI_layout(i,j,m,n);
                         B[id] = id/dim;                          
  });
  checkResult(A, B, array_len);


  
//
// Clean up. 
//
  delete[] A;
  delete[] B;
  std::cout << "\n DONE!...\n";

  return 0;
}

template<typename T>
void checkResult(T A, T B, int len)
{

  bool match = true;
  for(int i=0; i<len; ++i){
    if( std::abs( A[i]-B[i] ) > 1e-8 ) {
      match = false;
    }
  }
  
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }

}
