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

/*
 *  Layout Example
 *
 *  Populates a multi-dimensional array
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  basic usage of the layout object
 */

//
// Functions for checking and printing results
//
template<typename T>
void checkResult(T A, T B, int len);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

//
//Define Dimension
//
  const int dim = 4;

//
// Define stride length
//  
  const int stride1 = 4;
  const int stride2 = 9;
  const int stride3 = 12;
  const int stride4 = 6;

//
//Define Range Segments 
//
  RAJA::RangeSegment range_1(0,stride1);
  RAJA::RangeSegment range_2(0,stride2);
  RAJA::RangeSegment range_3(0,stride3);
  RAJA::RangeSegment range_4(0,stride4);

//
// Allocate and initialize vector data.
//
  int array_len = stride1*stride2*stride3*stride4;
  int* A = new int[array_len];
  int* B = new int[array_len];

  //Wrap A in a view
  RAJA::View<int, RAJA::Layout<dim> > Aview(A, stride1, stride2, stride3, stride4);

  //Create layout object
  RAJA::Layout<dim> layout(stride1, stride2, stride3, stride4);

//----------------------------------------------------------------------------//
//
// In the following, we show how a layout
// object may be used to convert from 1D index 
// to a 4D index. 
//----------------------------------------------------------------------------//   
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, array_len), [=] (int id) {
      
      //Map from 1D index to 4D index
      int i,j,m,n;
      layout.toIndices(id,i,j,m,n);

      Aview(i,j,m,n) = id%dim;
  });


//----------------------------------------------------------------------------//
//
// In the following, we show how a layout
// object may be used to convert from a 4D index 
// to a 1D index.
//----------------------------------------------------------------------------//   
  using NESTED_EXEC_POL =
    RAJA::nested::Policy< 
    RAJA::nested::For<3, RAJA::seq_exec>,
    RAJA::nested::For<2, RAJA::seq_exec>,
    RAJA::nested::For<1, RAJA::seq_exec>,                
    RAJA::nested::For<0, RAJA::seq_exec> >;


  RAJA::nested::forall(NESTED_EXEC_POL{},
                       RAJA::make_tuple(range_1,range_2,range_3, range_4),
                       [=](int i, int j, int m, int n) {

       int id = layout(i,j,m,n);
       B[id] = id%dim;
  });

//
// Check result
//
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
