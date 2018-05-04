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
 *  Offset example
 *  Example uses a 5-box stencil
 *  to compute the sum of interior boxes
 *  intialized to be one. 
 *  We assume N x N interior nodes and
 *  padd zeros on the edge for a box of
 *  size (N+2) x (N+2)
 *
 *  -----------------
 *  | 0 | 0 | 0 | 0 |
 *  -----------------
 *  | 0 | 1 | 1 | 0 |
 *  -----------------
 *  | 0 | 1 | 1 | 0 |
 *  -----------------
 *  | 0 | 0 | 0 | 0 |   
 *  -----------------
 *
 * Illustration of N=2 box. 
 *  
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Index range segment 
 *    -  Execution policies
 */

//C-Style macros with offsets
#define box0(row,col) box0[(col+offset) + (N+2)*(row+offset)]
#define box1(row,col) box1[(col+offset) + (N+2)*(row+offset)]

//
// Functions for checking and printing results
//
void printBox(int* Box, int N); 

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA five-box-stencil example...\n";

//
// Define vector length
//
  const int N = 4; 
  const int boxSize = (N+2)*(N+2);

//
// Allocate and initialize vector data.
//
  int * box0 = memoryManager::allocate<int>(boxSize);
  int * box1 = memoryManager::allocate<int>(boxSize);

  std::memset(box0,boxSize*sizeof(int),0);
  std::memset(box1,boxSize*sizeof(int),0);

//
// C-Style intialization
//
  int offset = 1;
  for(int row=0; row<N; ++row){
    for(int col=0; col<N; ++col){      
      box0(row,col) = 1;
    }
  }

  //printBox(box0,N);
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of five-box-stencil...\n";
//
// Perform five-box stencil sum
//
  for(int row=0; row<N; ++row){
    for(int col=0; col<N; ++col){    
      
      box1(row,col) = box0(row,col) + box0(row-1,col) 
                    + box0(row+1,col) + box0(row,col-1) + box0(row,col+1);
      
    }
  }
  
  //printBox(box1,N);
//----------------------------------------------------------------------------//


//
//RAJA version
//
  RAJA::RangeSegment col_range(0,N); 
  RAJA::RangeSegment row_range(0,N); 
  

//
// Next we replace the macros with the RAJA views, making utility of their
// offset feature. 
//
  using NESTED_EXEC_POL = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,    // row
        RAJA::statement::For<0, RAJA::seq_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >  
    >;

  RAJA::kernel<NESTED_EXEC_POL>(RAJA::make_tuple(col_range, row_range), 
                                [=](int col, int row) {
                                  
                                  
                                  

                                });


//----------------------------------------------------------------------------//

//
// Clean up. 
//
  memoryManager::deallocate(box0);
  memoryManager::deallocate(box1);
  
  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Print Box
//
void printBox(int* box, int len)
{
  std::cout << std::endl;
  for(int row=0; row < len+2; ++row){
    for(int col=0; col< len+2; ++col){

      const int id = col + (len+2)*row;
      std::cout <<box[id] <<" ";

    }
    std::cout << " " << std::endl;
  }
  std::cout << std::endl;

}
