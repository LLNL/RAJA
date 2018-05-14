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
#include <cmath>

#include "RAJA/RAJA.hpp"
#include "memoryManager.hpp"

/*
 *  Offset example
 *  Example uses a 5-box stencil
 *  to compute the sum of interior boxes.
 *  We assume N x N interior nodes and
 *  a padded edge of zeros for a box of
 *  size (N+2) x (N+2)
 *
 * Input: 
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
 * Expected output for N = 2:
 *  -----------------
 *  | 0 | 0 | 0 | 0 |
 *  -----------------
 *  | 0 | 3 | 3 | 0 |
 *  -----------------
 *  | 0 | 3 | 3 | 0 |
 *  -----------------
 *  | 0 | 0 | 0 | 0 |   
 *  -----------------
 *
 * We simplify index calculuations by using 
 * RAJA::make_offset_layout and RAJA::Views.
 * RAJA::make_offset_layout enables developers 
 * to adjust the start and end values of the array.
 * 
 * Here we chose to enumerate our interior in the 
 * following manner. 
 *
 *  ---------------------------------------
 *  | (-1, 2) | (0, 2)  | (1, 2)  | (2, 2)|   
 *  ---------------------------------------
 *  | (-1, 1) | (0, 1)  | (1, 1)  | (2, 1) |   
 *  ---------------------------------------
 *  | (-1, 0) | (0, 0)  | (1, 0)  | (2, 0) |   
 *  ---------------------------------------
 *  | (-1,-1) | (0, -1) | (1, -1) | (2, -1)|   
 *  ---------------------------------------
 *
 *  Notably (0, 0) corresponds
 *  to the bottom left corner of the region we wish to 
 *  apply our stencil to.
 *  
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    -  Offset-layouts for RAJA Views
 *    -  Index range segment 
 *    -  Execution policies
 */

/*
 * Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
#endif

//C-Style macros with offsets
#define box0(row,col) box0[(col-offset) + (N+2)*(row-offset)]
#define box_ref(row,col) box_ref[(col-offset) + (N+2)*(row-offset)]

//
// Functions for printing and checking results
//
void printBox(int * Box, int boxN); 
void checkResult(int * compBox, int * refBox, int boxSize);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA box-stencil example...\n";

//
// Define vector length
//
  const int N = 4; 
  const int boxN = N+2;
  const int boxSize = boxN*boxN;

//
// Allocate and initialize vector data
//
  int * box0    = memoryManager::allocate<int>(boxSize*sizeof(int));
  int * box1    = memoryManager::allocate<int>(boxSize*sizeof(int));
  int * box_ref = memoryManager::allocate<int>(boxSize*sizeof(int));

  std::memset(box0,0,boxSize*sizeof(int));
  std::memset(box1,0,boxSize*sizeof(int));
  std::memset(box_ref,0,boxSize*sizeof(int));

//
// C-Style intialization
//
  const int offset = -1;
  for(int row=0; row<N; ++row){
    for(int col=0; col<N; ++col){
      box0(row,col) = 1;
    }
  }
  
  //prints intial box
  //printBox(box0,boxN);
//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of five-box-stencil...\n";
//
// Perform five-box stencil sum
//
  for(int row=0; row<N; ++row){
    for(int col=0; col<N; ++col){    
      
      box_ref(row,col) = box0(row,col) + box0(row-1,col)         
                       + box0(row+1,col) + box0(row,col-1) + box0(row,col+1);
      
    }
  }
  
  //printBox(box_ref,boxN);
//----------------------------------------------------------------------------//

//
//RAJA versions
//

//
//Create loop bounds
//
  RAJA::RangeSegment col_range(0,N);
  RAJA::RangeSegment row_range(0,N); 
  
//
// Here we replace the macros with RAJA views and make utility of offset-layouts.
//

//
//Dimension of the array
//
  const int DIM = 2;

//
//The first array corresponds to the coordinates of the bottom left corner
//and the second array corresponds to the coordinates of the top right corner.
//
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{-1,-1}}, {{N,N}});
  RAJA::View<int, RAJA::OffsetLayout<DIM>>box0view(box0,layout);
  RAJA::View<int, RAJA::OffsetLayout<DIM>>box1view(box1,layout);
  
//----------------------------------------------------------------------------//
  std::cout << "\n Running sequential box-stencil (RAJA-Kernel - sequential)...\n";  
  using NESTED_EXEC_POL = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,    // row
        RAJA::statement::For<0, RAJA::seq_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >  
    >;

  RAJA::kernel<NESTED_EXEC_POL>
    (RAJA::make_tuple(col_range, row_range), 
     [=](int col, int row) {
      
      box1view(row,col) = box0view(row,col) + box0view(row-1,col) 
                        + box0view(row+1,col) + box0view(row,col-1) + box0view(row,col+1);
    });

  //printBox(box1,boxN);
  checkResult(box1, box_ref, boxSize);
//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential box-stencil (RAJA-Kernel - omp parallel for)...\n";  
  using NESTED_EXEC_POL2 = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::omp_parallel_for_exec, // row
        RAJA::statement::For<0, RAJA::seq_exec,            // col
          RAJA::statement::Lambda<0>
        > 
      > 
    >;

  RAJA::kernel<NESTED_EXEC_POL2>
    (RAJA::make_tuple(col_range, row_range), 
     [=](int col, int row) {
      
      box1view(row,col) = box0view(row,col) + box0view(row-1,col) 
                        + box0view(row+1,col) + box0view(row,col-1) + box0view(row,col+1);
    });

  //printBox(box1,N);
  checkResult(box1, box_ref, boxSize);
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running sequential box-stencil (RAJA-Kernel - cuda)...\n";

  using NESTED_EXEC_POL5 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_threadblock_exec<CUDA_BLOCK_SIZE>,   //row
          RAJA::statement::For<0, RAJA::cuda_threadblock_exec<CUDA_BLOCK_SIZE>, //col
            RAJA::statement::Lambda<0>
           >
          >
        >
      >;

  RAJA::kernel<NESTED_EXEC_POL2>
    (RAJA::make_tuple(col_range, row_range), 
     [=](int col, int row) {
      
      box1view(row,col) = box0view(row,col) + box0view(row-1,col) 
                        + box0view(row+1,col) + box0view(row,col-1) + box0view(row,col+1);
    });

  //printBox(box1,boxN);
  checkResult(box1, box_ref, boxSize);
//----------------------------------------------------------------------------//
#endif

//
// Clean up. 
//
  memoryManager::deallocate(box0);
  memoryManager::deallocate(box1);
  memoryManager::deallocate(box_ref);
  
  std::cout << "\n DONE!...\n";
  return 0;
}

//
// Print Box
//
void printBox(int* box, int boxN)
{
  std::cout << std::endl;
  for(int row=0; row < boxN; ++row){
    for(int col=0; col< boxN; ++col){

      const int id = col + boxN*row;
      std::cout <<box[id] <<" ";

    }
    std::cout << " " << std::endl;
  }
  std::cout << std::endl;

}

//
//Check Result
//
void checkResult(int * compBox, int * refBox, int len)
{

  bool pass = true;
  
  for(int i = 0; i <len; ++i ){
    if(compBox[i] != refBox[i]) pass = false;
  }

  if ( pass ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}
