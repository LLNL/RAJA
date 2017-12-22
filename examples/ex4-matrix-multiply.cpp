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
//#include <cmath>
#include <iostream>
#include <cstring>
//#include <algorithm>
//#include <initializer_list>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Matrix Multiplication Example
 *
 *  Example computes the product of two square matrices and introduces
 *  RAJA nested loop capabilities via a sequence of implementations.
 *
 *  RAJA features shown:
 *    - `forall` loop iteration template method
 *    - Index range segment
 *    - view abstraction
 *    - 'nested' loop abstractions
 *    - nested loop reordering
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE_X = 16;
const int CUDA_BLOCK_SIZE_Y = 16;
#endif


//
// Define dimensionality of matrices.
//
const int DIM = 2;

//
// Define macros to simplify row-col indexing (non-RAJA implementations only)
//
#define A(r, c) A[c + N * r]
#define B(r, c) B[c + N * r]
#define C(r, c) C[c + N * r]

//
// Functions for checking results
//
template <typename T>
void checkResult(T *C, int N);

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);

//
// Functions for printing results
//
template <typename T>
void printResult(T *C, int N);

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA matrix multiplication example...\n";

//
// Define num rows/cols in matrix
//
  const int N = 1000;

//
// Allocate and initialize matrix data.
//
  double *A = memoryManager::allocate<double>(N * N);
  double *B = memoryManager::allocate<double>(N * N);
  double *C = memoryManager::allocate<double>(N * N);

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A(row, col) = row;
      B(row, col) = col;
    }
  }


  std::cout << "\n Running C-version of daxpy...\n";

  std::memset(C, 0, N*N * sizeof(double)); 

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += A(row, k) * B(k, col);
      }

      C(row, col) = dot;
    }
  }
  checkResult<double>(C, N);
//printResult<double>(C, N);  


//
// In the following RAJA implementations of matrix multiplication, we 
// use RAJA 'View' objects to access the matrix data. A RAJA view
// holds a pointer to a data array and enables multi-dimensional indexing 
// into that data, similar to the macros we defined above.
//
  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, N);

//
// Here, we define RAJA range segments to define the ranges of
// row and column indices
//
  RAJA::RangeSegment row_range(0, N);
  RAJA::RangeSegment col_range(0, N);


//
// In the first RAJA implementation, we replace the outer 'row' loop
// with a RAJA::forall statement. The lambda loop body contains the
// inner loops.
//

  std::cout << "\n Running sequential mat-mult (RAJA-row)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 

  RAJA::forall<RAJA::seq_exec>( row_range, [=](int row) {    

    for (int col = 0; col < N; ++col) {
        
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
        
      Cview(row, col) = dot;

    }
      
  });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);  


//
// Next, we replace the outer 'row' loop and the inner 'col' loop 
// with RAJA::forall statements.
//

  std::cout << "\n Running sequential mat-mult (RAJA-row, RAJA-col)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  RAJA::forall<RAJA::seq_exec>( row_range, [=](int row) {    

    RAJA::forall<RAJA::seq_exec>( col_range, [=](int col) {    

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }

      Cview(row, col) = dot;

    });

  });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//
// Next, we use a RAJA nested::foral method to execute the calculation.
// This is different than RAJA 'forall' and so a few points of exmplanation
// are in order:
//
// 1) A range and lambda index argument are required for each level in
//    the loop nest. Here, we have two of each since we have a doubly-nested
//    loop.
// 2) The ranges for the loop nest level are specified in a RAJA tuple object.
//    The order of ranges in the tuple must match the order of args to the
//    lambda for this to be correct, in general. RAJA provides strongly-typed
//    indices to help with this. However, this example does not use them.
// 3) An execution policy is required for each level in the loop nest. These
//    are specified in the 'RAJA::nested::For' templates in the 
//    'RAJA::nested::Policy type.
// 4) The loop nest ordering is specified in the nested execution policy -- 
//    the first 'For' policy is the outermost loop, the second 'For' policy 
//    is the loop nested inside the outermost loop, and so on. 
// 5) The integer values that are the first template arguments to the policies
//    indicate which range/lambda argument, the policy applies to.
//

  std::cout << "\n Running sequential mat-mult (RAJA-nested)...\n";
    
  std::memset(C, 0, N*N * sizeof(double));

  using NESTED_EXEC_POL = 
    RAJA::nested::Policy< RAJA::nested::For<1, RAJA::seq_exec>,    // row
                          RAJA::nested::For<0, RAJA::seq_exec> >;  // col

  RAJA::nested::forall(NESTED_EXEC_POL{},
                       RAJA::make_tuple(col_range, row_range),
                       [=](int col, int row) {
      
    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += Aview(row, k) * Bview(k, col);
    }
        
    Cview(row, col) = dot;

  });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
  

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running OpenMP mat-mult (RAJA-nested)...\n";
  
  using NESTED_EXEC_POL1 = 
    RAJA::nested::Policy< RAJA::nested::For<1, RAJA::omp_parallel_for_exec>,// row
                          RAJA::nested::For<0, RAJA::seq_exec> >;        // col

  RAJA::nested::forall(NESTED_EXEC_POL1{}, 
                       RAJA::make_tuple(col_range, row_range),
                       [=](int col, int row) {
      
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }

      Cview(row, col) = dot;

  });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - swap loops)...\n";

  //
  // Note that we swap the template arguments for the nested policy. 
  // This swaps the loop nest ordering so the col loop is on the outside
  // and the row loop is nested within it. The execution policies on each
  // loop remain the same as the previous implementation; i.e., col iterations
  // run sequentially, while row iterations execute in parallel.
  // 
  using NESTED_EXEC_POL2 =
    RAJA::nested::Policy< RAJA::nested::For<0, RAJA::seq_exec>,         // col
                          RAJA::nested::For<1, RAJA::omp_parallel_for_exec> >;// row

  RAJA::nested::forall(NESTED_EXEC_POL2{},
                       RAJA::make_tuple(col_range, row_range),
                       [=](int col, int row) {
  
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }

      Cview(row, col) = dot;

  });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif


#if defined(RAJA_ENABLE_CUDA)
  std::cout << "\n Running CUDA mat-mult (RAJA-nested)...\n";

  //
  // This policy collapses the col and row loops into a single CUDA kernel
  // using two-dimensional CUDA thread blocks with x and y dimensions defined
  // by the CUDA_BLOCK_SIZE_X and CUDA_BLOCK_SIZE_Y template arguments,
  // respectively. 
  // 
  using NESTED_EXEC_POL3 = 
    RAJA::nested::Policy< RAJA::nested::CudaCollapse<
      RAJA::nested::For<1, RAJA::cuda_threadblock_y_exec<CUDA_BLOCK_SIZE_Y>>,
      RAJA::nested::For<0, RAJA::cuda_threadblock_x_exec<CUDA_BLOCK_SIZE_X>>> >;

  RAJA::nested::forall(NESTED_EXEC_POL3{},
                       RAJA::make_tuple(col_range, row_range), 
                       [=] RAJA_DEVICE (int col, int row) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
          
      Cview(row, col) = dot;

  });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif

//
// Clean up.
//
  memoryManager::deallocate(A);
  memoryManager::deallocate(B);
  memoryManager::deallocate(C);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Functions to check result and report P/F.
//
template <typename T>
void checkResult(T* C, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( abs( C(row, col) - row * col * N ) > 10e-12 ) { match = false; } 
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( abs( Cview(row, col) - row * col * N ) > 10e-12 ) { match = false; }
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Functions to print result.
//
template <typename T>
void printResult(T* C, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = "
                << C(row, col) << std::endl;
    }
  }
  std::cout << std::endl;
}

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = "
                << Cview(row, col) << std::endl;
    }
  }
  std::cout << std::endl;
}
