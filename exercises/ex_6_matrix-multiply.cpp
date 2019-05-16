//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Matrix Multiplication Example
 *
 *  Example computes the product of two square matrices and introduces
 *  RAJA nested loop capabilities via a sequence of implementations.
 *
 *  RAJA features shown:
 *    - Index range segment
 *    - View abstraction
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

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

//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of matrix multiplication...\n";

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


//----------------------------------------------------------------------------//

//
// In the following RAJA implementations of matrix multiplication, we 
// use RAJA 'View' objects to access the matrix data. A RAJA view
// holds a pointer to a data array and enables multi-dimensional indexing 
// into that data, similar to the macros we defined above.
//
  
  //TODO: Create view objects

//----------------------------------------------------------------------------//

//
// Here, we define RAJA range segments to define the ranges of
// row, column, and dot-product loops
//

  //TODO: Use RAJA range segments to define the iteration space
  //      for the column, row, and dot product ranges

//----------------------------------------------------------------------------//

//
// In the next few examples, we show ways that we can use RAJA::forall
// statements for the matrix multiplication kernel. This usage is not
// recommended for performance reasons. Specifically, it limits the amount
// of parallelism that can be exposed to less than is possible. We show 
// this usage here, to make this point clear. Later in this file, we 
// introduce RAJA nested loop abstractions and show that we can extract all 
// available parallelism.
//
//
// In the first RAJA implementation, we replace the outer 'row' loop
// and 'col' loops with a RAJA::forall statement. The lambda expression contains the
// inner loops.
//

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential mat-mult (RAJA-row)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 

  //TODO: Create a matrix multiplication kernel with
  //      an outer RAJA forall loops.

  //checkResult<double>(Cview, N); //TODO: Uncomment once Cview is implemented
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

//
// Next, we use a RAJA::kernel method to execute the kernel. These examples,
// illustrate the basic kernel interface and mechanics. The execution policies
// express the outer row and col loops using the RAJA kernel interface. Later, 
// in this file we show some more complex policy examples where we express all 
// three loops using the kernel interface and use additional kernel features.
//
// This is different than RAJA::forall and so a few points of exmplanation
// are in order:
//
// 1) A range and lambda index argument are required for each level in
//    the loop nest. Here, we have two of each since we have a doubly-nested
//    loop.
// 2) A range for each loop nest level is specified in a RAJA tuple object.
//    The order of ranges in the tuple must match the order of args to the
//    lambda for this to be correct, in general. RAJA provides strongly-typed
//    indices to help with this. However, this example does not use them.
// 3) An execution policy is required for each level in the loop nest. These
//    are specified in the 'RAJA::statement::For' templates in the
//    'RAJA::KernelPolicy type.
// 4) The loop nest ordering is specified in the nested execution policy -- 
//    the first 'For' policy is the outermost loop, the second 'For' policy 
//    is the loop nested inside the outermost loop, and so on. 
// 5) The integer values that are the first template arguments to the policies
//    indicate which range/lambda argument, the policy applies to.
//

  std::cout << "\n Running sequential mat-mult (RAJA-nested)...\n";
    
  std::memset(C, 0, N*N * sizeof(double));

    //TODO: Create a matrix multiplication kernel using
    //      RAJA kernel and sequential policies.
    //      Use a single lambda to encapsulate the loop
    //      body.

  //checkResult<double>(Cview, N); //TODO: Uncomment once Cview is implemented
//printResult<double>(Cview, N);
  

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - omp outer)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 
  
  //TODO: Create a matrix multiplication kernel using
  //      RAJA kernel and an omp parallel outer loop.
  //      Use a single lambda to encapsulate the loop
  //      body.

  //checkResult<double>(Cview, N); //TODO: Uncomment once Cview is implemented
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - collapse)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 
  
  //
  // This policy collapses the row and col loops in an OpenMP parallel region.
  // This is the same as using an OpenMP 'parallel for' directive on the 
  // outer loop with a 'collapse(2) clause.
  //

  //TODO: Create a matrix multiplication kernel with
  //      a collapse OpenMP parallel statement.
  //      Use a single lambda to encapsulate the loop
  //      body.

  //checkResult<double>(Cview, N); //TODO: Uncomment once Cview is implemented
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_OPENMP

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running OpenMP mat-mult with multiple lambdas and loop collapse...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //TODO: Create a matrix multiplication kernel.
  //      Use three lambdas and an OpenMP collapse policy.

  //checkResult<double>(Cview, N); //TODO: Uncomment once Cview is implemented
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_OPENMP

//----------------------------------------------------------------------------//

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
      if ( std::abs( C(row, col) - row * col * N ) > 10e-12 ) { 
        match = false; 
      }
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
      if ( std::abs( Cview(row, col) - row * col * N ) > 10e-12 ) { 
        match = false; 
      }
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
