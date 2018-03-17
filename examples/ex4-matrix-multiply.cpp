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
 *    - Collapsing loops under OpenMP and CUDA policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
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

/*
  Define CUDA matrix multiplication kernel for comparison to RAJA version
*/
#if defined(RAJA_ENABLE_CUDA)
__global__ void matMultKernel(int N, double* C, double* A, double* B)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ( row < N && col < N ) {
    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += A(row, k) * B(k, col);
    }

    C(row, col) = dot;
  }
}
#endif

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
//const int N = CUDA_BLOCK_SIZE * CUDA_BLOCK_SIZE; 

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
  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, N);

//----------------------------------------------------------------------------//

//
// Here, we define RAJA range segments to define the ranges of
// row and column indices
//
  RAJA::RangeSegment row_range(0, N);
  RAJA::RangeSegment col_range(0, N);


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
// with a RAJA::forall statement. The lambda expression contains the
// inner loops.
//

//----------------------------------------------------------------------------//

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


//----------------------------------------------------------------------------//

//
// Next, we replace the outer 'row' loop and the inner 'col' loop 
// with RAJA::forall statements. This will also work with parallel
// execution policies, such as OpenMP and CUDA, with caveats and
// restrictions.
//
// However, nesting RAJA::forall calls like this is not recommended as
// it limits the ability to expose parallelism and flexibility for
// implementation alternatives.
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

//----------------------------------------------------------------------------//

//
// Next, we use a RAJA::kernel method to execute the calculation.
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

  using NESTED_EXEC_POL = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,    // row
        RAJA::statement::For<0, RAJA::seq_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >  
    >;

  RAJA::kernel<NESTED_EXEC_POL>(
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
  

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - omp outer)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 
  
  using NESTED_EXEC_POL1 = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::omp_parallel_for_exec, // row
        RAJA::statement::For<0, RAJA::seq_exec,            // col
          RAJA::statement::Lambda<0>
        > 
      > 
    >;

  RAJA::kernel<NESTED_EXEC_POL1>(
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

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - omp inner)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 
  
  //
  // Swapping the template arguments in this nested policy swaps the loop 
  // nest ordering so the col loop is on the outside and the row loop is 
  // nested within it. The execution policies on each loop remain the same 
  // as the previous implementation; i.e., col (outer) iterations run 
  // sequentially, while row (inner) iterations execute in parallel.
  // 
  using NESTED_EXEC_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::seq_exec,                  // col
        RAJA::statement::For<1, RAJA::omp_parallel_for_exec,   // row
          RAJA::statement::Lambda<0>
        > 
      > 
    >;

  RAJA::kernel<NESTED_EXEC_POL2>(
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

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - collapse)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 
  
  //
  // This policy collapses the row and col loops in an OpenMP parallel region.
  // This is the same as using an OpenMP 'parallel for' directive on the 
  // outer loop with a 'collapse(2) clause.
  //
  using NESTED_EXEC_POL3 = 
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<1, 0>,   // row, col
        RAJA::statement::Lambda<0>
      > 
    >;

  RAJA::kernel<NESTED_EXEC_POL3>(
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

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA mat-mult (RAJA-nested - POL4)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 
  
  //
  // This policy replaces the loop nest with a single CUDA kernel launch
  // (kernel body is the lambda loop body) where the row indices are 
  // assigned to thread blocks and the col indices are assigned to
  // threads within each block.
  // 
  // This is equivalent to launching a CUDA kernel with grid dimension N
  // and blocksize N; i.e., kernel<<<N, N>>> and defining row = blockIdx.x
  // and col = threadIdx.x in the kernel.
  //
  using NESTED_EXEC_POL4 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_block_exec,
          RAJA::statement::For<0, RAJA::cuda_thread_exec,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

  RAJA::kernel<NESTED_EXEC_POL4>(
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


//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA tiled mat-mult (RAJA-POL5)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy collapses the col and row loops into a single CUDA kernel
  // using two-dimensional CUDA thread blocks with x and y dimensions defined
  // by CUDA_BLOCK_SIZE arguments.
  //
  // When the matrix dimension N is an integer multiple of CUDA_BLOCK_SIZE, 
  // the CUDA grid and thread dimension kernel launch parameters will be the 
  // same as in this kernel and the one above.
  // 
  using NESTED_EXEC_POL5 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_threadblock_exec<CUDA_BLOCK_SIZE>,
          RAJA::statement::For<0, RAJA::cuda_threadblock_exec<CUDA_BLOCK_SIZE>,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

  RAJA::kernel<NESTED_EXEC_POL5>(
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



//----------------------------------------------------------------------------//

//
// The following example shows a raw CUDA version of the RAJA CUDA 
// matrix-multiplication kernel above. We include it for comparison.
//

//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA tiled mat-mult (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 

  // Define thread block dimensions
  dim3 blockdim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
  // Define grid dimensions to match the RAJA version above
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x), 
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

//printf("griddim = (%d,%d), blockdim = (%d,%d)\n", (int)griddim.x, (int)griddim.y, (int)blockdim.x, (int)blockdim.y);

  // Launch CUDA kernel defined near the top of this file.
  matMultKernel<<<griddim, blockdim>>>(N, C, A, B);

  cudaDeviceSynchronize();

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

#endif // if RAJA_ENABLE_CUDA


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
