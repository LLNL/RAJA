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
 *  Matrix Multiplication Example with Complex Policies
 *
 *  This is the same matrix multiplication example as in the file
 *  ex4-matrix-multiply.cpp but shows how to use more complex RAJA 'kernel'
 *  policies.
 *
 *  RAJA features shown:
 *    - Index range segment
 *    - View abstraction
 *    - 'RAJA::kernel' loop execution with complex policies; e.g.,
 *       - Multiple lambdas
 *       - Thread-local storage
 *       - Shared memory tiling windows
 *
 * For CPU execution RAJA shared memory windows can be used to improve 
 * performance via cache blocking. For CUDA GPU execution, CUDA shared 
 * memory can be used to improve performance. 
 *
 *
 * See examples in ex4-matrix-multiply.cpp for basic RAJA nested loop 
 * constructs via RAJA::kernel abstractions.
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Define shared memory tile size (TILE_SIZE x TILE_SIZE)
*/
#define TILE_SIZE 16


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

#define OFFSET(r, c) (c + N * r)

/*
  Define CUDA shared memory matrix multiplication kernels to compare to RAJA
*/
#if defined(RAJA_ENABLE_CUDA)

//
// CUDA kernel for matrix multiplication where matrices are assumed to be
// square with dimensions equal to integer multiple of TILE_SIZE.
//
__global__ void matMultShmemKernel(int N, double* C, double* A, double* B)
{
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // set pointer to submatrix of C; each thread block computes on one
  // submatrix of C
  double* Csub = &C[TILE_SIZE*OFFSET(block_row, block_col)];

  // thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Shared memory arrays to store A and B submatrices.
  __shared__ double As[TILE_SIZE][TILE_SIZE];
  __shared__ double Bs[TILE_SIZE][TILE_SIZE];

  // each thread computes one element of Csub, accumulating result into dot
  double dot = 0.0;

  // loop over submatrices of A, B to compute Csub, multiplying each 
  // pair of submatrices together and accumulate result
  for (int m = 0; m < RAJA_DIVIDE_CEILING_INT(N,TILE_SIZE); ++m) {

    // set pointer to submatrices of A, B
    double* Asub = &A[TILE_SIZE*OFFSET(block_row, m)];
    double* Bsub = &B[TILE_SIZE*OFFSET(m, block_col)];

    // Load Asub, Bsub into shared memory, each thread loads on element 
    // of each submatrix
    As[row][col] = Asub[OFFSET(row, col)];
    Bs[row][col] = Bsub[OFFSET(row, col)];

    // sync threads so sumatrixes are loaded before multiplying them
    __syncthreads();

    // multiply submatrices and accumulate result
    for (int k = 0; k < TILE_SIZE; ++k) {
      dot += As[row][k] * Bs[k][col];
    }

    // sync threads so preceding computation is done before loading two
    // new submatrices
    __syncthreads();
  }

  // write element of C
  Csub[OFFSET(row, col)] = dot;
}

//
// CUDA kernel for matrix multiplication where matrices are assumed to be
// square but dimensions do not have to equal integer multiple of TILE_SIZE.
//
__global__ void matMultShmemKernel_any(int N, double* C, double* A, double* B)
{
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // set pointer to submatrix of C; each thread block computes on one
  // submatrix of C
  double* Csub = &C[TILE_SIZE*OFFSET(block_row, block_col)];

  // thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Shared memory arrays to store A and B submatrices.
  __shared__ double As[TILE_SIZE][TILE_SIZE];
  __shared__ double Bs[TILE_SIZE][TILE_SIZE];

  // each thread computes one element of Csub, accumulating result into dot
  double dot = 0.0;

  // loop over submatrices of A, B to compute Csub, multiplying each 
  // pair of submatrices together and accumulate result
  for (int m = 0; m < RAJA_DIVIDE_CEILING_INT(N,TILE_SIZE); ++m) {

    // set pointer to submatrices of A, B
    double* Asub = &A[TILE_SIZE*OFFSET(block_row, m)];
    double* Bsub = &B[TILE_SIZE*OFFSET(m, block_col)];

    // Load Asub, Bsub into shared memory, each thread loads on element 
    // of each submatrix. Note: we set invalid submatrix entries to zero.
    if ( (m*TILE_SIZE + col < N) && (block_row*TILE_SIZE + row < N) ) {
      As[row][col] = Asub[OFFSET(row, col)];
    } else {
      As[row][col] = 0.0;
    }
    if ( (m*TILE_SIZE + row < N) && (block_col*TILE_SIZE + col < N) ) {
      Bs[row][col] = Bsub[OFFSET(row, col)];
    } else {
      Bs[row][col] = 0.0;
    }

    // sync threads so sumatrixes are loaded before multiplying them
    __syncthreads();

    // multiply submatrices and accumulate result
    for (int k = 0; k < TILE_SIZE; ++k) {
      dot += As[row][k] * Bs[k][col];
    }

    // sync threads so preceding computation is done before loading two
    // new submatrices
    __syncthreads();
  }

  // write element of C if indices are in range
  if ( (block_row*TILE_SIZE + row < N) && (block_col*TILE_SIZE + col < N) ) {
    Csub[OFFSET(row, col)] = dot;
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

  std::cout << "\n\nRAJA matrix multiplication complex policy example...\n";

//
// Define num rows/cols in matrix
//
  const int N = 1000;
//const int N = TILE_SIZE * TILE_SIZE;

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
// row, column, and dot product indices
//
  RAJA::RangeSegment row_range(0, N);
  RAJA::RangeSegment col_range(0, N);
  RAJA::RangeSegment dot_range(0, N);

//----------------------------------------------------------------------------//

  //
  // We start by implementing the matrix-multiplication example in the
  // file ex4-matrix-multiply.cpp using multipl lambdas. This helps lay 
  // the foundation for more complex kernels in which we introduce 
  // shared memory.
  // 

#if 0 // WORK-IN-PROGRESS-1
  std::cout << "\n Running sequential mat-mult multiple lambdas (RAJA-POL1)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy executes the col, row and k (inner dot product) loops 
  // sequentially using a triply-nested loop execution policy and three 
  // lambda expressions to: initialize the dot product variable, define
  // the 'k' inner loop row-col dot product body, and to store the 
  // computed row-col dot product in the proper location in the result matrix.
  //
  // Note that we also pass the scalar dot product variable into each lambda
  // via a single value tuple parameter. This enables the same variable to be
  // by all three lambdas.
  //
  using NESTED_EXEC_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Lambda<0>,  // dot = 0.0
          RAJA::statement::For<2, RAJA::seq_exec,
            RAJA::statement::Lambda<1> // inner loop: dot += ...
          >,
          RAJA::statement::Lambda<2>   // set C(row, col) = dot
        >
      >
    >;

  RAJA::kernel_param<NESTED_EXEC_POL1>(
                RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot = 0.0;
    },

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       Cview(row, col) = dot;
    }

  );
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // WORK-IN-PROGRESS-1

#if 0 // WORK-IN-PROGRESS-2
  Build on previous kernel by adding tiled sequential shared memory (cache blocking) 

  std::cout << "\n Running sequential mat-mult (RAJA-tiled-shmem)...\n";
    
  std::memset(C, 0, N*N * sizeof(double));

  using NESTED_EXEC_POL2 = 

  RAJA::kernel_param<NESTED_EXEC_POL2>(
                RAJA::make_tuple(col_range, row_range, dot_range),

  );
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // WORK-IN-PROGRESS-2 
  
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
//  Add examples showing OpenMP shared memory analogue of previous examples....
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA tiled mat-mult (RAJA-POL3)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy collapses the col and row loops into a single CUDA kernel
  // using two-dimensional CUDA thread blocks with x and y dimensions defined
  // by the TILE_SIZE arguments.
  //
  // Note also that we use three lambdas here and implement the inner
  // dot product loop using RAJA For loop construct.
  //
  using NESTED_EXEC_POL3 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_threadblock_exec<TILE_SIZE>,
          RAJA::statement::For<0, RAJA::cuda_threadblock_exec<TILE_SIZE>,
            RAJA::statement::Lambda<0>,  // dot = 0.0
            RAJA::statement::For<2, RAJA::seq_exec,
              RAJA::statement::Lambda<1> // dot += ...
            >,
            RAJA::statement::Lambda<2>   // set C entry
          >
        >
      >
    >;

  RAJA::kernel_param<NESTED_EXEC_POL3>(
                RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot = 0.0;
    },

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       Cview(row, col) = dot;
    }

  );
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//
//  Add example showing CUDA shared memory analogue of previous examples....

#if 0 // WORK-IN-PROGRESS-3
  std::cout << "\n Running CUDA tiled mat-mult (RAJA-tiled-shmem)...\n";

  std::memset(C, 0, N*N * sizeof(double)); 
  
  //
  // This policy collapses the col and row loops into a single CUDA kernel
  // using two-dimensional CUDA thread blocks with x and y dimensions defined
  // by the TILE_SIZE arguments.
  // 
  using NESTED_EXEC_POL2 =
    RAJA::nested::Policy< 
      RAJA::nested::CudaKernel<
        RAJA::nested::Tile<1, RAJA::nested::tile_fixed<TILE_SIZE>, RAJA::cuda_block_exec,
          RAJA::nested::Tile<0, RAJA::nested::tile_fixed<TILE_SIZE>, RAJA::cuda_block_exec,

            RAJA::nested::For<1, RAJA::cuda_thread_exec,
              RAJA::nested::For<0, RAJA::cuda_thread_exec,
                RAJA::nested::Lambda<2>    // dot = 0.0
              >
            >,

            RAJA::nested::Tile<2, RAJA::nested::tile_fixed<TILE_SIZE>, RAJA::seq_exec,

              RAJA::nested::SetShmemWindow<

                RAJA::nested::For<2, RAJA::cuda_thread_exec,
                  RAJA::nested::For<1, RAJA::cuda_thread_exec,
                    RAJA::nested::Lambda<0>
                  >
                >,

                RAJA::nested::For<0, RAJA::cuda_thread_exec,
                  RAJA::nested::For<2, RAJA::cuda_thread_exec,
                    RAJA::nested::Lambda<1>
                  >
                >,

                RAJA::nested::CudaSyncThreads, 

                RAJA::nested::For<1, RAJA::cuda_thread_exec,
                  RAJA::nested::For<0, RAJA::cuda_thread_exec,
                    RAJA::nested::For<2, RAJA::seq_exec,
                      RAJA::nested::Lambda<2> 
                    >
                  > 
                >,

              >,

            >, 

            RAJA::nested::CudaSyncThreads, 

            RAJA::nested::For<1, RAJA::cuda_thread_exec,
              RAJA::nested::For<0, RAJA::cuda_thread_exec,
                RAJA::nested::Lambda<3>
              >
            >

          >  // Tile 0

        >  // Tile 1
       
      > // CudaKernel
   >;

  RAJA::RangeSegment dot_range(0, N);
  auto segments = RAJA::make_tuple(col_range, row_range, dot_range);

  ShmemTile<cuda_shmem, double, RAJA::nested::ArgList<0,2>, RAJA::SizeList<TILE_SIZE, TILE_SIZE>, decltype(segments)> shmem_A;

  ShmemTile<cuda_shmem, double, RAJA::nested::ArgList<2,1>, RAJA::SizeList<TILE_SIZE, TILE_SIZE>, decltype(segments)> shmem_B;

  RAJA::nested::forall_param<NESTED_EXEC_POL2>(
                segments,

    RAJA::tuple<double>{0.0}, 

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       shmem_A(row, k) = A(row, k);
    },

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       shmem_B(k, col) = B(k, col);
    },

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot = 0.0;
    },

    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },
         
    [=] RAJA_DEVICE (int col, int row, int k, double& dot) { 
       Cview(row, col) = dot;
    }

  );
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

#endif // WORK-IN-PROGRESS

//----------------------------------------------------------------------------//

  //
  // For comparison, the following examples show raw CUDA implementations of
  // the RAJA CUDA shared memory kernel above. 
  // 
  // Note: The CUDA kernels are defined near the top of this file.
  // 

// Define thread block dimensions
  dim3 blockdim(TILE_SIZE, TILE_SIZE);
// Define grid dimensions to match the RAJA version above
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

//printf("griddim = (%d,%d), blockdim = (%d,%d)\n", (int)griddim.x, (int)griddim.y, (int)blockdim.x, (int)blockdim.y);

//
// NOTE: we choose the CUDA kernel we run based on matrix dimension.
//
if (N % TILE_SIZE == 0) {

  std::cout << "\n Running CUDA tiled mat-mult with shared memory (no RAJA)\n";
  std::cout << " NOTE: variant assumes N = n * TILE_SIZE...\n";

  std::memset(C, 0, N*N * sizeof(double));

  matMultShmemKernel<<<griddim, blockdim>>>(N, C, A, B);

  cudaDeviceSynchronize();

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

} else {  // N is not integer multiple of tile size

  std::cout << "\n Running CUDA tiled mat-mult with shared memory (no RAJA)\n";

  std::memset(C, 0, N*N * sizeof(double));

  // Launch CUDA kernel defined near the top of this file.
  matMultShmemKernel_any<<<griddim, blockdim>>>(N, C, A, B);

  cudaDeviceSynchronize();

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

}

#endif // if defined(RAJA_ENABLE_CUDA) 

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
