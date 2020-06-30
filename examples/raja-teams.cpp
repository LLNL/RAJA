//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  RAJA Teams Example: 
 *  Matrix-matrix multiplication with shared memory
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
// _matmult_macros_start
#define A(r, c) A[c + N * r]
#define B(r, c) B[c + N * r]
#define C(r, c) C[c + N * r]
// _matmult_macros_end

/*
  Define CUDA matrix multiplication kernel for comparison to RAJA version
  Adapted from CUDA programming guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
*/
#if defined(RAJA_ENABLE_CUDA)
__global__ void matMultKernel(int N, double* C, double* A, double* B)
{
  // Block row and column
  const int by = blockIdx.y;
  const int bx = blockIdx.x;

   // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  double Cvalue(0.0);

  // Thread row and column within local Csub
  const int ty = threadIdx.y; //local row
  const int tx = threadIdx.x; //local column

  const int row = by * CUDA_BLOCK_SIZE + ty;  // Matrix row index
  const int col = bx * CUDA_BLOCK_SIZE + tx;  // Matrix column index

  // Shared memory used to store Asub and Bsub respectively
  __shared__ double As[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
  __shared__ double Bs[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (N / CUDA_BLOCK_SIZE); ++m) {

    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[ty][tx] = A[row*N + m*CUDA_BLOCK_SIZE + tx];
    Bs[ty][tx] = B[(m*CUDA_BLOCK_SIZE + ty)*N + col];

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();

    // Multiply Asub and Bsub together
    for (int e = 0; e < CUDA_BLOCK_SIZE; ++e)
      Cvalue += As[ty][e] * Bs[e][tx];

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write Csub to device memory
  // Each thread writes one element
  C[col + N*row] = Cvalue;

}
#endif

struct TeamIdx{int x, y, z;};

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


bool use_device = true;

#define RAJA_FORALL_TEAMS(bx, by, use_device, Nx, Ny, TEAMS, ...) \
   RAJA_ForallWrap_Teams(                     \
      Nx,                                     \
      Ny,                                     \
      use_device,                             \
      [=] RAJA_DEVICE (int bx, int by) { __VA_ARGS__ }, \
      [=](int bx, int by) { __VA_ARGS__ },    \
      TEAMS.X,                                 \
      TEAMS.Y,                                 \
      TEAMS.Z)

template <typename DBODY, typename HBODY>
inline void RAJA_ForallWrap_Teams(int Nx,
                                  int Ny,
                                   bool device,
                                   DBODY &&d_body,
                                   HBODY &&h_body,
                                   const int X,
                                   const int Y,
                                   const int Z)
{
  using RAJA::statement::For;
  using RAJA::statement::Lambda;
  using RAJA::Segs;

  if(device)
  {
    RAJA::kernel<RAJA::KernelPolicy<
      RAJA::statement::CudaKernelAsync<
        For<0, RAJA::cuda_block_x_direct,
        For<1, RAJA::cuda_block_y_direct,
        For<2, RAJA::cuda_thread_x_direct,
        For<3, RAJA::cuda_thread_y_direct,
         For<4, RAJA::cuda_thread_z_direct,
             Lambda<0, Segs<0, 1>>>>>>>>>>
    (RAJA::make_tuple
     (RAJA::RangeSegment(0,Nx),
      RAJA::RangeSegment(0,Ny),
      RAJA::RangeSegment(0,X),
      RAJA::RangeSegment(0,Y),
      RAJA::RangeSegment(0,Z)),
     d_body);
  }else
  {
    //RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, N), h_body);
    RAJA::kernel<RAJA::KernelPolicy<
      For<0, RAJA::loop_exec,
        For<1, RAJA::loop_exec, 
          Lambda<0, Segs<0, 1>>>>>>
      (RAJA::make_tuple(RAJA::RangeSegment(0, Nx),
                        RAJA::RangeSegment(0, Ny)), 
       h_body);
  }

}

#if defined(__CUDA_ARCH__)
#define INNER_LAMBDA [=] __device__

#define TEAM_LOOP_3D(Nx, Ny, Nz, ...)                          \
  for (int tz = threadIdx.z; tz < Nz; tz += blockDim.z)        \
    for (int ty = threadIdx.y; ty < Ny; ty += blockDim.y)      \
      for (int tx = threadIdx.x; tx < Nx; tx += blockDim.x)    \
        {   TeamIdx teamIdx{tx, ty, tz};                       \
            __VA_ARGS__                                        \
              }

#define TEAM_LOOP_2D(Nx, Ny, ...)                             \
  for (int tz = threadIdx.z; tz < 1; tz += blockDim.z)        \
    for (int ty = threadIdx.y; ty < Ny; ty += blockDim.y)     \
      for (int tx = threadIdx.x; tx < Nx; tx += blockDim.x)    \
        {  TeamIdx teamIdx{tx, ty, tz};                         \
            __VA_ARGS__                                        \
              }

#define TEAM_LOOP_1D(Nx, ...)                                 \
  for (int tz = threadIdx.z; tz < 1; tz += blockDim.z)        \
    for (int ty = threadIdx.y; ty < 1; ty += blockDim.y)     \
      for (int tx = threadIdx.x; tx < Nx; tx += blockDim.x) \
        {   TeamIdx teamIdx{tx, ty, tz};                       \
            __VA_ARGS__                                        \
              }

#else
#define INNER_LAMBDA [=]

#define TEAM_LOOP_3D(Nx, Ny, Nz, ...)           \
  for (int tz = 0; tz < Nz; tz++)                \
    for (int ty = 0; ty < Ny; ty++)             \
      for (int tx = 0; tx < Nx; tx++)          \
        {   TeamIdx teamIdx{tx, ty, tz};         \
            __VA_ARGS__                           \
              }

#define TEAM_LOOP_2D(Nx, Ny, ...)        \
  for (int tz = 0; tz < 1; tz++)         \
    for (int ty = 0; ty < Ny; ty++)     \
      for (int tx = 0; tx < Nx; tx++)  \
        {   TeamIdx teamIdx{tx, ty, tz};   \
            __VA_ARGS__                   \
              }

#define TEAM_LOOP_1D(Nx, ...)          \
  for (int tz = 0; tz < 1; tz++)        \
    for (int ty = 0; ty < 1; ty++)     \
      for (int tx = 0; tx < Nx; tx++) \
        {   TeamIdx teamIdx{tx, ty, tz};  \
            __VA_ARGS__                  \
              }
#endif



#if defined(__CUDA_ARCH__)
#define TEAM_SHARED __shared__
#define TEAM_SYNC __syncthreads();
#else
#define TEAM_SHARED
#define TEAM_SYNC
#endif


//Sketch of member private memory
template<size_t N, size_t XDIM_,
         size_t YDIM_, size_t ZDIM_>
struct PrivateMemory
{
  const int XDim{XDIM_};
  const int YDim{YDIM_};
  const int ZDim{ZDIM_};

#if defined(__CUDA_ARCH__)
  double Array[N];
#else
  double Array[N*XDIM_*YDIM_*ZDIM_];
#endif

  RAJA_HOST_DEVICE
  double &operator()(int i, TeamIdx teamIdx)
  {
#if defined(__CUDA_ARCH__)
    return Array[i];
#else
    int offset = N*teamIdx.x + N*XDim*teamIdx.y + N*XDim*YDim*teamIdx.z;
    return Array[i + offset];
#endif
  }

};

//Struct with general Team info
template<int Nx, int Ny, int Nz>
struct Teams
{
  const int X{Nx};
  const int Y{Ny};
  const int Z{Nz};
  template<int N>
  using PrivateMem = PrivateMemory<N, Nx, Ny,Nz>;

  RAJA_HOST_DEVICE
  static void TeamSync() { TEAM_SYNC; }
};



int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA matrix multiplication example...\n";

//
// Define num rows/cols in matrix
//
  const int NBlocks = 4;
  const int N = CUDA_BLOCK_SIZE*NBlocks;

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

  // _matmult_cstyle_start
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += A(row, k) * B(k, col);
      }
      C(row, col) = dot;

    }
  }
  // _matmult_cstyle_end

  checkResult<double>(C, N);
//printResult<double>(C, N);

//----------------------------------------------------------------------------//

  bool use_device[2]={true, false};


  const int NxTeams = NBlocks;
  const int NThreads = CUDA_BLOCK_SIZE; 
  using Team_t = Teams<NThreads,NThreads,1>;

  //Loop through GPU and CPU kernels
  for(int j=0; j<2; ++j) {

    if(use_device[j]) {
      printf("\n (GPU) RAJA Teams matrix multiplication... \n");
    }else{
      printf("\n (CPU) RAJA Teams matrix multiplication... \n");
    }

  RAJA_FORALL_TEAMS(bx, by, use_device[j],
                    NBlocks, NBlocks, Team_t{},
  {
    //Thread/Team member private memory 
    Team_t myTeam; 
    Team_t::PrivateMem<1> cValue; 

    //Team Shared memory
    TEAM_SHARED double As[NThreads][NThreads];
    TEAM_SHARED double Bs[NThreads][NThreads];
    
    TEAM_LOOP_2D(NThreads, NThreads,{
        cValue(0, teamIdx) = 0.0; 
    }); 

    //Slide accross matrix 
    for (int m = 0; m < (N / NThreads); ++m) {

      TEAM_LOOP_2D(NThreads, NThreads, {

          const int tx = teamIdx.x; 
          const int ty = teamIdx.y;
          const int row = by * NThreads + ty;  // Matrix row index
          const int col = bx * NThreads + tx;  // Matrix column index
          
          As[ty][tx] = A[row*N + m*NThreads + tx];
          Bs[ty][tx] = B[(m*NThreads + ty)*N + col];          
        }); 

      TEAM_SYNC; 
      
      TEAM_LOOP_2D(NThreads, NThreads, {
          for(int e=0; e<NThreads; ++e){
            cValue(0, teamIdx) += As[ty][e] * Bs[e][tx]; 
          }
        }); 
      
      TEAM_SYNC; 

    }//slide across matrix 

    TEAM_LOOP_2D(NThreads, NThreads, {
        
        const int tx = teamIdx.x; 
        const int ty = teamIdx.y;
        const int row = by * NThreads + ty;  // Matrix row index
        const int col = bx * NThreads + tx;  // Matrix column index
        C[col + N*row] = cValue(0, teamIdx);
      });
    
  });

   cudaDeviceSynchronize();
   checkResult<double>(C, N);
  }

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

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

  checkResult<double>(C, N);
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
