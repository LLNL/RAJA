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
 *  RAJA Teams Example
 *
 */

/*
  Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
#endif

#if defined(RAJA_ENABLE_HIP)
#define HIP_BLOCK_SIZE 16
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
*/
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
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

struct TeamIdx{int tx, ty, tz;};

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

#define RAJA_FORALL_TEAMS(i, use_device, N, TEAMS, ...)  \
   RAJA_ForallWrap_Teams(                     \
      N,                                      \
      use_device,                             \
      [=] RAJA_DEVICE (int i) { __VA_ARGS__ }, \
      [=](int i) { __VA_ARGS__ },              \
      TEAMS.X,                                 \
      TEAMS.Y,                                 \
      TEAMS.Z)

template <typename DBODY, typename HBODY>
inline void RAJA_ForallWrap_Teams(const int N,
                                   bool device,
                                   DBODY &&d_body,
                                   HBODY &&h_body,
                                   const int X,
                                   const int Y,
                                   const int Z)
{

  if(device)
  {
    
    using RAJA::statement::For;
    using RAJA::statement::Lambda;
    using RAJA::Segs;
    
    RAJA::kernel<RAJA::KernelPolicy<
      RAJA::statement::CudaKernelAsync<
        For<0, RAJA::cuda_block_x_direct,
        For<1, RAJA::cuda_thread_x_direct,
        For<2, RAJA::cuda_thread_y_direct,
         For<3, RAJA::cuda_thread_z_direct,
         Lambda<0, Segs<0>>>>>>>>>
    (RAJA::make_tuple
     (RAJA::RangeSegment(0,N),
      RAJA::RangeSegment(0,X),
      RAJA::RangeSegment(0,Y),
      RAJA::RangeSegment(0,Z)),
     d_body);
  }else
  {
    RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, N), h_body);
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
template<size_t N, size_t XDIM,
         size_t YDIM, size_t ZDIM>
struct PrivateMemory
{
  const int X{XDIM};
  const int Y{YDIM};
  const int Z{ZDIM};

#if defined(__CUDA_ARCH__)
  double Array[N];
#else
  double Array[N*XDIM*YDIM*ZDIM];
#endif

  RAJA_HOST_DEVICE
  double &operator()(int i, TeamIdx teamIdx) 
  {
#if defined(__CUDA_ARCH__)  
    return Array[i];
#else
    int offset = N*teamIdx.tx + N*XDIM*teamIdx.ty + N*XDIM*YDIM*teamIdx.tz;
    return Array[i + offset];
#endif
  }

};

//Struct with general Team info
template<int Nx, int Ny, int Nz>
struct Teams
{
  const int X{Nx};
  const int Y{Nx};
  const int Z{Nx};
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


  int Nteams = 2; 
  using Team_t = Teams<5,1,1>;

  //Loop through GPU and CPU kernels
  for(int j=0; j<2; ++j) { 

    if(use_device[j]) {
      printf("RAJA Teams running on GPU  \n");
    }else{
      printf("RAJA Teams running on CPU  \n");
    }

  printf("Displaying team member exclusive values \n");
  RAJA_FORALL_TEAMS(i, use_device[j], Nteams, Team_t{},
  {
    
    Team_t myTeam;
    Team_t::PrivateMem<1> p_a;
    TEAM_SHARED double s_a[5]; 
    
    TEAM_LOOP_1D(myTeam.X,
    {
      p_a(0, teamIdx) = teamIdx.tx;
      s_a[teamIdx.tx] = 1; 
    });

    Team_t::TeamSync();

    TEAM_LOOP_1D(myTeam.X,
    {
     printf("pa_[%d] = %f \n", teamIdx.tx, p_a(0, teamIdx));
    });

    TEAM_LOOP_1D(1,
    {
     double sum(0); 
     for(int i=0; i<5; ++i) {
       sum += s_a[i]; 
     }

     printf("Shared memory sum %f  \n", sum);
   });

  });

   cudaDeviceSynchronize();
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

#if defined(RAJA_ENABLE_HIP)

  std::cout << "\n Running HIP mat-mult with multiple lambdas (RAJA-POL8)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  hipErrchk(hipMemcpy( d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice ));

  using EXEC_POL8 =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::For<1, RAJA::hip_block_x_loop,    // row
          RAJA::statement::For<0, RAJA::hip_thread_x_loop, // col
            RAJA::statement::Lambda<0, RAJA::Params<0>>,   // dot = 0.0
            RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1> // dot += ...
            >,
            RAJA::statement::Lambda<2,
              RAJA::Segs<0,1>,
              RAJA::Params<0>>   // set C = ...
          >
        >
      >
    >;

  RAJA::kernel_param<EXEC_POL8>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] RAJA_DEVICE (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += d_Aview(row, k) * d_Bview(k, col);
    },

    // lambda 2
    [=] RAJA_DEVICE (int col, int row, double& dot) {
       d_Cview(row, col) = dot;
    }

  );

  hipErrchk(hipMemcpy( C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost ));
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


  //----------------------------------------------------------------------------//

  std::cout << "\n Running HIP mat-mult with multiple lambdas - lambda args in statements (RAJA-POL9)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  hipErrchk(hipMemcpy( d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice ));

  using EXEC_POL9b =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::Tile<1, RAJA::tile_fixed<HIP_BLOCK_SIZE>, RAJA::hip_block_y_loop,
          RAJA::statement::Tile<0, RAJA::tile_fixed<HIP_BLOCK_SIZE>, RAJA::hip_block_x_loop,
            RAJA::statement::For<1, RAJA::hip_thread_y_loop, // row
              RAJA::statement::For<0, RAJA::hip_thread_x_loop, // col
                RAJA::statement::Lambda<0, Params<0>>,  // dot = 0.0
                RAJA::statement::For<2, RAJA::seq_exec,
                  RAJA::statement::Lambda<1, Segs<0,1,2>, Params<0>> // dot += ...
                >,
                  RAJA::statement::Lambda<2, Segs<0,1>, Params<0>>   // set C = ...
              >
            >
          >
        >
      >
    >;

  RAJA::kernel_param<EXEC_POL9b>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] RAJA_DEVICE (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += d_Aview(row, k) * d_Bview(k, col);
    },

    // lambda 2
    [=] RAJA_DEVICE (int col, int row, double& dot) {
       d_Cview(row, col) = dot;
    }

  );

  hipErrchk(hipMemcpy( C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost ));
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running HIP tiled mat-mult (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  hipErrchk(hipMemcpy( d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice ));

  // Define thread block dimensions
  dim3 blockdim(HIP_BLOCK_SIZE, HIP_BLOCK_SIZE);
  // Define grid dimensions to match the RAJA version above
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

//printf("griddim = (%d,%d), blockdim = (%d,%d)\n", (int)griddim.x, (int)griddim.y, (int)blockdim.x, (int)blockdim.y);

  // Launch HIP kernel defined near the top of this file.
  hipLaunchKernelGGL((matMultKernel), dim3(griddim), dim3(blockdim), 0, 0, N, d_C, d_A, d_B);

  hipDeviceSynchronize();

  hipErrchk(hipMemcpy( C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost ));
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

  memoryManager::deallocate_gpu(d_A);
  memoryManager::deallocate_gpu(d_B);
  memoryManager::deallocate_gpu(d_C);
#endif // if RAJA_ENABLE_HIP

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
