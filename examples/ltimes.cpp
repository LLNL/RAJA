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
#include "RAJA/util/Timer.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

/*
 *  LTimes Example
 *
 *  Here is the reference implementation of the LTimes kernel:

    for (int m = 0; m < num_m; ++m) {
      for (int d = 0; d < num_d; ++d) {
        for (int g = 0; g < num_g; ++g) {
          for (int z = 0; z < num_z; ++z) {
            phi[m*num_g*num_z + g*num_z + z] +=
              L[m*num_d + d] * psi[d*num_g*num_z + g*num_z + z];
          }
        }
      }
    }
 
 *
 *  Note that RAJA Views and Layouts are used for multi-dimensional
 *  array data access in all variants.
 */

#define DEBUG_LTIMES 
//#undef DEBUG_LTIMES

using namespace RAJA;
using namespace RAJA::statement;

//
// Index value types for strongly-typed indices must be defined outside 
// function scope for RAJA CUDA variants to work.
//
// These types provide strongly-typed index values so if something is wrong
// in loop ordering and/or indexing, the code will likely not compile.
//
RAJA_INDEX_VALUE(IM, "IM");
RAJA_INDEX_VALUE(ID, "ID");
RAJA_INDEX_VALUE(IG, "IG");
RAJA_INDEX_VALUE(IZ, "IZ");

//
// View types for arrays.
//
// The Layout template parameter defines the indexing dimensionality, the
// linear index type (when mutli-dimensional indices are converted to linear),
// and which dimension is stride-1. The args that follow indicate the index
// types used and order to index into the View.
//

// phi[m, g, z]
using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

// psi[d, g, z]
using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

// L[m, d]
using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;


//
// Function to check results
//
void checkResult(PhiView& phi, LView& L, PsiView& Psi,
                 const Index_type num_m, 
                 const Index_type num_d,
                 const Index_type num_g,
                 const Index_type num_z);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  std::cout << "\n\nRAJA LTIMES example...\n\n";

//----------------------------------------------------------------------------//
// Define array dimensions, allocate arrays, define Layouts and Views, etc.

  const Index_type num_m = 25;
  const Index_type num_g = 48;
  const Index_type num_d = 80;
  const Index_type num_z = 64*1024;

  std::cout << "num_m = " << num_m << ", num_g = " << num_g << 
               ", num_d = " << num_d << ", num_z = " << num_z << "\n\n";

  const Index_type L_size   = num_m * num_d;
  const Index_type psi_size = num_d * num_g * num_z;
  const Index_type phi_size = num_m * num_g * num_z;

  std::vector<double> L_vec(num_m * num_d);
  std::vector<double> psi_vec(num_d * num_g * num_z);
  std::vector<double> phi_vec(num_m * num_g * num_z);

  double* L_data   = &L_vec[0];
  double* psi_data = &psi_vec[0];
  double* phi_data = &phi_vec[0];

  for (Index_type i = 0; i < L_size; ++i) {
    L_data[i] = i;
  }

  for (Index_type i = 0; i < psi_size; ++i) {
    psi_data[i] = 2*i;
  }

  LView L(L_data, 
          RAJA::make_permuted_layout( {{num_m, num_d}},
          RAJA::as_array<RAJA::Perm<0, 1> >::get() ) );
  PsiView psi(psi_data,
              RAJA::make_permuted_layout( {{num_d, num_g, num_z}},
              RAJA::as_array<RAJA::Perm<0, 1, 2> >::get() ) );
  PhiView phi(phi_data,
              RAJA::make_permuted_layout( {{num_m, num_g, num_z}}, \
              RAJA::as_array<RAJA::Perm<0, 1, 2> >::get() ) );


//----------------------------------------------------------------------------//

{
  std::cout << "\n Running C-version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  RAJA::Timer timer;
 
  timer.start(); 
  for (IM m(0); m < num_m; ++m) {
    for (ID d(0); d < num_d; ++d) {
      for (IG g(0); g < num_g; ++g) {
        for (IZ z(0); z < num_z; ++z) {
          phi(m, g, z) += L(m, d) * psi(d, g, z);
        }
      }
    }
  }
  timer.stop(); 

  std::cout << "  C-version of LTimes run time (sec.): " 
            << timer.elapsed() << std::endl;

#if defined(DEBUG_LTIMES) && 0
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}

//----------------------------------------------------------------------------//

{
  std::cout << "\n Running RAJA sequential version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  using EXECPOL = 
    RAJA::KernelPolicy<
       For<0, loop_exec,  // m       
         For<1, loop_exec,  // d
           For<2, loop_exec,  // g 
             For<3, simd_exec,  // z
               Lambda<0>
             >
           >
         >
       >
     >;

  auto segments = RAJA::make_tuple(TypedRangeSegment<IM>(0, num_m),
                                   TypedRangeSegment<ID>(0, num_d),
                                   TypedRangeSegment<IG>(0, num_g),
                                   TypedRangeSegment<IZ>(0, num_z));

  RAJA::Timer timer;

  timer.start();
  RAJA::kernel<EXECPOL>( segments,
    [=] (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );
  timer.stop();

  std::cout << "  RAJA sequential version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;

#if defined(DEBUG_LTIMES) && 0
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
{
  std::cout << "\n Running RAJA OpenMP version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  using EXECPOL =
    RAJA::KernelPolicy<
      For<0, omp_parallel_for_exec,  // m
        For<1, loop_exec,  // d 
          For<2, loop_exec,  // g 
            For<3, simd_exec,  // z 
              Lambda<0>
            >
          >
        >
      >
    >;

  auto segments = RAJA::make_tuple(TypedRangeSegment<IM>(0, num_m),
                                   TypedRangeSegment<ID>(0, num_d),
                                   TypedRangeSegment<IG>(0, num_g),
                                   TypedRangeSegment<IZ>(0, num_z));

  RAJA::Timer timer;

  timer.start();
  RAJA::kernel<EXECPOL>( segments,
    [=] (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );
  timer.stop();

  std::cout << "  RAJA OpenMP version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;

#if defined(DEBUG_LTIMES) && 0
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
{
  std::cout << "\n Running RAJA CUDA version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  double* dL_data   = nullptr;
  double* dpsi_data = nullptr;
  double* dphi_data = nullptr;

  cudaErrchk( cudaMalloc( (void**)&dL_data, L_size * sizeof(double) ) );
  cudaErrchk( cudaMemcpy( dL_data, L_data, L_size * sizeof(double),
                          cudaMemcpyHostToDevice ) );
  cudaErrchk( cudaMalloc( (void**)&dpsi_data, psi_size * sizeof(double) ) );
  cudaErrchk( cudaMemcpy( dpsi_data, psi_data, psi_size * sizeof(double),
                          cudaMemcpyHostToDevice ) );
  cudaErrchk( cudaMalloc( (void**)&dphi_data, phi_size * sizeof(double) ) );
  cudaErrchk( cudaMemcpy( dphi_data, phi_data, phi_size * sizeof(double),
                          cudaMemcpyHostToDevice ) );

  LView dL(dL_data,
           RAJA::make_permuted_layout( {{num_m, num_d}},
           RAJA::as_array<RAJA::Perm<0, 1> >::get() ) );
  PsiView dpsi(dpsi_data,
               RAJA::make_permuted_layout( {{num_d, num_g, num_z}},
               RAJA::as_array<RAJA::Perm<0, 1, 2> >::get() ) );
  PhiView dphi(dphi_data,
               RAJA::make_permuted_layout( {{num_m, num_g, num_z}}, \
               RAJA::as_array<RAJA::Perm<0, 1, 2> >::get() ) );

  using EXECPOL =
    RAJA::KernelPolicy<
      CudaKernelAsync<    
        For<0, cuda_block_exec,  // m
          For<2, cuda_block_exec,  // g
            For<3, cuda_thread_exec,  // z
              For<1, seq_exec,  // d
                Lambda<0>
              >
            >
          >
        >
      >         
    >;          
                 
  auto segments = RAJA::make_tuple(TypedRangeSegment<IM>(0, num_m),
                                   TypedRangeSegment<ID>(0, num_d),
                                   TypedRangeSegment<IG>(0, num_g),
                                   TypedRangeSegment<IZ>(0, num_z));

  RAJA::Timer timer;

  cudaErrchk( cudaDeviceSynchronize() );
  timer.start();
  RAJA::kernel<EXECPOL>( segments,
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z) {
       dphi(m, g, z) += dL(m, d) * dpsi(d, g, z);
    }
  );
  cudaErrchk( cudaDeviceSynchronize() );
  timer.stop();

  std::cout << "  RAJA CUDA version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;

  cudaErrchk( cudaMemcpy( phi_data, dphi_data, phi_size * sizeof(double),
                          cudaMemcpyDeviceToHost ) );

  cudaErrchk( cudaFree( dL_data ) );
  cudaErrchk( cudaFree( dpsi_data ) );
  cudaErrchk( cudaFree( dphi_data ) );

#if defined(DEBUG_LTIMES) && 0
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA) && 0
{
  std::cout << "\n Running RAJA CUDA + shmem version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  double* dL_data   = nullptr;
  double* dpsi_data = nullptr;
  double* dphi_data = nullptr;

  cudaErrchk( cudaMalloc( (void**)&dL_data, L_size * sizeof(double) ) );
  cudaErrchk( cudaMemcpy( dL_data, L_data, L_size * sizeof(double),
                          cudaMemcpyHostToDevice ) );
  cudaErrchk( cudaMalloc( (void**)&dpsi_data, psi_size * sizeof(double) ) );
  cudaErrchk( cudaMemcpy( dpsi_data, psi_data, psi_size * sizeof(double),
                          cudaMemcpyHostToDevice ) );
  cudaErrchk( cudaMalloc( (void**)&dphi_data, phi_size * sizeof(double) ) );
  cudaErrchk( cudaMemcpy( dphi_data, phi_data, phi_size * sizeof(double),
                          cudaMemcpyHostToDevice ) );

  // phi[m, g, z]
  using dPhiView = TypedView<double, Layout<3, Index_type, 0>, IM, IG, IZ>;

  // psi[d, g, z]
  using dPsiView = TypedView<double, Layout<3, Index_type, 0>, ID, IG, IZ>;

  // L[m, d]
  using dLView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;


  dLView dL(dL_data,
            RAJA::make_permuted_layout( {{num_m, num_d}},
            RAJA::as_array<RAJA::Perm<0, 1> >::get() ) );
  dPsiView dpsi(dpsi_data,
                RAJA::make_permuted_layout( {{num_d, num_g, num_z}},
                RAJA::as_array<RAJA::Perm<2, 1, 0> >::get() ) );
  dPhiView dphi(dphi_data,
                RAJA::make_permuted_layout( {{num_m, num_g, num_z}}, \
                RAJA::as_array<RAJA::Perm<2, 1, 0> >::get() ) );

  static const int tile_m = 48;
  static const int tile_d = 48;
  static const int tile_z = 64;

  auto segments = RAJA::make_tuple(TypedRangeSegment<IM>(0, num_m),
                                   TypedRangeSegment<ID>(0, num_d),
                                   TypedRangeSegment<IG>(0, num_g),
                                   TypedRangeSegment<IZ>(0, num_z));

  using shmem_L_T = ShmemTile<cuda_shmem, double, 
                              ArgList<1,0>, 
                              SizeList<tile_d, tile_m>, decltype(segments)>;
  shmem_L_T shmem_L;

  using shmem_psi_T = ShmemTile<cuda_shmem, double, 
                                ArgList<1,3>, 
                                SizeList<tile_d, tile_z>, decltype(segments)>;
  shmem_psi_T shmem_psi;

  using EXECPOL =
    RAJA::KernelPolicy<
      CudaKernelAsync<

        statement::Tile<0, statement::tile_fixed<tile_m>, seq_exec,  // m
          statement::Tile<1, statement::tile_fixed<tile_d>, seq_exec,  // d

            // Load L matrix for each tile
            SetShmemWindow<
              For<1, cuda_thread_exec,  // d 
                For<0, cuda_thread_exec,  // m
                  Lambda<1>
                >
              >
            >,  // SetShmemWindow
            CudaSyncThreads,
 
            // Distribute g, z across blocks
            For<2, cuda_block_exec, // g
              
              // Distribute chunks of z over blocks
              statement::Tile<3, statement::tile_fixed<tile_z>, cuda_block_exec, // z

                SetShmemWindow<

                  // Load Psi for g, z
                  For<3, cuda_thread_exec,  // z
                    For<1, cuda_thread_exec,  // d 
                      Lambda<2>
                    >
                  >, 
                  CudaSyncThreads,

                  // Compute phi for all m and this g, z tile 
                  // (using thread-local storage)
                  For<3, cuda_thread_exec,  // z
                    For<0, cuda_thread_exec,  // m
                      Thread<
                        Lambda<3>,
                        For<1, seq_exec,  // d
                          Lambda<4>
                        >,
                        Lambda<5> 
                      >  // Thread
                    >  // m
                  >, // z
                  CudaSyncThreads

                >  // SetShmemWindow

              >  // Tile z

            >  // g

          >  // Tile d
        >  // Tile m

      >  // CudaKernelAsync 
    >;  // Kernel Policy
                 
  RAJA::Timer timer;

  cudaErrchk( cudaDeviceSynchronize() );
  timer.start();
  RAJA::kernel_param<EXECPOL>( segments,
    RAJA::make_tuple( shmem_L,
                      shmem_psi,
                      0.0 ), // thread-private temp (last arg in lambdas)

    // Lambda<0> : Original single lambda implementation
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z, 
                     shmem_L_T&, shmem_psi_T&, double&) {
      dphi(m, g, z) += dL(m, d) * dpsi(d, g, z);
    },

    // Lambda<1> : Load L matrix into shmem
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z, 
                     shmem_L_T sh_L&, shmem_psi_T&, double&) {
      sh_L(d, m) = dL(m, d);
    },

    // Lambda<2> : Load slice of psi into shmem
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z, 
                     shmem_L_T&, shmem_psi_T& sh_psi, double&) {
      sh_psi(d, z) = dpsi(d, g, z);
    },

    // Lambda<3> : Load phi(m,g,z) into phi_local
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z, 
                     shmem_L_T&, shmem_psi_T&, double& phi_local) {
      phi_local = dphi(m, g, z);
    },

    // Lambda<4> : Compute phi_local
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z, 
                     shmem_L_T sh_L&, shmem_psi_T& sh_psi, double& phi_local) {
      phi_local += sh_L(d, m) * sh_psi(d, z);
    },

    // Lambda<5> : Store phi
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z, 
                     shmem_L_T&, shmem_psi_T&, double& phi_local) {
      dphi(m, g, z) = phi_local;
    }
  );
  cudaErrchk( cudaDeviceSynchronize() );
  timer.stop();

  std::cout << "  RAJA CUDA + shmem version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;

  cudaErrchk( cudaMemcpy( phi_data, dphi_data, phi_size * sizeof(double),
                          cudaMemcpyDeviceToHost ) );

  cudaErrchk( cudaFree( dL_data ) );
  cudaErrchk( cudaFree( dpsi_data ) );
  cudaErrchk( cudaFree( dphi_data ) );

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)
{
  std::cout << "\n Running RAJA CUDA + shmem version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  double* dL_data   = nullptr;
  double* dpsi_data = nullptr;
  double* dphi_data = nullptr;

  cudaErrchk( cudaMalloc( (void**)&dL_data, L_size * sizeof(double) ) );
  cudaErrchk( cudaMemcpy( dL_data, L_data, L_size * sizeof(double),
                          cudaMemcpyHostToDevice ) );
  cudaErrchk( cudaMalloc( (void**)&dpsi_data, psi_size * sizeof(double) ) );
  cudaErrchk( cudaMemcpy( dpsi_data, psi_data, psi_size * sizeof(double),
                          cudaMemcpyHostToDevice ) );
  cudaErrchk( cudaMalloc( (void**)&dphi_data, phi_size * sizeof(double) ) );
  cudaErrchk( cudaMemcpy( dphi_data, phi_data, phi_size * sizeof(double),
                          cudaMemcpyHostToDevice ) );

  // psi[d, g, z]
  using dPsiView = RAJA::TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi[m, g, z]
  using dPhiView = RAJA::TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;

  // L[m, d]
  using dLView = RAJA::TypedView<double, Layout<2, int, 1>, IM, ID>;

  // create views on data
  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  dLView L(
      dL_data,
      make_permuted_layout({num_m, num_d}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{2, 1, 0}};
  dPsiView psi(
      dpsi_data,
      make_permuted_layout({num_d, num_g, num_z}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{2, 1, 0}};
  dPhiView phi(
      dphi_data,
      make_permuted_layout({num_m, num_g, num_z}, phi_perm));

  static const int tile_m = 48;
  static const int tile_d = 48;
  static const int tile_z = 64;

      using Pol = RAJA::KernelPolicy<
               CudaKernelAsync<

                RAJA::statement::Tile<1, RAJA::statement::tile_fixed<tile_m>, seq_exec,
                  RAJA::statement::Tile<2, RAJA::statement::tile_fixed<tile_d>, seq_exec,

                      // First, load up L matrix for each block
                      SetShmemWindow<

                For<2, cuda_thread_exec, // d

                        For<1, cuda_thread_exec, Lambda<2>> //m

                >
                      >,

                      CudaSyncThreads,

                      // Distribute g, z across blocks
                      For<0, cuda_block_exec, // g

                      // distribute chunks of z over blocks
                      RAJA::statement::Tile<3, RAJA::statement::tile_fixed<tile_z>, cuda_block_exec,


                        SetShmemWindow<

                            // Load Psi for this g,z
                          For<3, cuda_thread_exec,  // z
                            For<2, cuda_thread_exec,
                                Lambda<3>
                            > // d
                          >,

                          CudaSyncThreads,

                            // Compute phi for all m's and this g,z
                          For<3, cuda_thread_exec,  // z
                            For<1, cuda_thread_exec, // m

                              // Compute phi(m,g,z) using thread-local storage
                              // and shared memory
                              Thread< // Thread count: tile_m*tile_z
                                // Load phi
                                Lambda<4>,

                                // Compute phi
                                For<2, seq_exec, Lambda<5>> ,  // d

                                // Store phi
                                Lambda<6>
                              >
                            > // m
                            >, // z
                            CudaSyncThreads
                          > // shmem
                        > // tile z
                      > //g
                    > // tile d
                  > // tile m
                > // kernel
              >; // policy


  auto segments = RAJA::make_tuple(
      TypedRangeSegment<IG, int>(0, num_g),
      TypedRangeSegment<IM, int>(0, num_m),
      TypedRangeSegment<ID, int>(0, num_d),
      TypedRangeSegment<IZ, int>(0, num_z));

  using shmem_L_t = ShmemTile<cuda_shmem, double, ArgList<2,1>, SizeList<tile_d, tile_m>, decltype(segments)>;
  shmem_L_t shmem_L;

  using shmem_psi_t = ShmemTile<cuda_shmem, double, ArgList<2,3>, SizeList<tile_d, tile_z>, decltype(segments)>;
  shmem_psi_t shmem_psi; 


  RAJA::Timer timer;

  cudaErrchk( cudaDeviceSynchronize() );
  timer.start();
  kernel_param<Pol>(

    segments,

    RAJA::make_tuple( shmem_L,
                      shmem_psi,
                      0.0), // thread private temp storage (last arg in lambdas)

    // Lambda<0>
     // Zero out phi
    [=] RAJA_HOST_DEVICE  (IG g, IM m, ID d, IZ z, shmem_L_t&, shmem_psi_t&, double&) {
      phi(m, g, z) = 0.0;
    },

    // Lambda<1>
    // Original single lambda implementation
    [=] RAJA_HOST_DEVICE  (IG g, IM m, ID d, IZ z, shmem_L_t&, shmem_psi_t&, double&) {
      phi(m, g, z) += L(m, d) * psi(d,g,z);
    },

    // Lambda<2>
    // load L matrix into shmem
    [=] RAJA_DEVICE  (IG g, IM m, ID d, IZ z, shmem_L_t& sh_L, shmem_psi_t&, double&) {
      sh_L(d, m) = L(m, d);
    },

    // Lambda<3>
    // load slice of psi into shared
    [=] RAJA_DEVICE  (IG g, IM m, ID d, IZ z, shmem_L_t&, shmem_psi_t& sh_psi, double&) {
      sh_psi(d,z) = psi(d,g,z);
    },

    // Lambda<4>
    // Load phi_m_g_z
    [=] RAJA_DEVICE  (IG g, IM m, ID d, IZ z, shmem_L_t&, shmem_psi_t&, double& phi_local) {
      phi_local = phi(m, g, z);
    },

    // Lambda<5>
    // Compute phi_m_g_z
    [=] RAJA_DEVICE  (IG g, IM m, ID d, IZ z, shmem_L_t& sh_L, shmem_psi_t& sh_psi, double& phi_local) {
      phi_local += sh_L(d, m) * sh_psi(d,z);
    },

    // Lambda<6>
    // Store phi_m_g_z
    [=] RAJA_DEVICE  (IG g, IM m, ID d, IZ z, shmem_L_t&, shmem_psi_t&, double& phi_local) {
      phi(m, g, z) = phi_local;
    }

  );
  cudaDeviceSynchronize();
  timer.stop();

  std::cout << "  RAJA CUDA + shmem version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;

  cudaErrchk( cudaMemcpy( phi_data, dphi_data, phi_size * sizeof(double),
                          cudaMemcpyDeviceToHost ) );

  cudaErrchk( cudaFree( dL_data ) );
  cudaErrchk( cudaFree( dpsi_data ) );
  cudaErrchk( cudaFree( dphi_data ) );

  L.set_data(&L_data[0]); 
  psi.set_data(&psi_data[0]); 
  phi.set_data(&phi_data[0]); 

#if defined(DEBUG_LTIMES)
  size_t nerrors = 0;
  double total_error = 0.0;

  for (IM m(0); m < num_m; ++m) {
    for (IG g(0); g < num_g; ++g) {
      for (IZ z(0); z < num_z; ++z) {
        double total = 0.0;
        for (ID d(0); d < num_d; ++d) {
          double val = L(m, d) * psi(d, g, z);
          total += val;
        }
        if (std::abs(total-phi(m, g, z)) > 1e-9) {
          ++nerrors;
        }
        total_error += std::abs(total-phi(m, g, z));
//std::cout << "(" << m << ", " << g << ", " << z << ") tot_error = "
//          << total_error << std::endl;
      }
    }
  }

  if ( nerrors == 0 ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL : " << nerrors << " errors!\n";
  }

//  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(PhiView& phi, LView& L, PsiView& psi,
                 const Index_type num_m, 
                 const Index_type num_d,
                 const Index_type num_g,
                 const Index_type num_z)
{
  size_t nerrors = 0;
  double total_error = 0.0;

  for (IM m(0); m < num_m; ++m) {
    for (IG g(0); g < num_g; ++g) {
      for (IZ z(0); z < num_z; ++z) {
        double total = 0.0;
        for (ID d(0); d < num_d; ++d) {
          double val = L(m, d) * psi(d, g, z);
          total += val;
        }
        if (std::abs(total-phi(m, g, z)) > 1e-9) {
          ++nerrors;
        }
        total_error += std::abs(total-phi(m, g, z));
      }
    }
  }

  if ( nerrors == 0 ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL : " << nerrors << " errors!\n";
  }
}
