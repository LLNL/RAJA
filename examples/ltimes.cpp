//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <vector>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#include <hip/hip_runtime.h>
#endif


/*
 *  LTimes Example
 *
 *  This example illustrates various ways to implement the LTimes kernel
 *  using RAJA. The reference implementation of the kernel is the first
 *  version of the kernel below.
 *
 *  Note that RAJA Views and Layouts are used for multi-dimensional
 *  array data access in all other variants.
 *
 *  RAJA features shown include:
 *    - Strongly-typed indices and ranges
 *    - View and Layout abstractions
 *    - Use of 'RAJA::kernel' abstractions for nested loops, including
 *      loop-level ordering changes, use of GPU shared memory, other
 *      RAJA 'statement' concepts
 *
 *  Note that calls to the checkResult() method after each variant is run
 *  are turned off so the example code runs much faster. If you want 
 *  to verify the results are correct, define the 'DEBUG_LTIMES' macro
 *  below or turn on checking for individual variants.
 */


//#define DEBUG_LTIMES


using namespace RAJA;

//
// Index value types for strongly-typed indices must be defined outside 
// function scope for RAJA CUDA variants to work.
//
// These types provide strongly-typed index values so if something is wrong
// in loop ordering and/or indexing, the code will likely not compile.
//
RAJA_INDEX_VALUE_T(IM, int, "IM");
RAJA_INDEX_VALUE_T(ID, int, "ID");
RAJA_INDEX_VALUE_T(IG, int, "IG");
RAJA_INDEX_VALUE_T(IZ, int, "IZ");


//
// Function to check results
//
template <typename PHIVIEW_T, typename LVIEW_T, typename PSIVIEW_T>
void checkResult(PHIVIEW_T& phi, LVIEW_T& L, PSIVIEW_T& psi,
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
    L_data[i] = i+1;
  }

  for (Index_type i = 0; i < psi_size; ++i) {
    psi_data[i] = 2*i+1;
  }

  // Note phi_data will be set to zero before each variant is run.


//----------------------------------------------------------------------------//

{
  std::cout << "\n Running baseline C-version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  // Using restrict doesn't make much of a difference for most compilers.
#if 1
  double * RAJA_RESTRICT L = L_data;
  double * RAJA_RESTRICT psi = psi_data;
  double * RAJA_RESTRICT phi = phi_data;
#else
  double * L = L_data;
  double * psi = psi_data;
  double * phi = phi_data;
#endif

  RAJA::Timer timer;
  timer.start();

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

  timer.stop();
  std::cout << "  C-version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;
}

//----------------------------------------------------------------------------//

{
  std::cout << "\n Running C-version of LTimes (with Views)...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  // 
  // L(m, d) : 1 -> d is stride-1 dimension 
  using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension 
  using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension 
  using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));


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

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}

//----------------------------------------------------------------------------//

{
  std::cout << "\n Running RAJA sequential version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  // 
  // L(m, d) : 1 -> d is stride-1 dimension 
  using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension 
  using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension 
  using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using EXECPOL = 
    RAJA::KernelPolicy<
       statement::For<0, loop_exec,  // m       
         statement::For<1, loop_exec,  // d
           statement::For<2, loop_exec,  // g 
             statement::For<3, simd_exec,  // z
               statement::Lambda<0>
             >
           >
         >
       >
     >;

  auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_m),
                                   RAJA::TypedRangeSegment<ID>(0, num_d),
                                   RAJA::TypedRangeSegment<IG>(0, num_g),
                                   RAJA::TypedRangeSegment<IZ>(0, num_z));

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

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}

//----------------------------------------------------------------------------//

{
  std::cout << "\n Running RAJA sequential ARGS version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using EXECPOL =
    RAJA::KernelPolicy<
       statement::For<0, loop_exec,  // m
         statement::For<1, loop_exec,  // d
           statement::For<2, loop_exec,  // g
             statement::For<3, simd_exec,  // z
               statement::Lambda<0, Segs<0, 1, 2, 3>>
             >
           >
         >
       >
     >;

  auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_m),
                                   RAJA::TypedRangeSegment<ID>(0, num_d),
                                   RAJA::TypedRangeSegment<IG>(0, num_g),
                                   RAJA::TypedRangeSegment<IZ>(0, num_z));

  RAJA::Timer timer;
  timer.start();

  RAJA::kernel<EXECPOL>( segments,
    [=] (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );

  timer.stop();
  std::cout << "  RAJA sequential ARGS version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}

//----------------------------------------------------------------------------//

{
  std::cout << "\n Running RAJA sequential shmem version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  constexpr size_t tile_m = 25;
  constexpr size_t tile_d = 80;
  constexpr size_t tile_z = 256;
  constexpr size_t tile_g = 0;

  using RAJA::statement::Param;

  using EXECPOL =
    RAJA::KernelPolicy<

      // Create memory tiles
      statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1,2>,

      // Tile outer m,d loops
      statement::Tile<0, tile_fixed<tile_m>, loop_exec,  // m
        statement::Tile<1, tile_fixed<tile_d>, loop_exec,  // d


            // Load L(m,d) for m,d tile into shmem
            statement::For<0, loop_exec,  // m
              statement::For<1, loop_exec,  // d
                statement::Lambda<0, Segs<0, 1>,
                                     Params<0>,
                                     Offsets<0, 1>>
              >
            >,

            // Run inner g, z loops with z loop tiled
            statement::For<2, loop_exec,  // g
              statement::Tile<3, tile_fixed<tile_z>, loop_exec,  // z


                  // Load psi into shmem
                  statement::For<1, loop_exec,  // d
                    statement::For<3, loop_exec,  // z
                      statement::Lambda<1, Segs<1, 2, 3>,
                                           Params<1>,
                                           Offsets<1, 2, 3>>
                    >
                  >,

                  // Compute phi
                  statement::For<0, loop_exec,  // m

                    // Load phi into shmem
                    statement::For<3, loop_exec,  // z
                      statement::Lambda<2, Segs<0, 2, 3>,
                                           Params<2>,
                                           Offsets<0, 2, 3>>
                    >,

                    // Compute phi in shmem
                    statement::For<1, loop_exec,  // d
                      statement::For<3, loop_exec,  // z
                        statement::Lambda<3, Params<0, 1, 2>,
                                             Offsets<0, 1, 2, 3>>
                      >
                    >,

                    // Store phi
                    statement:: For<3, loop_exec,  // z
                      statement::Lambda<4, Segs<0, 2, 3>,
                                           Params<2>,
                                           Offsets<0, 2, 3>>
                    >
                  >  // m


              >  // Tile z
            >  // g


        >  // Tile d
      >  // Tile m
      > // LocalMemory
    >; // KernelPolicy



  //
  // Define statically dimensioned local arrays used in kernel
  //

  using shmem_L_t = RAJA::TypedLocalArray<double,
                        RAJA::PERM_JI,
                        RAJA::SizeList<tile_m, tile_d>,
                        IM, ID>;
  shmem_L_t shmem_L;


  using shmem_psi_t = RAJA::TypedLocalArray<double,
                        RAJA::PERM_IJK,
                        RAJA::SizeList<tile_d, tile_g, tile_z>,
                        ID, IG, IZ>;
  shmem_psi_t shmem_psi;


  using shmem_phi_t = RAJA::TypedLocalArray<double,
                        RAJA::PERM_IJK,
                        RAJA::SizeList<tile_m, tile_g, tile_z>,
                        IM, IG, IZ>;
  shmem_phi_t shmem_phi;


  RAJA::Timer timer;
  timer.start();

  RAJA::kernel_param<EXECPOL>(

    RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_m),
                     RAJA::TypedRangeSegment<ID>(0, num_d),
                     RAJA::TypedRangeSegment<IG>(0, num_g),
                     RAJA::TypedRangeSegment<IZ>(0, num_z)),
    // For kernel_param, second arg is a tuple of data objects used in lambdas.
    // They are the last args in all lambdas (after indices).
    RAJA::make_tuple( shmem_L,
                      shmem_psi,
                      shmem_phi),


    // Lambda<0> : Load L into shmem
    [=] (IM m, ID d,
         shmem_L_t& sh_L,
         IM tm, ID td)
    {
      sh_L(tm, td) = L(m, d);
    },

    // Lambda<1> : Load psi into shmem
    [=] (ID d, IG g, IZ z,
         shmem_psi_t& sh_psi,
         ID td, IG tg, IZ tz)
    {
      sh_psi(td, tg, tz) = psi(d, g, z);
    },

    // Lambda<2> : Load phi into shmem
    [=] (IM m, IG g, IZ z,
         shmem_phi_t& sh_phi,
         IM tm, IG tg, IZ tz)
    {
      sh_phi(tm, tg, tz) = phi(m, g, z);
    },

    // Lambda<3> : Compute phi in shmem
    [=] (shmem_L_t& sh_L, shmem_psi_t& sh_psi, shmem_phi_t& sh_phi,
        IM tm, ID td, IG tg, IZ tz)
    {
      sh_phi(tm, tg, tz) += sh_L(tm, td) * sh_psi(td, tg, tz);
    },

    // Lambda<4> : Store phi
    [=] (IM m, IG g, IZ z,
         shmem_phi_t& sh_phi,
         IM tm, IG tg, IZ tz)
    {
      phi(m, g, z) = sh_phi(tm, tg, tz);
    }

  );

  timer.stop();
  std::cout << "  RAJA sequential shmem version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
{
  std::cout << "\n Running RAJA OpenMP version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  // 
  // L(m, d) : 1 -> d is stride-1 dimension 
  using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension 
  using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension 
  using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));


#if 1
  using EXECPOL =
    RAJA::KernelPolicy<
      statement::For<0, omp_parallel_for_exec,  // m
        statement::For<1, loop_exec,  // d 
          statement::For<2, loop_exec,  // g 
            statement::For<3, simd_exec,  // z 
              statement::Lambda<0>
            >
          >
        >
      >
    >;
#else
  //
  // Benefits of using OpenMP collapse depends on compiler, platform,
  // relative segment sizes.
  //
  using EXECPOL =
    RAJA::KernelPolicy<
      statement::Collapse<omp_parallel_collapse_exec,
                          RAJA::ArgList<0, 2, 3>,   // m, g, z
        statement::For<1, loop_exec,  // d 
          statement::Lambda<0>
        >
      >
    >;
#endif

  auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_m),
                                   RAJA::TypedRangeSegment<ID>(0, num_d),
                                   RAJA::TypedRangeSegment<IG>(0, num_g),
                                   RAJA::TypedRangeSegment<IZ>(0, num_z));

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

#if defined(DEBUG_LTIMES)
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

  //
  // View types and Views/Layouts for indexing into arrays
  // 
  // L(m, d) : 1 -> d is stride-1 dimension 
  using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension 
  using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension 
  using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(dL_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(dpsi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(dphi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using EXECPOL =
    RAJA::KernelPolicy<
      statement::CudaKernelAsync<    
        statement::For<0, cuda_block_x_loop,  // m
          statement::For<2, cuda_block_y_loop,  // g
            statement::For<3, cuda_thread_x_loop,  // z
              statement::For<1, seq_exec,  // d
                statement::Lambda<0>
              >
            >
          >
        >
      >         
    >;          
                 
  auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_m),
                                   RAJA::TypedRangeSegment<ID>(0, num_d),
                                   RAJA::TypedRangeSegment<IG>(0, num_g),
                                   RAJA::TypedRangeSegment<IZ>(0, num_z));

  RAJA::Timer timer;
  cudaErrchk( cudaDeviceSynchronize() );
  timer.start();

  RAJA::kernel<EXECPOL>( segments,
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
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

  // Reset data in Views to CPU data
  L.set_data(L_data); 
  psi.set_data(psi_data); 
  phi.set_data(phi_data); 

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


  //
  // View types and Views/Layouts for indexing into arrays
  // 
  // L(m, d) : 1 -> d is stride-1 dimension 
  using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(dL_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(dpsi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(dphi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));


  static const int tile_m = 25;
  static const int tile_d = 90;
  static const int tile_g = 0;
  static const int tile_z = 40;




  //
  // Define statically dimensioned local arrays used in kernel
  //

  using shmem_L_t = RAJA::TypedLocalArray<double,
                        RAJA::PERM_IJ,
                        RAJA::SizeList<tile_m, tile_d>,
                        IM, ID>;
  shmem_L_t shmem_L;


  using shmem_psi_t = RAJA::TypedLocalArray<double,
                        RAJA::PERM_IJK,
                        RAJA::SizeList<tile_d, tile_g, tile_z>,
                        ID, IG, IZ>;
  shmem_psi_t shmem_psi;



  //
  // Define our execution policy
  //

  using RAJA::Segs;
  using RAJA::Params;
  using RAJA::Offsets;

  using EXECPOL =
    RAJA::KernelPolicy<
      statement::CudaKernelAsync<
        statement::InitLocalMem<cuda_shared_mem, ParamList<0,1>,
          // Tile outer m,d loops 
          statement::Tile<0, tile_fixed<tile_m>, seq_exec,  // m
            statement::Tile<1, tile_fixed<tile_d>, seq_exec,  // d

              // Load L for m,d tile into shmem 
              statement::For<1, cuda_thread_x_loop,  // d
                statement::For<0, cuda_thread_y_direct,   // m
                  statement::Lambda<0, Segs<0,1>, Params<0>, Offsets<0,1>>
                >
              >,
              statement::CudaSyncThreads,

              // Distribute g, z across blocks and tile z
              statement::For<2, cuda_block_y_loop, // g
                statement::Tile<3, tile_fixed<tile_z>, cuda_block_x_loop,  // z

                  // Load phi into thread local storage
                  statement::For<3, cuda_thread_x_direct,  // z
                    statement::For<0, cuda_thread_y_direct, // m
                      statement::Lambda<2, Segs<0,2,3>, Params<2>>
                    >
                  >,

                  // Load slice of psi into shmem
                  statement::For<3,cuda_thread_x_direct,  // z
                    statement::For<1, cuda_thread_y_loop, // d (reusing y)
                      statement::Lambda<1, Segs<1,2,3>, Params<1>, Offsets<1,2,3>>
                    >
                  >,
                  statement::CudaSyncThreads,

                  // Compute phi
                  statement::For<3, cuda_thread_x_direct,  // z
                    statement::For<0, cuda_thread_y_direct, // m

                      // Compute thread-local Phi value and store
                      statement::For<1,  seq_exec,  // d
                        statement::Lambda<3, Segs<0,1,2,3>, Params<0,1,2>, Offsets<0,1,2,3>>
                      > // d
                    >  // m
                  >,  // z
                  
                  // finish tile over directions
                  statement::CudaSyncThreads,

                  // Write out phi from thread local storage
                  statement::For<3, cuda_thread_x_direct,  // z
                    statement::For<0, cuda_thread_y_direct, // m
                      statement::Lambda<4, Segs<0,2,3>, Params<2>>
                    >
                  >,
                  statement::CudaSyncThreads
                
                >  // Tile z
              >  // g

            >  // Tile d
          >  // Tile m
        > // init shmem
      >  // CudaKernelAsync

    >;  // KernelPolicy


  


  RAJA::Timer timer;
  cudaErrchk( cudaDeviceSynchronize() );
  timer.start();

  RAJA::kernel_param<EXECPOL>(
      RAJA::make_tuple(
      RAJA::TypedRangeSegment<IM>(0, num_m),
      RAJA::TypedRangeSegment<ID>(0, num_d),
      RAJA::TypedRangeSegment<IG>(0, num_g),
      RAJA::TypedRangeSegment<IZ>(0, num_z)),

    // For kernel_param, second arg is a tuple of data objects used in lambdas.
    // They are the last args in all lambdas (after indices).
    // Here, the last entry '0.0' yields a thread-private temporary for
    // computing a phi value, for shared memory before writing to phi array.
    RAJA::make_tuple( shmem_L,
                      shmem_psi,
                      0.0),

    // Lambda<0> : Load L into shmem
    [=] RAJA_DEVICE (IM m, ID d,
                     shmem_L_t& sh_L,
                     IM tm, ID td) {
      sh_L(tm, td) = L(m, d);
    },

    // Lambda<1> : Load slice of psi into shmem
    [=] RAJA_DEVICE (ID d, IG g, IZ z,
                     shmem_psi_t& sh_psi,
                     ID td, IG tg, IZ tz) {

      sh_psi(td, tg, tz) = psi(d, g, z);
    },

    // Lambda<2> : Load thread-local phi value
    [=] RAJA_DEVICE (IM m, IG g, IZ z,
                     double& phi_local) {

      phi_local = phi(m, g, z);
    },

    // Lambda<3> Compute thread-local phi value
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z,
                     shmem_L_t& sh_L, shmem_psi_t& sh_psi, double& phi_local,
                     IM tm, ID td, IG tg, IZ tz) {

      phi_local += sh_L(tm, td) *  sh_psi(td, tg, tz);
    },

    // Lambda<4> : Store phi
    [=] RAJA_DEVICE (IM m, IG g, IZ z,
                     double& phi_local) {

      phi(m, g, z) = phi_local;
    }

  );

  cudaDeviceSynchronize();
  timer.stop();
  std::cout << "  RAJA CUDA + shmem version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;



#if defined(DEBUG_LTIMES)

  cudaErrchk( cudaMemcpy( phi_data, dphi_data, phi_size * sizeof(double),
                          cudaMemcpyDeviceToHost ) );

  // Reset data in Views to CPU data
  L.set_data(L_data); 
  psi.set_data(psi_data); 
  phi.set_data(phi_data); 
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif


  cudaErrchk( cudaFree( dL_data ) );
  cudaErrchk( cudaFree( dpsi_data ) );
  cudaErrchk( cudaFree( dphi_data ) );
}
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
{
  std::cout << "\n Running RAJA HIP version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  double* dL_data   = nullptr;
  double* dpsi_data = nullptr;
  double* dphi_data = nullptr;

  hipErrchk( hipMalloc( (void**)&dL_data, L_size * sizeof(double) ) );
  hipErrchk( hipMemcpy( dL_data, L_data, L_size * sizeof(double),
                          hipMemcpyHostToDevice ) );
  hipErrchk( hipMalloc( (void**)&dpsi_data, psi_size * sizeof(double) ) );
  hipErrchk( hipMemcpy( dpsi_data, psi_data, psi_size * sizeof(double),
                          hipMemcpyHostToDevice ) );
  hipErrchk( hipMalloc( (void**)&dphi_data, phi_size * sizeof(double) ) );
  hipErrchk( hipMemcpy( dphi_data, phi_data, phi_size * sizeof(double),
                          hipMemcpyHostToDevice ) );

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(dL_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(dpsi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(dphi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using EXECPOL =
    RAJA::KernelPolicy<
      statement::HipKernelAsync<
        statement::For<0, hip_block_x_loop,  // m
          statement::For<2, hip_block_y_loop,  // g
            statement::For<3, hip_thread_x_loop,  // z
              statement::For<1, seq_exec,  // d
                statement::Lambda<0>
              >
            >
          >
        >
      >
    >;

  auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_m),
                                   RAJA::TypedRangeSegment<ID>(0, num_d),
                                   RAJA::TypedRangeSegment<IG>(0, num_g),
                                   RAJA::TypedRangeSegment<IZ>(0, num_z));

  RAJA::Timer timer;
  hipErrchk( hipDeviceSynchronize() );
  timer.start();

  RAJA::kernel<EXECPOL>( segments,
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );

  hipErrchk( hipDeviceSynchronize() );
  timer.stop();
  std::cout << "  RAJA HIP version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;

  hipErrchk( hipMemcpy( phi_data, dphi_data, phi_size * sizeof(double),
                          hipMemcpyDeviceToHost ) );

  hipErrchk( hipFree( dL_data ) );
  hipErrchk( hipFree( dpsi_data ) );
  hipErrchk( hipFree( dphi_data ) );

  // Reset data in Views to CPU data
  L.set_data(L_data);
  psi.set_data(psi_data);
  phi.set_data(phi_data);

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
{
  std::cout << "\n Running RAJA HIP + shmem version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  double* dL_data   = nullptr;
  double* dpsi_data = nullptr;
  double* dphi_data = nullptr;

  hipErrchk( hipMalloc( (void**)&dL_data, L_size * sizeof(double) ) );
  hipErrchk( hipMemcpy( dL_data, L_data, L_size * sizeof(double),
                          hipMemcpyHostToDevice ) );
  hipErrchk( hipMalloc( (void**)&dpsi_data, psi_size * sizeof(double) ) );
  hipErrchk( hipMemcpy( dpsi_data, psi_data, psi_size * sizeof(double),
                          hipMemcpyHostToDevice ) );
  hipErrchk( hipMalloc( (void**)&dphi_data, phi_size * sizeof(double) ) );
  hipErrchk( hipMemcpy( dphi_data, phi_data, phi_size * sizeof(double),
                          hipMemcpyHostToDevice ) );


  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(dL_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(dpsi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(dphi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));


  static const int tile_m = 25;
  static const int tile_d = 90;
  static const int tile_g = 0;
  static const int tile_z = 40;




  //
  // Define statically dimensioned local arrays used in kernel
  //

  using shmem_L_t = RAJA::TypedLocalArray<double,
                        RAJA::PERM_IJ,
                        RAJA::SizeList<tile_m, tile_d>,
                        IM, ID>;
  shmem_L_t shmem_L;


  using shmem_psi_t = RAJA::TypedLocalArray<double,
                        RAJA::PERM_IJK,
                        RAJA::SizeList<tile_d, tile_g, tile_z>,
                        ID, IG, IZ>;
  shmem_psi_t shmem_psi;



  //
  // Define our execution policy
  //

  using RAJA::statement::Param;
  using RAJA::Segs;
  using RAJA::Params;
  using RAJA::Offsets;

  using EXECPOL =
    RAJA::KernelPolicy<
      statement::HipKernelAsync<
        statement::InitLocalMem<hip_shared_mem, ParamList<0,1>,
          // Tile outer m,d loops 
          statement::Tile<0, tile_fixed<tile_m>, seq_exec,  // m
            statement::Tile<1, tile_fixed<tile_d>, seq_exec,  // d

              // Load L for m,d tile into shmem 
              statement::For<1, hip_thread_x_loop,  // d
                statement::For<0, hip_thread_y_direct,   // m
                  statement::Lambda<0, Segs<0,1>, Params<0>, Offsets<0,1>>
                >
              >,
              statement::HipSyncThreads,

              // Distribute g, z across blocks and tile z
              statement::For<2, hip_block_y_loop, // g
                statement::Tile<3, tile_fixed<tile_z>, hip_block_x_loop,  // z

                  // Load phi into thread local storage
                  statement::For<3, hip_thread_x_direct,  // z
                    statement::For<0, hip_thread_y_direct, // m
                      statement::Lambda<2, Segs<0,2,3>, Params<2>>
                    >
                  >,

                  // Load slice of psi into shmem
                  statement::For<3, hip_thread_x_direct,  // z
                    statement::For<1, hip_thread_y_loop, // d (reusing y)
                      statement::Lambda<1, Segs<1,2,3>, Params<1>, Offsets<1,2,3>>
                    >
                  >,
                  statement::HipSyncThreads,

                  // Compute phi
                  statement::For<3, hip_thread_x_direct,  // z
                    statement::For<0, hip_thread_y_direct, // m

                      // Compute thread-local Phi value and store
                      statement::For<1,  seq_exec,  // d
                        statement::Lambda<3, Segs<0,1,2,3>, Params<0,1,2>, Offsets<0,1,2,3>>
                      > // d
                    >  // m
                  >,  // z
                  
                  // finish tile over directions
                  statement::HipSyncThreads,

                  // Write out phi from thread local storage
                  statement::For<3, hip_thread_x_direct,  // z
                    statement::For<0, hip_thread_y_direct, // m
                      statement::Lambda<4, Segs<0,2,3>, Params<2>>
                    >
                  >,
                  statement::HipSyncThreads
                
                >  // Tile z
              >  // g

            >  // Tile d
          >  // Tile m
        > // init shmem
      >  // HipKernelAsync

    >;  // KernelPolicy




  RAJA::Timer timer;
  hipErrchk( hipDeviceSynchronize() );
  timer.start();

  RAJA::kernel_param<EXECPOL>(
      RAJA::make_tuple(
      RAJA::TypedRangeSegment<IM>(0, num_m),
      RAJA::TypedRangeSegment<ID>(0, num_d),
      RAJA::TypedRangeSegment<IG>(0, num_g),
      RAJA::TypedRangeSegment<IZ>(0, num_z)),

    // For kernel_param, second arg is a tuple of data objects used in lambdas.
    // They are the last args in all lambdas (after indices).
    // Here, the last entry '0.0' yields a thread-private temporary for
    // computing a phi value, for shared memory before writing to phi array.
    RAJA::make_tuple( shmem_L,
                      shmem_psi,
                      0.0),

    // Lambda<0> : Load L into shmem
    [=] RAJA_DEVICE (IM m, ID d,
                     shmem_L_t& sh_L,
                     IM tm, ID td) {
      sh_L(tm, td) = L(m, d);
    },

    // Lambda<1> : Load slice of psi into shmem
    [=] RAJA_DEVICE (ID d, IG g, IZ z,
                     shmem_psi_t& sh_psi,
                     ID td, IG tg, IZ tz) {

      sh_psi(td, tg, tz) = psi(d, g, z);
    },

    // Lambda<2> : Load thread-local phi value
    [=] RAJA_DEVICE (IM m, IG g, IZ z,
                     double& phi_local) {

      phi_local = phi(m, g, z);
    },

    // Lambda<3> Compute thread-local phi value
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z,
                     shmem_L_t& sh_L, shmem_psi_t& sh_psi, double& phi_local,
                     IM tm, ID td, IG tg, IZ tz) {

      phi_local += sh_L(tm, td) *  sh_psi(td, tg, tz);
    },

    // Lambda<4> : Store phi
    [=] RAJA_DEVICE (IM m, IG g, IZ z,
                     double& phi_local) {

      phi(m, g, z) = phi_local;
    }

  );

  hipDeviceSynchronize();
  timer.stop();
  std::cout << "  RAJA HIP + shmem version of LTimes run time (sec.): "
            << timer.elapsed() << std::endl;



#if defined(DEBUG_LTIMES)

  hipErrchk( hipMemcpy( phi_data, dphi_data, phi_size * sizeof(double),
                          hipMemcpyDeviceToHost ) );

  // Reset data in Views to CPU data
  L.set_data(L_data);
  psi.set_data(psi_data);
  phi.set_data(phi_data);
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif


  hipErrchk( hipFree( dL_data ) );
  hipErrchk( hipFree( dpsi_data ) );
  hipErrchk( hipFree( dphi_data ) );
}
#endif

//----------------------------------------------------------------------------//

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
template <typename PHIVIEW_T, typename LVIEW_T, typename PSIVIEW_T>
void checkResult(PHIVIEW_T& phi, LVIEW_T& L, PSIVIEW_T& psi,
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
