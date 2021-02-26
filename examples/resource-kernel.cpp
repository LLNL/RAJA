//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
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

using namespace RAJA;

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
  const Index_type num_z = 64*64;
  //const Index_type num_z = 64*1024;

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


#if defined(RAJA_ENABLE_CUDA)
{
  std::cout << "\n Running RAJA CUDA version of LTimes...\n";

  RAJA::resources::Cuda cuda_res;
  std::cout << " Cuda Resource Stream : " << cuda_res.get_stream() << "\n";
  std::cout << " Cuda Resource Default Stream : " << resources::Cuda::get_default().get_stream() << "\n";

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
              > // For seq_exec
            > // For thread x
          > // For block y
        > // For block x
      > // CudaKernelAsync       
    >; // KernelPolicy 
                 
  auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_m),
                                   RAJA::TypedRangeSegment<ID>(0, num_d),
                                   RAJA::TypedRangeSegment<IG>(0, num_g),
                                   RAJA::TypedRangeSegment<IZ>(0, num_z));

  RAJA::Timer timer;
  cudaErrchk( cudaDeviceSynchronize() );
  timer.start();

  RAJA::kernel_resource<EXECPOL>( segments, cuda_res,
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

  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
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
      }
    }
  }

  if ( nerrors == 0 ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL : " << nerrors << " errors!\n";
  }
}
