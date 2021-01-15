//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Place the following line before including RAJA to enable
// statistics on the Vector abstractions
//#define RAJA_ENABLE_VECTOR_STATS

// Un-comment the following line to run correctness checks on each variant
//#define DEBUG_LTIMES

#include "RAJA/config.hpp"

#define VARIANT_C                    0
#define VARIANT_C_VIEWS              0
#define VARIANT_RAJA_SEQ             0
#define VARIANT_RAJA_SEQ_ARGS        0
#define VARIANT_RAJA_TEAMS_SEQ       0
#define VARIANT_RAJA_VECTOR          0
#define VARIANT_RAJA_MATRIX          1
#define VARIANT_RAJA_TEAMS_MATRIX    1
#define VARIANT_RAJA_SEQ_SHMEM       0
#define VARIANT_RAJA_MATRIX_SHMEM    0

#if defined(RAJA_ENABLE_OPENMP)
#define VARIANT_RAJA_OPENMP          0
#endif

#if defined(RAJA_ENABLE_CUDA)
#define VARIANT_CUDA_KERNEL          1
#define VARIANT_CUDA_TEAMS           1
#define VARIANT_CUDA_TEAMS_MATRIX    1
#define VARIANT_CUDA_KERNEL_SHMEM    1
#endif

#if defined(RAJA_ENABLE_HIP)
#define RAJA_HIP_KERNEL              1
#define RAJA_HIP_KERNEL_SHMEM        1
#endif


extern "C" {
  void dgemm_(char * transa, char * transb, int * m, int * n, int * k,
              double * alpha, double * A, int * lda,
              double * B, int * ldb, double * beta,
              double *, int * ldc);
}

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
                 const int num_m, 
                 const int num_d,
                 const int num_g,
                 const int num_z);



int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  std::cout << "\n\nRAJA LTIMES example...\n\n";

//----------------------------------------------------------------------------//
// Define array dimensions, allocate arrays, define Layouts and Views, etc.
  // Note: rand()/RAND_MAX is always zero, but forces the compiler to not
  // optimize out these values as compile time constants
  const int num_m = 25 + (rand()/RAND_MAX);
  const int num_g = 48 + (rand()/RAND_MAX);
  const int num_d = 80 + (rand()/RAND_MAX);

//  const int num_m = 8 + (rand()/RAND_MAX);
//  const int num_g = 8 + (rand()/RAND_MAX);
//  const int num_d = 8 + (rand()/RAND_MAX);


#ifdef DEBUG_LTIMES
  const int num_iter = 1 + (rand()/RAND_MAX);;
  // use a decreased number of zones since this will take a lot longer
  // and we're not really measuring performance here
  const long num_z = 128 + (rand()/RAND_MAX);
#else
  const int num_iter = 10 + (rand()/RAND_MAX);
  const int num_z = 32*1024 + (rand()/RAND_MAX);
#endif



  double total_flops = 2.0*num_g*num_z*num_d*num_m*num_iter;

  std::cout << "num_m = " << num_m << ", num_g = " << num_g <<
               ", num_d = " << num_d << ", num_z = " << num_z << "\n\n";

  std::cout << "total flops:  " << (long)total_flops << "\n";

  const long L_size   = num_m * num_d;
  const long psi_size = num_d * num_g * num_z;
  const long phi_size = num_m * num_g * num_z;

  std::vector<double> L_vec(num_m * num_d);
  std::vector<double> psi_vec(num_d * num_g * num_z);
  std::vector<double> phi_vec(num_m * num_g * num_z);

  double* L_data   = &L_vec[0];
  double* psi_data = &psi_vec[0];
  double* phi_data = &phi_vec[0];

  for (int i = 0; i < L_size; ++i) {
    L_data[i] = i+1;
  }

  for (int i = 0; i < psi_size; ++i) {
    psi_data[i] = 2*i+1;
  }

  // Note phi_data will be set to zero before each variant is run.


//----------------------------------------------------------------------------//

#if VARIANT_C
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

  for (int iter = 0;iter < num_iter;++ iter)
    for (int g = 0; g < num_g; ++g) {
      for (int z = 0; z < num_z; ++z) {
        for (int m = 0; m < num_m; ++m) {
          for (int d = 0; d < num_d; ++d) {
            phi[g*num_z*num_m + z*num_m + m] +=
              L[d*num_m + m] * psi[g*num_z*num_d + z*num_d + d];
        }
      }
    }
  }

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  C-version of LTimes run time (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;

}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_C_VIEWS
{
  std::cout << "\n Running C-version of LTimes (with Views)...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  // 
  // L(m, d) : 1 -> d is stride-1 dimension 
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension 
  using PsiView = TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension 
  using PhiView = TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{2, 1, 0}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{2, 1, 0}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));


  RAJA::Timer timer;
  timer.start(); 

  for (int iter = 0;iter < num_iter;++ iter)
    for (IG g(0); g < num_g; ++g) {
      for (IZ z(0); z < num_z; ++z) {
        for (IM m(0); m < num_m; ++m) {
          for (ID d(0); d < num_d; ++d) {
            phi(m, g, z) += L(m, d) * psi(d, g, z);
        }
      }
    }
  }

  timer.stop(); 
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  C-version of LTimes run time (with Views) (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;


#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_RAJA_SEQ
{
  std::cout << "\n Running RAJA sequential version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  // 
  // L(m, d) : 1 -> d is stride-1 dimension 
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension 
  using PsiView = TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension 
  using PhiView = TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{2, 1, 0}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{2, 1, 0}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using EXECPOL = 
    RAJA::KernelPolicy<
      statement::For<2, loop_exec,  // g
        statement::For<3, loop_exec,  // z
         statement::For<0, loop_exec,  // m
           statement::For<1, simd_exec,  // d
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

  for (int iter = 0;iter < num_iter;++ iter)
  RAJA::kernel<EXECPOL>( segments,
    [=] (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA sequential version of LTimes run time (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;


#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_RAJA_SEQ_ARGS
{
  std::cout << "\n Running RAJA sequential ARGS version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{2, 1, 0}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{2, 1, 0}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using EXECPOL =
    RAJA::KernelPolicy<
      statement::For<2, loop_exec,  // g
        statement::For<3, loop_exec,  // z
           statement::For<0, loop_exec,  // m
             statement::For<1, simd_exec,  // d
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

  for (int iter = 0;iter < num_iter;++ iter)
  RAJA::kernel<EXECPOL>( segments,
    [=] (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA sequential ARGS version of LTimes run time (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;


#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_RAJA_TEAMS_SEQ
{
  std::cout << "\n Running RAJA Teams sequential version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;


  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{2, 1, 0}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{2, 1, 0}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));


  using pol_launch = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t>;
  using pol_g = RAJA::expt::LoopPolicy<RAJA::loop_exec>;
  using pol_z = RAJA::expt::LoopPolicy<RAJA::loop_exec>;
  using pol_m = RAJA::expt::LoopPolicy<RAJA::loop_exec>;
  using pol_d = RAJA::expt::LoopPolicy<RAJA::loop_exec>;



  RAJA::Timer timer;
  timer.start();

  for (int iter = 0;iter < num_iter;++ iter){
    RAJA::expt::launch<pol_launch>(RAJA::expt::HOST, RAJA::expt::Resources(), [=](RAJA::expt::LaunchContext ctx){

      RAJA::expt::loop<pol_g>(ctx, RAJA::TypedRangeSegment<IG>(0, num_g), [&](IG g){
        RAJA::expt::loop<pol_z>(ctx, RAJA::TypedRangeSegment<IZ>(0, num_z), [&](IZ z){
          RAJA::expt::loop<pol_m>(ctx, RAJA::TypedRangeSegment<IM>(0, num_m), [&](IM m){
            RAJA::expt::loop<pol_d>(ctx, RAJA::TypedRangeSegment<ID>(0, num_d), [&](ID d){
              phi(m, g, z) += L(m, d) * psi(d, g, z);
            });
          });
        });
      });

    }); // laucnch
  } // iter

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA Teams sequential version of LTimes run time (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;


#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif


}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_RAJA_VECTOR
{
  std::cout << "\n Running RAJA vectorized version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{1, 0, 2}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{1, 0, 2}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using vector_t = RAJA::StreamVector<double,2>;
  using VecIZ = RAJA::VectorIndex<IZ, vector_t>;

  using EXECPOL =
    RAJA::KernelPolicy<
      statement::For<2, loop_exec,  // g
        statement::For<3, vector_exec<vector_t>,  // z
          statement::For<0, loop_exec,  // m
            statement::For<1, loop_exec,  // d

               statement::Lambda<0>
             >
           >
         >
       >
     >;


#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::resetVectorStats();
#endif


  auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_m),
                                   RAJA::TypedRangeSegment<ID>(0, num_d),
                                   RAJA::TypedRangeSegment<IG>(0, num_g),
                                   RAJA::TypedRangeSegment<IZ>(0, num_z));

  RAJA::Timer timer;
  timer.start();

  for (int iter = 0;iter < num_iter;++ iter)
  RAJA::kernel<EXECPOL>( segments,
    [=] (IM m, ID d, IG g, VecIZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA vectorized version of LTimes run time (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;

#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::printVectorStats();
#endif

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_RAJA_MATRIX
{
  std::cout << "\n Running RAJA row-major matrix version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;


  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{1, 2, 0}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{1, 2, 0}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using matrix_t = RAJA::MatrixRegister<double, ColMajorLayout>;

	std::cout << "vector width: " << matrix_t::s_dim_elem(0) << std::endl;

  using RowM = RAJA::RowIndex<IM, matrix_t>;
  using ColD = RAJA::ColIndex<ID, matrix_t>;
  using ColZ = RAJA::ColIndex<IZ, matrix_t>;


#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::resetVectorStats();
#endif

// rzgenies: 1  cores at 2.1GHz, 2FMAs/cycle=16ops/cycle = 33.6 GF peak
// rzgenies: 36 cores at 2.1GHz, 2FMAs/cycle=16ops/cycle =  1.2 TF peak



  RAJA::Timer timer;
  timer.start();

  for (int iter = 0;iter < num_iter;++ iter){

  RAJA::forall<loop_exec>(RAJA::TypedRangeSegment<IG>(0, num_g),
    [=](IG g)
    {

#if 1

      auto rows_m = RowM::all();
      auto cols_z = ColZ::all();
      auto cols_d = ColD::all();
      auto rows_d = toRowIndex(cols_d);

        phi(rows_m, g, cols_z) +=
            L(rows_m, cols_d) * psi(rows_d, g, cols_z);

#else
        // try using BLAS DGEMM for comparison
//        void dgemm_(char * transa, char * transb, int * m, int * n, int * k,
//                    double * alpha, double * A, int * lda,
//                    double * B, int * ldb, double * beta,
//                    double *, int * ldc);
        char transa = 'n';
        char transb = 'n';
        int m = num_m;
        int n = num_z;
        int k = num_d;
        double alpha = 1.0;
        double beta = 1.0;
        double *Aptr = &L(IM(0), ID(0));
        double *Bptr = &psi(ID(0), g, IZ(0));
        double *Cptr = &phi(IM(0), g, IZ(0));
        int lda = m;
        int ldb = k;
        int ldc = m;
        dgemm_(&transa, &transb, &m, &n, &k,
               &alpha,
               Aptr, &lda,
               Bptr, &ldb,
               &beta,
               Cptr, &ldc);

#endif
    });



  }

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA row-major matrix version of LTimes run time (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;

#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::printVectorStats();
#endif

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif


}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_RAJA_MATRIX0
{
  std::cout << "\n Running RAJA column-major matrix version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{2, 1, 0}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{2, 1, 0}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using matrix_t = RAJA::RegisterMatrix<double, MATRIX_COL_MAJOR>;


  using RowM = RAJA::RowIndex<IM, matrix_t>;
  using ColD = RAJA::ColIndex<ID, matrix_t>;
  using ColZ = RAJA::ColIndex<IZ, matrix_t>;

  using EXECPOL =
    RAJA::KernelPolicy<
    statement::For<2, loop_exec,  // g
      statement::For<3, matrix_col_exec<matrix_t>,  // z
       statement::For<0, matrix_row_exec<matrix_t>,  // m
         statement::For<1, matrix_col_exec<matrix_t>,  // d

           statement::Lambda<0>
         >

       >
     > >
     >;


#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::resetVectorStats();
#endif


// rzgenies: 1  cores at 2.1GHz, 2FMAs/cycle=16ops/cycle = 33.6 GF peak
// rzgenies: 36 cores at 2.1GHz, 2FMAs/cycle=16ops/cycle =  1.2 TF peak



  auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_m),
                                   RAJA::TypedRangeSegment<ID>(0, num_d),
                                   RAJA::TypedRangeSegment<IG>(0, num_g),
                                   RAJA::TypedRangeSegment<IZ>(0, num_z));


  RAJA::Timer timer;
  timer.start();

  for (int iter = 0;iter < num_iter;++ iter){
		RAJA::kernel<EXECPOL>( segments,
			[=] (RowM m, ColD d, IG g, ColZ z) {
				phi(m, g, z) = L(m, d) * psi(toRowIndex(d), g, z) + phi(m, g, z);
			}
		);
  }

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA column-major matrix version of LTimes run time (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;

#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::printVectorStats();
#endif

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif


}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_RAJA_TEAMS_MATRIX
{
  std::cout << "\n Running RAJA Teams column-major matrix version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{1, 2, 0}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{1, 2, 0}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  using matrix_t = RAJA::MatrixRegister<double, RAJA::ColMajorLayout>;


  //using RowM = RAJA::RowIndex<IM, matrix_t>;
  using RowM = RAJA::RowIndex<IM, matrix_t>;

  using ColD = RAJA::ColIndex<ID, matrix_t>;
  using ColZ = RAJA::ColIndex<IZ, matrix_t>;

  using pol_launch = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t>;
  using pol_g = RAJA::expt::LoopPolicy<RAJA::loop_exec>;
  using pol_z = RAJA::expt::LoopPolicy<matrix_col_exec<matrix_t>>;
  using pol_m = RAJA::expt::LoopPolicy<matrix_row_exec<matrix_t>>;
  using pol_d = RAJA::expt::LoopPolicy<matrix_col_exec<matrix_t>>;




#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::resetVectorStats();
#endif


  RAJA::Timer timer;
  timer.start();


  for (int iter = 0;iter < num_iter;++ iter){
    RAJA::expt::launch<pol_launch>(RAJA::expt::HOST, RAJA::expt::Resources(), [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx){

      RAJA::expt::loop<pol_g>(ctx, RAJA::TypedRangeSegment<IG>(0, num_g), [&](IG g){
#if 0

        phi(rowM, g, colZ) += L(rowM, colD) * psi(toRow(colD), g, colZ);

#else
        RAJA::expt::loop<pol_z>(ctx, RAJA::TypedRangeSegment<IZ>(0, num_z), [&](ColZ z){
          RAJA::expt::loop<pol_m>(ctx, RAJA::TypedRangeSegment<IM>(0, num_m), [&](RowM m){

            RAJA::expt::loop<pol_d>(ctx, RAJA::TypedRangeSegment<ID>(0, num_d), [&](ColD d){
              phi(m,g,z) += L(m, d) * psi(toRowIndex(d), g, z);
            });

          });
        });
#endif
      });

    });
  }

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA Teams column-major matrix version of LTimes run time (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;

#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::printVectorStats();
#endif

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif


}
#endif

//----------------------------------------------------------------------------//
#if VARIANT_RAJA_SEQ_SHMEM
{
  std::cout << "\n Running RAJA sequential shmem version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 2>, IM, IG, IZ>;

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

  for (int iter = 0;iter < num_iter;++ iter)
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
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA sequential shmem version of LTimes run time (sec.): "
            << t <<", GFLOPS/sec: " << gflop_rate << std::endl;


#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_RAJA_MATRIX_SHMEM
{
  std::cout << "\n Running RAJA column-major matrix shmem version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{2, 1, 0}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{2, 1, 0}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));

  constexpr size_t tile_m = 32;
  constexpr size_t tile_d = 64;
  constexpr size_t tile_z = 16;
  constexpr size_t tile_g = 0;

  using matrix_t = RAJA::RegisterMatrix<double, RAJA::MATRIX_COL_MAJOR>;


  using RowM = RAJA::RowIndex<IM, matrix_t>;
  using RowD = RAJA::RowIndex<ID, matrix_t>;
  using ColD = RAJA::ColIndex<ID, matrix_t>;
  using ColZ = RAJA::ColIndex<IZ, matrix_t>;

  using RAJA::statement::Param;

  using EXECPOL =
    RAJA::KernelPolicy<



      statement::For<2, loop_exec,  // g

      // Create memory tiles
      statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0,1,2>,

      // Tile outer m,d loops
      statement::Tile<0, tile_fixed<tile_m>, loop_exec,  // m
        statement::Tile<1, tile_fixed<tile_d>, loop_exec,  // d


            // Load L(m,d) for m,d tile into shmem
            statement::For<0, matrix_row_exec<matrix_t>,  // m
              statement::For<1, matrix_col_exec<matrix_t>,  // d
                statement::Lambda<0, Segs<0, 1>,
                                     Params<0>,
                                     Offsets<0, 1>>
              >
            >,


            // Run inner z loops with z loop tiled
              statement::Tile<3, tile_fixed<tile_z>, loop_exec,  // z


                  // Load psi into shmem
                  statement::For<1, matrix_row_exec<matrix_t>,  // d
                    statement::For<3, matrix_col_exec<matrix_t>,  // z
                      statement::Lambda<1, Segs<1, 2, 3>,
                                           Params<1>,
                                           Offsets<1, 2, 3>>
                    >
                  >,

                  // Compute phi
                  statement::For<0, matrix_row_exec<matrix_t>,  // m

                    // Load phi into shmem
                    statement::For<3, matrix_col_exec<matrix_t>,  // z
                      statement::Lambda<2, Segs<0, 2, 3>,
                                           Params<2>,
                                           Offsets<0, 2, 3>>
                    >,

                    // Compute phi in shmem
                    statement::For<1, matrix_col_exec<matrix_t>,  // d
                      statement::For<3, matrix_col_exec<matrix_t>,  // z
                        statement::Lambda<3, Params<0, 1, 2>,
                                             Offsets<0, 1, 2, 3>>
                      >
                    >,

                    // Store phi
                    statement:: For<3, matrix_col_exec<matrix_t>,  // z
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
                        RAJA::PERM_JKI,
                        RAJA::SizeList<tile_d, tile_g, tile_z>,
                        ID, IG, IZ>;
  shmem_psi_t shmem_psi;


  using shmem_phi_t = RAJA::TypedLocalArray<double,
                        RAJA::PERM_JKI,
                        RAJA::SizeList<tile_m, tile_g, tile_z>,
                        IM, IG, IZ>;
  shmem_phi_t shmem_phi;

#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::resetVectorStats();
#endif

  RAJA::Timer timer;
  timer.start();

  for (int iter = 0;iter < num_iter;++ iter)
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
    [=] (RowM m, ColD d,
         shmem_L_t& sh_L,
         RowM tm, ColD td)
    {
      sh_L(tm, td) = L(m, d);
    },

    // Lambda<1> : Load psi into shmem
    [=] (RowD d, IG g, ColZ z,
         shmem_psi_t& sh_psi,
         RowD td, IG tg, ColZ tz)
    {
      sh_psi(td, tg, tz) = psi(d, g, z);
    },

    // Lambda<2> : Load phi into shmem
    [=] (RowM m, IG g, ColZ z,
         shmem_phi_t& sh_phi,
         RowM tm, IG tg, ColZ tz)
    {
      sh_phi(tm, tg, tz) = phi(m, g, z);
    },

    // Lambda<3> : Compute phi in shmem
    [=] (shmem_L_t& sh_L, shmem_psi_t& sh_psi, shmem_phi_t& sh_phi,
        RowM tm, ColD td, IG tg, ColZ tz)
    {
      sh_phi(tm, tg, tz) = sh_L(tm, td) * sh_psi(toRowIndex(td), tg, tz) + sh_phi(tm, tg, tz);
    },

    // Lambda<4> : Store phi
    [=] (RowM m, IG g, ColZ z,
         shmem_phi_t& sh_phi,
         RowM tm, IG tg, ColZ tz)
    {
      phi(m, g, z) = sh_phi(tm, tg, tz);
    }

  );

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA column-major matrix shmem version of LTimes run time (sec.): "
            << timer.elapsed() <<", GFLOPS/sec: " << gflop_rate << std::endl;

#ifdef RAJA_ENABLE_VECTOR_STATS
  RAJA::tensor_stats::printVectorStats();
#endif

#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}


#endif
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP) && (VARIANT_RAJA_OPENMP)
{
  std::cout << "\n Running RAJA OpenMP version of LTimes...\n";

  std::memset(phi_data, 0, phi_size * sizeof(double));

  //
  // View types and Views/Layouts for indexing into arrays
  // 
  // L(m, d) : 1 -> d is stride-1 dimension 
  using LView = TypedView<double, Layout<2, int, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension 
  using PsiView = TypedView<double, Layout<3, int, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension 
  using PhiView = TypedView<double, Layout<3, int, 2>, IM, IG, IZ>;

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

  for (int iter = 0;iter < num_iter;++ iter)
  RAJA::kernel<EXECPOL>( segments,
    [=] (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA OpenMP version of LTimes run time (sec.): "
            << timer.elapsed() <<", GFLOPS/sec: " << gflop_rate << std::endl;


#if defined(DEBUG_LTIMES)
  checkResult(phi, L, psi, num_m, num_d, num_g, num_z);
#endif
}
#endif

//----------------------------------------------------------------------------//

#if VARIANT_CUDA_KERNEL
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
  using LView = TypedView<double, Layout<2, int, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension 
  using PsiView = TypedView<double, Layout<3, int, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension 
  using PhiView = TypedView<double, Layout<3, int, 2>, IM, IG, IZ>;

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

  for (int iter = 0;iter < num_iter;++ iter)
  RAJA::kernel<EXECPOL>( segments,
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );

  cudaErrchk( cudaDeviceSynchronize() );
  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA CUDA version of LTimes run time (sec.): "
            << timer.elapsed() <<", GFLOPS/sec: " << gflop_rate << std::endl;


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

#if VARIANT_CUDA_TEAMS
{
  std::cout << "\n Running RAJA CUDA Teams version of LTimes...\n";

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


  using pol_launch = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t, RAJA::expt::cuda_launch_t<true, 512> >;
  using pol_g = RAJA::expt::LoopPolicy<RAJA::loop_exec, cuda_block_x_loop>;
  using pol_z = RAJA::expt::LoopPolicy<RAJA::loop_exec, cuda_thread_y_loop>;
  using pol_m = RAJA::expt::LoopPolicy<RAJA::loop_exec, cuda_thread_x_loop>;
  using pol_d = RAJA::expt::LoopPolicy<RAJA::loop_exec, RAJA::loop_exec>;


  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(dL_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(dpsi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(dphi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));


  RAJA::Timer timer;
  cudaErrchk( cudaDeviceSynchronize() );
  timer.start();


  for (int iter = 0;iter < num_iter;++ iter){
    RAJA::expt::launch<pol_launch>(
        RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(160, 1, 1),
                              RAJA::expt::Threads(8, 64, 1)),
        [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx)
    {
      RAJA::expt::loop<pol_g>(ctx, RAJA::TypedRangeSegment<IG>(0, num_g), [&](IG g){
        RAJA::expt::loop<pol_z>(ctx, RAJA::TypedRangeSegment<IZ>(0, num_z), [&](IZ z){
          RAJA::expt::loop<pol_m>(ctx, RAJA::TypedRangeSegment<IM>(0, num_m), [&](IM m){

            double acc = phi(m, g, z);

            RAJA::expt::loop<pol_d>(ctx, RAJA::TypedRangeSegment<ID>(0, num_d), [&](ID d){

              acc += L(m, d) * psi(d, g, z);


            });

            phi(m,g,z) = acc;
          });
        });
      });

    });

  }
  cudaErrchk( cudaDeviceSynchronize() );

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA CUDA Teams version of LTimes run time (sec.): "
            << timer.elapsed() <<", GFLOPS/sec: " << gflop_rate << std::endl;


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

#if VARIANT_CUDA_TEAMS_MATRIX
{
  std::cout << "\n Running RAJA CUDA Teams+Matrix version of LTimes...\n";

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


  using matrix_t = RAJA::RegisterMatrix<double, RAJA::MATRIX_COL_MAJOR, RAJA::cuda_warp_register<4>>;

  using RowM = RAJA::RowIndex<IM, matrix_t>;
  using ColD = RAJA::ColIndex<ID, matrix_t>;
  using ColZ = RAJA::ColIndex<IZ, matrix_t>;

  using pol_launch = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t, RAJA::expt::cuda_launch_t<true , 512> >;
  using pol_g = RAJA::expt::LoopPolicy<RAJA::loop_exec, cuda_block_x_loop>;
  using pol_z = RAJA::expt::LoopPolicy<matrix_col_exec<matrix_t>, cuda_thread_y_matrix_col_loop<matrix_t> >;
  using pol_m = RAJA::expt::LoopPolicy<matrix_row_exec<matrix_t>, cuda_warp_matrix_row_loop<matrix_t> >;
  using pol_d = RAJA::expt::LoopPolicy<matrix_col_exec<matrix_t>, matrix_col_exec<matrix_t>>;


  //
  // View types and Views/Layouts for indexing into arrays
  //
  // L(m, d) : 1 -> d is stride-1 dimension
  using LView = TypedView<double, Layout<2, int, 0>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 0>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 0>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{1, 0}};
  LView L(dL_data,
          RAJA::make_permuted_layout({{num_m, num_d}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{1, 2, 0}};
  PsiView psi(dpsi_data,
              RAJA::make_permuted_layout({{num_d, num_g, num_z}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{1, 2, 0}};
  PhiView phi(dphi_data,
              RAJA::make_permuted_layout({{num_m, num_g, num_z}}, phi_perm));


  RAJA::Timer timer;
  cudaErrchk( cudaDeviceSynchronize() );
  timer.start();

  auto seg_g = RAJA::TypedRangeSegment<IG>(0, num_g);
  auto seg_z = RAJA::TypedRangeSegment<IZ>(0, num_z);
  auto seg_m = RAJA::TypedRangeSegment<IM>(0, num_m);
  auto seg_d = RAJA::TypedRangeSegment<ID>(0, num_d);

  for (int iter = 0;iter < num_iter;++ iter){
    RAJA::expt::launch<pol_launch>(
        RAJA::expt::DEVICE,
        RAJA::expt::Resources(RAJA::expt::Teams(num_g, 1, 1),
                              RAJA::expt::Threads(16, 32, 1)),
        [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx)
    {
      RAJA::expt::loop<pol_g>(ctx, seg_g, [&](IG g){
        RAJA::expt::loop<pol_z>(ctx, seg_z, [&](ColZ z){


          RAJA::expt::loop<pol_m>(ctx, seg_m, [&](RowM m){

            matrix_t acc = phi(m, g, z);

            RAJA::expt::loop<pol_d>(ctx, seg_d, [&](ColD d){

//              phi(m,g,z) += L(m, d) * psi(toRowIndex(d), g, z);
              acc += L(m, d) * psi(toRowIndex(d), g, z);
//              L(m, d).load().multiply_accumulate(psi(toRowIndex(d), g, z).load(), acc);


            });

            phi(m,g,z) = acc;
          });
        });
      });

    });

  }
  cudaErrchk( cudaDeviceSynchronize() );

  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA CUDA Teams+Matrix version of LTimes run time (sec.): "
            << timer.elapsed() <<", GFLOPS/sec: " << gflop_rate << std::endl;


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

#if VARIANT_CUDA_KERNEL_SHMEM
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
  using LView = TypedView<double, Layout<2, int, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 2>, IM, IG, IZ>;

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

  for (int iter = 0;iter < num_iter;++ iter)
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
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA CUDA + shmem version of LTimes run time (sec.): "
            << timer.elapsed() <<", GFLOPS/sec: " << gflop_rate << std::endl;



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

#if RAJA_HIP_KERNEL
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
  using LView = TypedView<double, Layout<2, int, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 2>, IM, IG, IZ>;

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

  for (int iter = 0;iter < num_iter;++ iter)
  RAJA::kernel<EXECPOL>( segments,
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );

  hipErrchk( hipDeviceSynchronize() );
  timer.stop();
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA HIP version of LTimes run time (sec.): "
            << timer.elapsed() <<", GFLOPS/sec: " << gflop_rate << std::endl;

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

#if RAJA_HIP_KERNEL_SHMEM
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
  using LView = TypedView<double, Layout<2, int, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension
  using PsiView = TypedView<double, Layout<3, int, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension
  using PhiView = TypedView<double, Layout<3, int, 2>, IM, IG, IZ>;

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

  for (int iter = 0;iter < num_iter;++ iter)
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
  double t = timer.elapsed();
  double gflop_rate = total_flops / t / 1.0e9;
  std::cout << "  RAJA HIP + shmem version of LTimes run time (sec.): "
            << timer.elapsed() <<", GFLOPS/sec: " << gflop_rate << std::endl;



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
                 const int num_m, 
                 const int num_d,
                 const int num_g,
                 const int num_z)
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
          printf("ERR: g=%d, z=%d, m=%d, val=%.12e, expected=%.12e\n",
              (int)*g, (int)*z, (int)*m, phi(m,g,z), total);
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
