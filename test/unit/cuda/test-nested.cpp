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

///
/// Source file containing tests for RAJA GPU nested loop kernels.
///

#include <time.h>
#include <cfloat>
#include <cstdlib>

#include <iostream>
#include <string>
#include <vector>

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

using namespace RAJA;

// block_size is needed by the reduction variables to setup shared memory
// Care should be used here to cover the maximum block dimensions used by this
// test
static const size_t block_size = 256;

///////////////////////////////////////////////////////////////////////////
//
// Example LTimes kernel test routines
//
// Demonstrates a 4-nested loop, the use of complex nested policies and
// the use of strongly-typed indices
//
// This routine computes phi(m, g, z) = SUM_d {  ell(m, d)*psi(d,g,z)  }
//
///////////////////////////////////////////////////////////////////////////

RAJA_INDEX_VALUE(IMoment, "IMoment");
RAJA_INDEX_VALUE(IDirection, "IDirection");
RAJA_INDEX_VALUE(IGroup, "IGroup");
RAJA_INDEX_VALUE(IZone, "IZone");

template <typename POL>
static void runLTimesTest(Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);

  // setup CUDA Reduction variables to be exercised
  ReduceSum<cuda_reduce<block_size>, double> pdsum(0.0);
  ReduceMin<cuda_reduce<block_size>, double> pdmin(DBL_MAX);
  ReduceMax<cuda_reduce<block_size>, double> pdmax(-DBL_MAX);
  ReduceMinLoc<cuda_reduce<block_size>, double> pdminloc(DBL_MAX, -1);
  ReduceMaxLoc<cuda_reduce<block_size>, double> pdmaxloc(-DBL_MAX, -1);

  // setup local Reduction variables as a crosscheck
  double lsum = 0.0;
  double lmin = DBL_MAX;
  double lmax = -DBL_MAX;
  ReduceMinLoc<seq_reduce, double> lminloc(lmin);
  ReduceMaxLoc<seq_reduce, double> lmaxloc(lmax);

  //
  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = drand48();
    // ell_data[i] = 0.0;
  }
  // ell_data[0] = 2.0;

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = drand48();
    // psi_data[i] = 0.0;
  }
  // psi_data[0] = 5.0;
  // create device memory
  double *d_ell, *d_phi, *d_psi;
  cudaErrchk(cudaMalloc(&d_ell, sizeof(double) * ell_data.size()));
  cudaErrchk(cudaMalloc(&d_phi, sizeof(double) * phi_data.size()));
  cudaErrchk(cudaMalloc(&d_psi, sizeof(double) * psi_data.size()));

  // Copy to device
  cudaMemcpy(d_ell,
             &ell_data[0],
             sizeof(double) * ell_data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_phi,
             &phi_data[0],
             sizeof(double) * phi_data.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_psi,
             &psi_data[0],
             sizeof(double) * psi_data.size(),
             cudaMemcpyHostToDevice);

  // create views on data
  typename POL::ELL_VIEW ell(
      d_ell,
      make_permuted_layout({num_moments, num_directions},
                           as_array<typename POL::ELL_PERM>::get()));
  typename POL::PSI_VIEW psi(
      d_psi,
      make_permuted_layout({num_directions, num_groups, num_zones},
                           as_array<typename POL::PSI_PERM>::get()));
  typename POL::PHI_VIEW phi(
      d_phi,
      make_permuted_layout({num_moments, num_groups, num_zones},
                           as_array<typename POL::PHI_PERM>::get()));

  // get execution policy
  using EXEC = typename POL::EXEC;

  // do calculation using RAJA
  forallN<EXEC, IMoment, IDirection, IGroup, IZone>(
      RangeSegment(0, num_moments),
      RangeSegment(0, num_directions),
      RangeSegment(0, num_groups),
      RangeSegment(0, num_zones),
      [=] __device__(IMoment m, IDirection d, IGroup g, IZone z) {
        // printf("%d,%d,%d,%d\n", *m, *d, *g, *z);
        double val = ell(m, d) * psi(d, g, z);
        phi(m, g, z) += val;
        pdsum += val;
        pdmin.min(val);
        pdmax.max(val);
        int index = *d + (*m * num_directions)
                    + (*g * num_directions * num_moments)
                    + (*z * num_directions * num_moments * num_groups);
        pdminloc.minloc(val, index);
        pdmaxloc.maxloc(val, index);
      });

  cudaDeviceSynchronize();
  // Copy to host the result
  cudaMemcpy(&phi_data[0],
             d_phi,
             sizeof(double) * phi_data.size(),
             cudaMemcpyDeviceToHost);

  // Free CUDA memory
  cudaFree(d_ell);
  cudaFree(d_phi);
  cudaFree(d_psi);

  // swap to host pointers
  ell.set_data(&ell_data[0]);
  phi.set_data(&phi_data[0]);
  psi.set_data(&psi_data[0]);
  for (IZone z(0); z < num_zones; ++z) {
    for (IGroup g(0); g < num_groups; ++g) {
      for (IMoment m(0); m < num_moments; ++m) {
        double total = 0.0;
        for (IDirection d(0); d < num_directions; ++d) {
          double val = ell(m, d) * psi(d, g, z);
          total += val;
          lmin = RAJA_MIN(lmin, val);
          lmax = RAJA_MAX(lmax, val);
          int index = *d + (*m * num_directions)
                      + (*g * num_directions * num_moments)
                      + (*z * num_directions * num_moments * num_groups);
          lminloc.minloc(val, index);
          lmaxloc.maxloc(val, index);
        }
        lsum += total;
        ASSERT_FLOAT_EQ(total, phi(m, g, z));
      }
    }
  }

  ASSERT_FLOAT_EQ(lsum, pdsum.get());
  ASSERT_FLOAT_EQ(lmin, pdmin.get());
  ASSERT_FLOAT_EQ(lmax, pdmax.get());
  ASSERT_FLOAT_EQ(lminloc.get(), pdminloc.get());
  ASSERT_FLOAT_EQ(lmaxloc.get(), pdmaxloc.get());
  ASSERT_EQ(lminloc.getLoc(), pdminloc.getLoc());
  ASSERT_EQ(lmaxloc.getLoc(), pdmaxloc.getLoc());
}

// Use thread-block mappings
struct PolLTimesA_GPU {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec,
                                seq_exec,
                                cuda_threadblock_x_exec<32>,
                                cuda_threadblock_y_exec<32>>>
      EXEC;

  // psi[direction, group, zone]
  typedef RAJA::TypedView<double, Layout<3>, IDirection, IGroup, IZone>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::TypedView<double, Layout<3>, IMoment, IGroup, IZone> PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::TypedView<double, Layout<2>, IMoment, IDirection> ELL_VIEW;

  typedef RAJA::PERM_IJK PSI_PERM;
  typedef RAJA::PERM_IJK PHI_PERM;
  typedef RAJA::PERM_IJ ELL_PERM;
};

// Use thread and block mappings
struct PolLTimesB_GPU {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<
      ExecList<seq_exec, seq_exec, cuda_thread_z_exec, cuda_block_y_exec>,
      Permute<PERM_IJKL>>
      EXEC;

  // psi[direction, group, zone]
  typedef RAJA::TypedView<double, Layout<3>, IDirection, IGroup, IZone>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::TypedView<double, Layout<3>, IMoment, IGroup, IZone> PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::TypedView<double, Layout<2>, IMoment, IDirection> ELL_VIEW;

  typedef RAJA::PERM_IJK PSI_PERM;
  typedef RAJA::PERM_IJK PHI_PERM;
  typedef RAJA::PERM_IJ ELL_PERM;
};

// Combine OMP Parallel, omp nowait, and cuda thread-block launch
#if defined(RAJA_ENABLE_OPENMP)
struct PolLTimesC_GPU {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec,
                                seq_exec,
                                omp_for_nowait_exec,
                                cuda_threadblock_y_exec<32>>,
                       OMP_Parallel<>>
      EXEC;

  // psi[direction, group, zone]
  typedef RAJA::TypedView<double, Layout<3>, IDirection, IGroup, IZone>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::TypedView<double, Layout<3>, IMoment, IGroup, IZone> PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::TypedView<double, Layout<2>, IMoment, IDirection> ELL_VIEW;

  typedef RAJA::PERM_IJK PSI_PERM;
  typedef RAJA::PERM_IJK PHI_PERM;
  typedef RAJA::PERM_IJ ELL_PERM;
};
#endif

// Combine TBB parallel loop, and cuda thread-block launch
#if defined(RAJA_ENABLE_TBB)
struct PolLTimesD_GPU {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec,
                                seq_exec,
                                tbb_for_exec,
                                cuda_threadblock_y_exec<32>>,
                       Permute<PERM_IJKL>>
      EXEC;

  // psi[direction, group, zone]
  typedef RAJA::TypedView<double, Layout<3>, IDirection, IGroup, IZone>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::TypedView<double, Layout<3>, IMoment, IGroup, IZone> PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::TypedView<double, Layout<2>, IMoment, IDirection> ELL_VIEW;

  typedef RAJA::PERM_IJK PSI_PERM;
  typedef RAJA::PERM_IJK PHI_PERM;
  typedef RAJA::PERM_IJ ELL_PERM;
};
#endif

using LTimesPolicies = ::testing::Types<PolLTimesA_GPU,
                                        PolLTimesB_GPU
#if defined(RAJA_ENABLE_OPENMP)
                                        ,PolLTimesC_GPU
#endif
#if defined(RAJA_ENABLE_TBB)
                                        ,PolLTimesD_GPU
#endif
                                        >;

template <typename POL>
class NestedCUDA : public ::testing::Test
{
public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TYPED_TEST_CASE_P(NestedCUDA);

CUDA_TYPED_TEST_P(NestedCUDA, LTimes)
{
  runLTimesTest<TypeParam>(2, 0, 7, 3);
  runLTimesTest<TypeParam>(2, 3, 7, 3);
  runLTimesTest<TypeParam>(2, 3, 32, 4);
  runLTimesTest<TypeParam>(25, 96, 8, 32);
  runLTimesTest<TypeParam>(100, 15, 7, 13);
}

REGISTER_TYPED_TEST_CASE_P(NestedCUDA, LTimes);

INSTANTIATE_TYPED_TEST_CASE_P(LTimes, NestedCUDA, LTimesPolicies);

CUDA_TEST(NestedCUDA, NegativeRange)
{
  double *data;
  double host_data[100];

  cudaMallocManaged((void **)&data, sizeof(double) * 100, cudaMemAttachGlobal);

  for (int i = 0; i < 100; ++i) {
    host_data[i] = i * 1.0;
  }

  forallN<NestedPolicy<
      ExecList<cuda_threadblock_y_exec<16>, cuda_threadblock_x_exec<16>>>>(
      RangeSegment(-2, 8), RangeSegment(-2, 8), [=] RAJA_DEVICE(int k, int j) {
        const int idx = ((k - -2) * 10) + (j - -2);
        data[idx] = idx * 1.0;
      });

  cudaDeviceSynchronize();

  for (int i = 0; i < 100; ++i) {
    ASSERT_EQ(host_data[i], data[i]);
  }
}

CUDA_TEST(NestedCUDA, PositiveRange)
{
  double *data;
  double host_data[100];

  cudaMallocManaged((void **)&data, sizeof(double) * 100, cudaMemAttachGlobal);

  for (int i = 0; i < 100; ++i) {
    host_data[i] = i * 1.0;
  }

  forallN<NestedPolicy<
      ExecList<cuda_threadblock_y_exec<16>, cuda_threadblock_x_exec<16>>>>(
      RangeSegment(2, 12), RangeSegment(2, 12), [=] RAJA_DEVICE(int k, int j) {
        const int idx = ((k - 2) * 10) + (j - 2);
        data[idx] = idx * 1.0;
      });

  cudaDeviceSynchronize();

  for (int i = 0; i < 100; ++i) {
    ASSERT_EQ(host_data[i], data[i]);
  }
}
