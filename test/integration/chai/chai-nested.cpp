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
/// Source file containing tests for CHAI in RAJA nested loops.
///
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
using namespace std;

#include "chai/ArrayManager.hpp"
#include "chai/ManagedArray.hpp"

/*
 * Simple tests using forallN and View
 */
CUDA_TEST(Chai, NestedSimpleOld) {
  typedef RAJA::NestedPolicy< RAJA::ExecList< RAJA::seq_exec, RAJA::seq_exec> > POLICY;
  typedef RAJA::NestedPolicy< RAJA::ExecList< RAJA::seq_exec, RAJA::cuda_thread_y_exec > > POLICY_GPU;

  const int X = 16;
  const int Y = 16;

  chai::ManagedArray<float> v1(X*Y);
  chai::ManagedArray<float> v2(X*Y);

  RAJA::forallN<POLICY>(RangeSegment(0,Y), RangeSegment(0,X), [=] (int i, int j) {
      int index = j*X + i;
      v1[index] = index;
  });

  RAJA::forallN<POLICY_GPU>(RangeSegment(0,Y), RangeSegment(0,X), [=] __device__ (int i, int j) {
      int index = j*X + i;
      v2[index] = v1[index]*2.0f;
  });
  cudaDeviceSynchronize();

  RAJA::forallN<POLICY>(RangeSegment(0,Y), RangeSegment(0,X), [=] (int i, int j) {
      int index = j*X + i;
      ASSERT_FLOAT_EQ(v1[index], index*1.0f);
      ASSERT_FLOAT_EQ(v2[index], index*2.0f);
  });
}


/*
 * Simple tests using nested::forall and View
 */
CUDA_TEST(Chai, NestedSimple) {
  typedef RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::seq_exec,
        RAJA::statement::For<1, RAJA::seq_exec> > > POLICY;
  typedef RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::seq_exec,
        RAJA::statement::CudaKernel<
          RAJA::statement::For<1, RAJA::cuda_threadblock_exec<32> > > > >POLICY_GPU;

  const int X = 16;
  const int Y = 16;

  chai::ManagedArray<float> v1(X*Y);
  chai::ManagedArray<float> v2(X*Y);

  RAJA::kernel<POLICY>(

      RAJA::make_tuple(RAJA::RangeSegment(0,Y), RAJA::RangeSegment(0,X) ),

      [=] (int i, int j) {
        int index = j*X + i;
        v1[index] = index;
  });

  RAJA::kernel<POLICY_GPU>(

      RAJA::make_tuple(RangeSegment(0,Y), RangeSegment(0,X) ),

      [=] __host__ __device__ (int i, int j) {
        int index = j*X + i;
        v2[index] = v1[index]*2.0f;
  });

  cudaDeviceSynchronize();

  RAJA::kernel<POLICY>(

      RAJA::make_tuple(RAJA::RangeSegment(0,Y), RAJA::RangeSegment(0,X) ),

      [=] (int i, int j) {
        int index = j*X + i;
        ASSERT_FLOAT_EQ(v1[index], index*1.0f);
        ASSERT_FLOAT_EQ(v2[index], index*2.0f);
  });
}

CUDA_TEST(Chai, NestedView) {
  typedef RAJA::NestedPolicy< RAJA::ExecList< RAJA::seq_exec, RAJA::seq_exec> > POLICY;
  typedef RAJA::NestedPolicy< RAJA::ExecList< RAJA::seq_exec, RAJA::cuda_thread_y_exec > > POLICY_GPU;

  const int X = 16;
  const int Y = 16;

  chai::ManagedArray<float> v1_array(X*Y);
  chai::ManagedArray<float> v2_array(X*Y);

  typedef RAJA::ManagedArrayView<float, RAJA::Layout<2> > view;

  view v1(v1_array, X, Y);
  view v2(v2_array, X, Y);

  RAJA::forallN<POLICY>(RangeSegment(0,Y), RangeSegment(0,X), [=] (int i, int j) {
      v1(i,j) = (i+(j*X)) * 1.0f;
  });

  RAJA::forallN<POLICY_GPU>(RangeSegment(0,Y), RangeSegment(0,X), [=] __device__ (int i, int j) {
      v2(i,j) = v1(i,j)*2.0f;
  });

  RAJA::forallN<POLICY>(RangeSegment(0,Y), RangeSegment(0,X), [=] (int i, int j) {
      ASSERT_FLOAT_EQ(v2(i,j), v1(i,j)*2.0f);
  });
}

CUDA_TEST(Chai, NestedView2) {
  typedef RAJA::NestedPolicy< RAJA::ExecList< RAJA::seq_exec, RAJA::seq_exec> > POLICY;

#ifdef RAJA_ENABLE_OPENMP
  typedef RAJA::NestedPolicy< RAJA::ExecList< RAJA::omp_for_nowait_exec, RAJA::cuda_thread_x_exec >, RAJA::OMP_Parallel<> > POLICY_GPU;
#else
  typedef RAJA::NestedPolicy< RAJA::ExecList< RAJA::seq_exec, RAJA::cuda_thread_x_exec > > POLICY_GPU;
#endif

  const int X = 16;
  const int Y = 16;

  chai::ManagedArray<float> v1_array(X*Y);
  chai::ManagedArray<float> v2_array(X*Y);

  typedef RAJA::ManagedArrayView<float, RAJA::Layout<2> > view;

  view v1(v1_array, X, Y);
  view v2(v2_array, X, Y);

  RAJA::forallN<POLICY>(RangeSegment(0,Y), RangeSegment(0,X), [=] (int i, int j) {
      v1(i,j) = (i+(j*X)) * 1.0f;
  });

  RAJA::forallN<POLICY_GPU>(RangeSegment(0,Y), RangeSegment(0,X), [=] __device__ (int i, int j) {
      v2(i,j) = v1(i,j)*2.0f;
  });

  RAJA::forallN<POLICY>(RangeSegment(0,Y), RangeSegment(0,X), [=] (int i, int j) {
      ASSERT_FLOAT_EQ(v2(i,j), v1(i,j)*2.0f);
  });
}

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
typedef struct {
  double val;
  int idx;
} minmaxloc_t;

// block_size is needed by the reduction variables to setup shared memory
// Care should be used here to cover the maximum block dimensions used by this
// test
const size_t block_size = 256;

RAJA_INDEX_VALUE(IMoment, "IMoment");
RAJA_INDEX_VALUE(IDirection, "IDirection");
RAJA_INDEX_VALUE(IGroup, "IGroup");
RAJA_INDEX_VALUE(IZone, "IZone");

template <typename POL>
void runLTimesTest(std::string const &policy,
                   Index_type num_moments,
                   Index_type num_directions,
                   Index_type num_groups,
                   Index_type num_zones)
{
  // cout << "\n TestLTimes " << num_moments << " moments, " << num_directions
  //      << " directions, " << num_groups << " groups, and " << num_zones
  //      << " zones"
  //      << " with policy " << policy << endl;

  // allocate data
  // phi is initialized to all zeros, the others are randomized
  chai::ManagedArray<double> ell_data(num_moments * num_directions);
  chai::ManagedArray<double> psi_data(num_directions * num_groups * num_zones);
  //chai::ManagedArray<double> phi_data(num_moments * num_groups * num_zones, 0.0);
  chai::ManagedArray<double> phi_data(num_moments * num_groups * num_zones);

  // setup CUDA Reduction variables to be exercised
  ReduceSum<cuda_reduce<block_size>, double> pdsum(0.0);
  ReduceMin<cuda_reduce<block_size>, double> pdmin(DBL_MAX);
  ReduceMax<cuda_reduce<block_size>, double> pdmax(-DBL_MAX);
  ReduceMinLoc<cuda_reduce<block_size>, double> pdminloc(DBL_MAX, -1);
  ReduceMaxLoc<cuda_reduce<block_size>, double> pdmaxloc(-DBL_MAX, -1);


  // data setup using RAJA to ensure that chai is activated
  RAJA::forall<RAJA::seq_exec>(0, (num_moments*num_directions), [=] (int i) {
    ell_data[i] = drand48();
  });

  RAJA::forall<RAJA::seq_exec>(0, (num_directions*num_groups*num_zones), [=] (int i) {
    psi_data[i] = drand48();
  });

  RAJA::forall<RAJA::seq_exec>(0, (num_moments*num_groups*num_zones), [=] (int i) {
    phi_data[i] = 0.0;
  });

  typename POL::ELL_VIEW ell(ell_data, RAJA::make_permuted_layout({num_moments, num_directions}, RAJA::as_array<typename POL::ELL_PERM>::get()));
  typename POL::PSI_VIEW psi(psi_data, RAJA::make_permuted_layout({num_directions, num_groups, num_zones}, RAJA::as_array<typename POL::PSI_PERM>::get()));
  typename POL::PHI_VIEW phi(phi_data, RAJA::make_permuted_layout({num_moments, num_groups, num_zones}, RAJA::as_array<typename POL::PHI_PERM>::get()));

  using EXEC = typename POL::EXEC;

  // do calculation using RAJA
  forallN<EXEC, IMoment, IDirection, IGroup, IZone>(
      RangeSegment(0, num_moments),
      RangeSegment(0, num_directions),
      RangeSegment(0, num_groups),
      RangeSegment(0, num_zones),
      [=] __device__(IMoment m, IDirection d, IGroup g, IZone z) {
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

  // Make sure data is copied to host for checking results.
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  rm->setExecutionSpace(chai::CPU);
  // setup local Reduction variables as a crosscheck
  double the_lsum = 0.0;
  double the_lmin = DBL_MAX;
  double the_lmax = -DBL_MAX;

  double* lsum = &the_lsum;
  double* lmin = &the_lmin;
  double* lmax = &the_lmax;

  forall<RAJA::seq_exec>(RangeSegment(0, num_zones), [=] (int z) {
    for (IGroup g(0); g < num_groups; ++g) {
      for (IMoment m(0); m < num_moments; ++m) {
        double total = 0.0;
        for (IDirection d(0); d < num_directions; ++d) {
          double val = ell(m, d) * psi(d, g, IZone(z));
          total += val;
          *lmin = RAJA_MIN(*lmin, val);
          *lmax = RAJA_MAX(*lmax, val);
          int index = *d + (*m * num_directions)
                      + (*g * num_directions * num_moments)
                      + (z * num_directions * num_moments * num_groups);
        }
        *lsum += total;

        // check answer with some reasonable tolerance
        ASSERT_FLOAT_EQ(total, phi(m, g, IZone(z)));
      }
    }
  });

  rm->setExecutionSpace(chai::NONE);

  ASSERT_FLOAT_EQ(*lsum, pdsum.get());
  ASSERT_FLOAT_EQ(*lmin, pdmin.get());
  ASSERT_FLOAT_EQ(*lmax, pdmax.get());
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
  typedef RAJA::TypedManagedArrayView<double, RAJA::Layout<3>, IDirection, IGroup, IZone>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::TypedManagedArrayView<double, RAJA::Layout<3>, IMoment, IGroup, IZone>
      PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::TypedManagedArrayView<double, RAJA::Layout<2>, IMoment, IDirection>
      ELL_VIEW;

  typedef RAJA::PERM_IJK PSI_PERM;
  typedef RAJA::PERM_IJK PHI_PERM;
  typedef RAJA::PERM_IJ ELL_PERM;
};

// Use thread and block mappings
struct PolLTimesB_GPU {
  // Loops: Moments, Directions, Groups, Zones
  typedef NestedPolicy<ExecList<seq_exec,
                                seq_exec,
                                cuda_thread_z_exec,
                                cuda_block_y_exec>,
                       Permute<PERM_IJKL>>
      EXEC;

  // psi[direction, group, zone]
  typedef RAJA::TypedManagedArrayView<double, RAJA::Layout<3>, IDirection, IGroup, IZone>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::TypedManagedArrayView<double, RAJA::Layout<3>, IMoment, IGroup, IZone>
      PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::TypedManagedArrayView<double, RAJA::Layout<2>, IMoment, IDirection>
      ELL_VIEW;

  typedef RAJA::PERM_IJK PSI_PERM;
  typedef RAJA::PERM_IJK PHI_PERM;
  typedef RAJA::PERM_IJ ELL_PERM;
};

// Combine OMP Parallel, omp nowait, and cuda thread-block launch
struct PolLTimesC_GPU {
  // Loops: Moments, Directions, Groups, Zones
#ifdef RAJA_ENABLE_OPENMP
  typedef NestedPolicy<ExecList<seq_exec,
                                seq_exec,
                                omp_for_nowait_exec,
                                cuda_threadblock_y_exec<32>>,
                       OMP_Parallel<>>
      EXEC;
#else
  typedef NestedPolicy<ExecList<seq_exec,
                                seq_exec,
                                seq_exec,
                                cuda_threadblock_y_exec<32>> >
      EXEC;
#endif

  // psi[direction, group, zone]
  typedef RAJA::TypedManagedArrayView<double, RAJA::Layout<3>, IDirection, IGroup, IZone>
      PSI_VIEW;

  // phi[moment, group, zone]
  typedef RAJA::TypedManagedArrayView<double, RAJA::Layout<3>, IMoment, IGroup, IZone>
      PHI_VIEW;

  // ell[moment, direction]
  typedef RAJA::TypedManagedArrayView<double, RAJA::Layout<2>, IMoment, IDirection>
      ELL_VIEW;

  typedef RAJA::PERM_IJK PSI_PERM;
  typedef RAJA::PERM_IJK PHI_PERM;
  typedef RAJA::PERM_IJ ELL_PERM;
};

void runLTimesTests(Index_type num_moments,
                    Index_type num_directions,
                    Index_type num_groups,
                    Index_type num_zones)
{
  runLTimesTest<PolLTimesA_GPU>(
      "PolLTimesA_GPU", num_moments, num_directions, num_groups, num_zones);
  runLTimesTest<PolLTimesB_GPU>(
      "PolLTimesB_GPU", num_moments, num_directions, num_groups, num_zones);
  runLTimesTest<PolLTimesC_GPU>(
      "PolLTimesC_GPU", num_moments, num_directions, num_groups, num_zones);
}

TEST(Chai, Nested) {
//  runLTimesTests(2, 0, 7, 3);
  runLTimesTests(2, 3, 7, 3);
  runLTimesTests(2, 3, 32, 4);
  runLTimesTests(25, 96, 8, 32);
  runLTimesTests(100, 15, 7, 13);
}
