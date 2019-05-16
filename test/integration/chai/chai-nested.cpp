//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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
 * Simple tests using nested::forall and View
 */
CUDA_TEST(Chai, NestedSimple)
{
  typedef RAJA::KernelPolicy<
      RAJA::statement::
          For<0, RAJA::seq_exec, RAJA::statement::For<1, RAJA::seq_exec>>>
      POLICY;

  using POLICY_GPU = RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::For<1, RAJA::cuda_block_x_loop,
        RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
          RAJA::statement::Lambda<0>
        >
      >
    >
  >;

  const int X = 16;
  const int Y = 16;

  chai::ManagedArray<float> v1(X * Y);
  chai::ManagedArray<float> v2(X * Y);

  RAJA::kernel<POLICY>(

      RAJA::make_tuple(RAJA::RangeSegment(0, Y), RAJA::RangeSegment(0, X)),

      [=](int i, int j) {
        int index = j * X + i;
        v1[index] = index;
      });

  RAJA::kernel<POLICY_GPU>(
      RAJA::make_tuple(RangeSegment(0, Y), RangeSegment(0, X)),

      [=] __host__ __device__(int i, int j) {
        int index = j * X + i;
        v2[index] = v1[index] * 2.0f;
      });

  cudaDeviceSynchronize();

  RAJA::kernel<POLICY>(

      RAJA::make_tuple(RAJA::RangeSegment(0, Y), RAJA::RangeSegment(0, X)),

      [=](int i, int j) {
        int index = j * X + i;
        ASSERT_FLOAT_EQ(v1[index], index * 1.0f);
        ASSERT_FLOAT_EQ(v2[index], index * 2.0f);
      });
}

CUDA_TEST(Chai, NestedView)
{
  typedef RAJA::KernelPolicy<
      RAJA::statement::
          For<0, RAJA::seq_exec, RAJA::statement::For<1, RAJA::seq_exec>>>
      POLICY;
  typedef RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
      RAJA::statement::For<1, RAJA::cuda_block_x_loop,
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop>>>>
      POLICY_GPU;

  const int X = 16;
  const int Y = 16;

  chai::ManagedArray<float> v1_array(X * Y);
  chai::ManagedArray<float> v2_array(X * Y);

  typedef RAJA::ManagedArrayView<float, RAJA::Layout<2>> view;

  view v1(v1_array, X, Y);
  view v2(v2_array, X, Y);

  RAJA::kernel<POLICY>(RAJA::make_tuple(RangeSegment(0, Y), RangeSegment(0, X)),
                        [=](int i, int j) { v1(i, j) = (i + (j * X)) * 1.0f; });

  RAJA::kernel<POLICY_GPU>(RAJA::make_tuple(RangeSegment(0, Y), RangeSegment(0, X)),
                            [=] __device__(int i, int j) {
                              v2(i, j) = v1(i, j) * 2.0f;
                            });

  RAJA::kernel<POLICY>(RAJA::make_tuple(RangeSegment(0, Y), RangeSegment(0, X)),
                        [=](int i, int j) {
                          ASSERT_FLOAT_EQ(v2(i, j), v1(i, j) * 2.0f);
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

RAJA_INDEX_VALUE_T(IM, int, "IM");
RAJA_INDEX_VALUE_T(ID, int, "ID");
RAJA_INDEX_VALUE_T(IG, int, "IG");
RAJA_INDEX_VALUE_T(IZ, int, "IZ");

void runLTimesTests(Index_type num_moments,
                   Index_type num_directions,
                   Index_type num_groups,
                   Index_type num_zones)
{
  // allocate data
  // phi is initialized to all zeros, the others are randomized
  chai::ManagedArray<double> L_data(num_moments * num_directions);
  chai::ManagedArray<double> psi_data(num_directions * num_groups * num_zones);
  chai::ManagedArray<double> phi_data(num_moments * num_groups * num_zones);

  RAJA::forall<RAJA::seq_exec>(
    RAJA::RangeSegment(0, (num_moments * num_directions)),
    [=](int i) {
      L_data[i] = i+2;
  });

  RAJA::forall<RAJA::seq_exec>(
    RAJA::RangeSegment(0, (num_directions * num_groups * num_zones)),
    [=](int i) { psi_data[i] = 2*i+1; });

  RAJA::forall<RAJA::seq_exec>(
    RAJA::RangeSegment(0, (num_moments * num_groups * num_zones)),
    [=](int i) { phi_data[i] = 0.0; });

  using LView = TypedManagedArrayView<double, Layout<2, Index_type, 1>, IM, ID>;

  // psi(d, g, z) : 2 -> z is stride-1 dimension 
  using PsiView = TypedManagedArrayView<double, Layout<3, Index_type, 2>, ID, IG, IZ>;

  // phi(m, g, z) : 2 -> z is stride-1 dimension 
  using PhiView = TypedManagedArrayView<double, Layout<3, Index_type, 2>, IM, IG, IZ>;

  std::array<RAJA::idx_t, 2> L_perm {{0, 1}};
  LView L(L_data,
          RAJA::make_permuted_layout({{num_moments, num_directions}}, L_perm));

  std::array<RAJA::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(psi_data,
              RAJA::make_permuted_layout({{num_directions, num_groups, num_zones}}, psi_perm));

  std::array<RAJA::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(phi_data,
              RAJA::make_permuted_layout({{num_moments, num_groups, num_zones}}, phi_perm));

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
                 
  auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<IM>(0, num_moments),
                                   RAJA::TypedRangeSegment<ID>(0, num_directions),
                                   RAJA::TypedRangeSegment<IG>(0, num_groups),
                                   RAJA::TypedRangeSegment<IZ>(0, num_zones));

  cudaErrchk( cudaDeviceSynchronize() );

  RAJA::kernel<EXECPOL>( segments,
    [=] RAJA_DEVICE (IM m, ID d, IG g, IZ z) {
       phi(m, g, z) += L(m, d) * psi(d, g, z);
    }
  );

  cudaErrchk( cudaDeviceSynchronize() );

  RAJA::forall<RAJA::seq_exec>(
    RAJA::TypedRangeSegment<IM>(0, num_moments), [=] (IM m) {
    for (IG g(0); g < num_groups; ++g) {
      for (IZ z(0); z < num_zones; ++z) {
        double total = 0.0;
        for (ID d(0); d < num_directions; ++d) {
          double val = L(m, d) * psi(d, g, z);
          total += val;
        }
        ASSERT_FLOAT_EQ(total, phi(m, g, z));
      }
    }
  });

  //rm->setExecutionSpace(chai::NONE);
}

TEST(Chai, LTimes)
{
  //  runLTimesTests(2, 0, 7, 3);
  runLTimesTests(2, 3, 7, 3);
  runLTimesTests(2, 3, 32, 4);
  runLTimesTests(25, 96, 8, 32);
  runLTimesTests(100, 15, 7, 13);
}
