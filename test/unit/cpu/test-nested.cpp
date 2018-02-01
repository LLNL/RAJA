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
/// Source file containing tests for RAJA nested-loop execution
///

#include <time.h>
#include <cmath>
#include <cstdlib>

#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "RAJA/RAJA.hpp"

using namespace RAJA;

#include "type_helper.hpp"

template <typename Perm>
struct PermuteOnly {
  using VIEW = View<Index_type, Layout<2>>;
  using PERM = Perm;
  using TILE = Permute<PERM>;
};

template <typename Perm>
struct SingleTile {
  using VIEW = View<Index_type, Layout<2>>;
  using PERM = Perm;
  using TILE = Tile<TileList<tile_fixed<8>, tile_fixed<16>>,
                    typename PermuteOnly<PERM>::TILE>;
};

template <typename Perm>
struct NestedTile {
  using VIEW = View<Index_type, Layout<2>>;
  using PERM = Perm;
  using TILE = Tile<TileList<tile_fixed<32>, tile_fixed<32>>,
                    typename SingleTile<PERM>::TILE>;
};

template <typename Transform, typename... Exec>
struct ExecInfo : public Transform {
  using EXEC = NestedPolicy<ExecList<Exec...>, typename Transform::TILE>;
};

#if defined(RAJA_ENABLE_OPENMP)
template <typename Transform, typename... Exec>
struct OMPExecInfo : public Transform {
  using EXEC =
      NestedPolicy<ExecList<Exec...>, OMP_Parallel<typename Transform::TILE>>;
};
#endif

using PERMS = std::tuple<PERM_IJ, PERM_JI>;

template <typename PERM>
using TRANSFORMS =
    std::tuple<PermuteOnly<PERM>, SingleTile<PERM>, NestedTile<PERM>>;

template <typename TRANSFORMS>
using POLICIES =
    std::tuple<ExecInfo<TRANSFORMS, seq_exec, seq_exec>
               ,ExecInfo<TRANSFORMS, seq_exec, loop_exec>
               ,ExecInfo<TRANSFORMS, loop_exec, loop_exec>
#if defined(RAJA_ENABLE_OPENMP)
               ,ExecInfo<TRANSFORMS, seq_exec, omp_parallel_for_exec>
               ,OMPExecInfo<TRANSFORMS, loop_exec, omp_for_nowait_exec>
#endif
#if defined(RAJA_ENABLE_TBB)
               ,ExecInfo<TRANSFORMS, loop_exec, tbb_for_exec>
#endif
               >;


using InstPolicies =
    ForTesting<tt::apply_t<POLICIES, tt::apply_t<TRANSFORMS, PERMS>>>;

template <typename POL>
class NestedTest : public ::testing::Test
{
public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TYPED_TEST_CASE_P(NestedTest);

TYPED_TEST_P(NestedTest, Nested2DTest)
{
  using POL = TypeParam;
  using Policy = typename POL::EXEC;
  using View = typename POL::VIEW;

  using Pair = std::pair<Index_type, Index_type>;

  for (auto size : {Pair(63, 255), Pair(37, 1)}) {

    Index_type size_i = std::get<0>(size);
    Index_type size_j = std::get<1>(size);

    std::vector<Index_type> v(size_i * size_j, 1);
    View view(v.data(),
              make_permuted_layout({{size_i, size_j}},
                                   RAJA::as_array<typename POL::PERM>::get()));

    forallN<Policy>(RangeSegment(1, size_i),
                    RangeSegment(0, size_j),
                    [=](Index_type i, Index_type j) {
                      view(0, j) += view(i, j);
                    });

    for (Index_type i = 0; i < size_i; ++i)
      for (Index_type j = 0; j < size_j; ++j)
        ASSERT_EQ((i == 0) ? size_i : 1, view(i, j));
  }
}

REGISTER_TYPED_TEST_CASE_P(NestedTest, Nested2DTest);

INSTANTIATE_TYPED_TEST_CASE_P(Nested2D, NestedTest, InstPolicies);

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

struct PolLTimesCommon {
  // psi[direction, group, zone]
  using PSI_VIEW = TypedView<double, Layout<3>, IDirection, IGroup, IZone>;
  // phi[moment, group, zone]
  using PHI_VIEW = TypedView<double, Layout<3>, IMoment, IGroup, IZone>;
  // ell[moment, direction]
  using ELL_VIEW = TypedView<double, Layout<2>, IMoment, IDirection>;
};

// Sequential
struct PolLTimesA : PolLTimesCommon {
  // Loops: Moments, Directions, Groups, Zones
  using EXEC = NestedPolicy<ExecList<seq_exec, seq_exec, seq_exec, seq_exec>>;
  using PSI_PERM = PERM_IJK;
  using PHI_PERM = PERM_IJK;
  using ELL_PERM = PERM_IJ;
};

// Sequential, reversed permutation
struct PolLTimesB : PolLTimesCommon {
  // Loops: Moments, Directions, Groups, Zones
  using EXEC = NestedPolicy<ExecList<seq_exec, seq_exec, seq_exec, seq_exec>,
                            Permute<PERM_LKJI>>;
  using PSI_PERM = PERM_KJI;
  using PHI_PERM = PERM_IJK;
  using ELL_PERM = PERM_JI;
};

// Sequential, Tiled, another permutation
struct PolLTimesC : PolLTimesCommon {
  // Loops: Moments, Directions, Groups, Zones
  using EXEC = NestedPolicy<
      ExecList<seq_exec, seq_exec, seq_exec, seq_exec>,
      Tile<TileList<tile_none, tile_none, tile_fixed<64>, tile_fixed<64>>,
           Permute<PERM_JKIL>>>;
  using PSI_PERM = PERM_IJK;
  using PHI_PERM = PERM_KJI;
  using ELL_PERM = PERM_IJ;
};

#ifdef RAJA_ENABLE_OPENMP

// Parallel on zones,  loop nesting: Zones, Groups, Moments, Directions
struct PolLTimesD_OMP : PolLTimesCommon {
  // Loops: Moments, Directions, Groups, Zones
  using EXEC =
      NestedPolicy<ExecList<seq_exec, seq_exec, seq_exec, omp_for_nowait_exec>,
                   OMP_Parallel<Permute<PERM_LKIJ>>>;
  using PSI_PERM = PERM_KJI;
  using PHI_PERM = PERM_KJI;
  using ELL_PERM = PERM_IJ;
};

// Same as D, but with tiling on zones and omp collapse on groups and zones
struct PolLTimesE_OMP : PolLTimesCommon {
  // Loops: Moments, Directions, Groups, Zones
  using EXEC = NestedPolicy<
      ExecList<seq_exec,
               seq_exec,
               omp_collapse_nowait_exec,
               omp_collapse_nowait_exec>,
      OMP_Parallel<
          Tile<TileList<tile_none, tile_none, tile_none, tile_fixed<16>>,
               Permute<PERM_LKIJ,
                       Execute  // implicit
                       >>>>;
  using PSI_PERM = PERM_KJI;
  using PHI_PERM = PERM_KJI;
  using ELL_PERM = PERM_IJ;
};

#endif

#ifdef RAJA_ENABLE_TBB

// Parallel on zones,  loop nesting: Zones, Groups, Moments, Directions
struct PolLTimesF_TBB : PolLTimesCommon {
  // Loops: Moments, Directions, Groups, Zones
  using EXEC =
      NestedPolicy<ExecList<seq_exec, seq_exec, seq_exec, tbb_for_exec>,
                   Permute<PERM_LKIJ>>;
  using PSI_PERM = PERM_KJI;
  using PHI_PERM = PERM_KJI;
  using ELL_PERM = PERM_IJ;
};

// Parallel on zones,  loop nesting: Zones, Groups, Moments, Directions
struct PolLTimesG_TBB : PolLTimesCommon {
  // Loops: Moments, Directions, Groups, Zones
  using EXEC =
      NestedPolicy<ExecList<seq_exec, seq_exec, seq_exec, tbb_for_dynamic>,
                   Permute<PERM_LKIJ>>;
  using PSI_PERM = PERM_KJI;
  using PHI_PERM = PERM_KJI;
  using ELL_PERM = PERM_IJ;
};

#endif

using LTimesPolicies = ::testing::Types<PolLTimesA,
                                        PolLTimesB,
                                        PolLTimesC
#if defined(RAJA_ENABLE_OPENMP)
                                        ,
                                        PolLTimesD_OMP,
                                        PolLTimesE_OMP
#endif
#if defined(RAJA_ENABLE_TBB)
                                        ,
                                        PolLTimesF_TBB,
                                        PolLTimesG_TBB
#endif
                                        >;

template <typename POL>
class LTimesTest : public ::testing::Test
{
public:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TYPED_TEST_CASE_P(LTimesTest);

TYPED_TEST_P(LTimesTest, LTimesNestedTest)
{
  using Args = std::array<Index_type, 4>;
  using POL = TypeParam;

  for (auto versions : {Args{{25, 96, 8, 32}}, Args{{100, 15, 7, 13}}}) {

    Index_type num_moments = versions[0];
    Index_type num_directions = versions[1];
    Index_type num_groups = versions[2];
    Index_type num_zones = versions[3];

    // allocate data
    // phi is initialized to all zeros, the others are randomized
    std::vector<double> ell_data(num_moments * num_directions);
    std::vector<double> psi_data(num_directions * num_groups * num_zones);
    std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);

    std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_real_distribution<double> rand_gen(0.0, 1.0);

    // randomize data
    for (size_t i = 0; i < ell_data.size(); ++i) {
      ell_data[i] = rand_gen(gen);
    }
    for (size_t i = 0; i < psi_data.size(); ++i) {
      psi_data[i] = rand_gen(gen);
    }

    // create views on data
    typename POL::ELL_VIEW ell(
        &ell_data[0],
        make_permuted_layout({{num_moments, num_directions}},
                             RAJA::as_array<typename POL::ELL_PERM>::get()));
    typename POL::PSI_VIEW psi(
        &psi_data[0],
        make_permuted_layout({{num_directions, num_groups, num_zones}},
                             RAJA::as_array<typename POL::PSI_PERM>::get()));
    typename POL::PHI_VIEW phi(
        &phi_data[0],
        make_permuted_layout({{num_moments, num_groups, num_zones}},
                             RAJA::as_array<typename POL::PHI_PERM>::get()));

    // get execution policy
    using EXEC = typename POL::EXEC;

    // do calculation using RAJA
    forallN<EXEC, IMoment, IDirection, IGroup, IZone>(
        RangeSegment(0, num_moments),
        RangeSegment(0, num_directions),
        RangeSegment(0, num_groups),
        RangeSegment(0, num_zones),
        [=](IMoment m, IDirection d, IGroup g, IZone z) {
          phi(m, g, z) += ell(m, d) * psi(d, g, z);
        });

    // CHECK ANSWER against the hand-written sequential kernel
    for (IZone z(0); z < num_zones; ++z) {
      for (IGroup g(0); g < num_groups; ++g) {
        for (IMoment m(0); m < num_moments; ++m) {
          double total = 0.0;
          for (IDirection d(0); d < num_directions; ++d) {
            total += ell(m, d) * psi(d, g, z);
          }
          ASSERT_FLOAT_EQ(total, phi(m, g, z));
        }
      }
    }
  }
}

REGISTER_TYPED_TEST_CASE_P(LTimesTest, LTimesNestedTest);

INSTANTIATE_TYPED_TEST_CASE_P(LTimes, LTimesTest, LTimesPolicies);
