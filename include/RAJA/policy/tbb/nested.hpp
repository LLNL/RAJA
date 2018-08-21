/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forall:nested
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_policy_tbb_nested_HPP
#define RAJA_policy_tbb_nested_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

#include <cassert>
#include <climits>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/pattern/nested.hpp"

#include "RAJA/policy/tbb/policy.hpp"

#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{
namespace nested
{

  struct tbb_collapse_exec {
  };

  template <typename... FOR>
  using TbbParallelCollapse = Collapse<tbb_collapse_exec, FOR...>;


  /////////
  //Collapsing two loops
  /////////

  // TODO, check that FT... are TBB policies
  template <typename FT0, typename FT1>
  struct Executor<Collapse<tbb_collapse_exec, FT0, FT1> > {

    static_assert(std::is_base_of<internal::ForBase, FT0>::value,
                  "Only For-based policies should get here");
    static_assert(std::is_base_of<internal::ForBase, FT1>::value,
                  "Only For-based policies should get here");
    template <typename WrappedBody>
    void operator()(Collapse<tbb_collapse_exec, FT0, FT1> const &, WrappedBody const &wrap)
    {

      auto b0 = std::begin(camp::get<FT0::index_val>(wrap.data.segment_tuple));
      auto b1 = std::begin(camp::get<FT1::index_val>(wrap.data.segment_tuple));
      auto e0 = std::end(camp::get<FT0::index_val>(wrap.data.segment_tuple));
      auto e1 = std::end(camp::get<FT1::index_val>(wrap.data.segment_tuple));

      auto l0 = std::distance(b0,e0);
      auto l1 = std::distance(b1,e1);

      {
        auto privatizer = RAJA::nested::thread_privatize(wrap);
        auto &private_wrap = privatizer.get_priv();

        using brange = tbb::blocked_range2d<decltype(l0), decltype(l1)>;

        tbb::parallel_for( brange(b0, e0, b1, e1),
          [=](const brange &r) {
            for (auto i0 : r.rows()) {
              for (auto i1 : r.cols()) {
                private_wrap.data.template assign_index<FT0::index_val>(b0[i0]);
                private_wrap.data.template assign_index<FT1::index_val>(b1[i1]);
                private_wrap();
              }
            }
          });
      }
    }
  };



  template <typename FT0, typename FT1, typename FT2>
  struct Executor<Collapse<tbb_collapse_exec, FT0, FT1, FT2 > > {

    static_assert(std::is_base_of<internal::ForBase, FT0>::value,
                  "Only For-based policies should get here");
    static_assert(std::is_base_of<internal::ForBase, FT1>::value,
                  "Only For-based policies should get here");
    static_assert(std::is_base_of<internal::ForBase, FT2>::value,
                  "Only For-based policies should get here");

    template <typename WrappedBody>
    void operator()(Collapse<tbb_collapse_exec, FT0, FT1, FT2> const &, WrappedBody const &wrap)
    {

      auto b0 = std::begin(camp::get<FT0::index_val>(wrap.data.segment_tuple));
      auto b1 = std::begin(camp::get<FT1::index_val>(wrap.data.segment_tuple));
      auto b2 = std::begin(camp::get<FT2::index_val>(wrap.data.segment_tuple));

      auto e0 = std::end(camp::get<FT0::index_val>(wrap.data.segment_tuple));
      auto e1 = std::end(camp::get<FT1::index_val>(wrap.data.segment_tuple));
      auto e2 = std::end(camp::get<FT2::index_val>(wrap.data.segment_tuple));

      auto l0 = std::distance(b0,e0);
      auto l1 = std::distance(b1,e1);
      auto l2 = std::distance(b2,e2);

      {
        auto privatizer = RAJA::nested::thread_privatize(wrap);
        auto &private_wrap = privatizer.get_priv();

        using brange = tbb::blocked_range3d<decltype(l0), decltype(l1), decltype(l2)>;

        tbb::parallel_for( brange(b0, e0, b1, e1, b2, e2),
          [=](const brange &r) {
            for (auto i0 : r.pages()) {
              for (auto i1 : r.rows()) {
                for (auto i2 : r.cols()) {
                  private_wrap.data.template assign_index<FT0::index_val>(b0[i0]);
                  private_wrap.data.template assign_index<FT1::index_val>(b1[i1]);
                  private_wrap.data.template assign_index<FT2::index_val>(b2[i2]);
                  private_wrap();
                }
              }
            }
          });
      }
    }
  };

}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_TBB guard

#endif  // closing endif for header file include guard
