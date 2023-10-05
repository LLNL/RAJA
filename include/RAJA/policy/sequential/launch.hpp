/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::seq
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_launch_sequential_HPP
#define RAJA_pattern_launch_sequential_HPP

#include "RAJA/pattern/launch/launch_core.hpp"
#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/pattern/params/forall.hpp" 

namespace RAJA
{

template <>
struct LaunchExecute<RAJA::null_launch_t> {
  template <typename BODY>
  static void exec(LaunchContext const& RAJA_UNUSED_ARG(ctx),
                   BODY const& RAJA_UNUSED_ARG(body))
  {
    RAJA_ABORT_OR_THROW("NULL Launch");
  }
};


template <>
struct LaunchExecute<RAJA::seq_launch_t> {

  template <typename BODY>
  static resources::EventProxy<resources::Resource>
  exec(RAJA::resources::Resource res, LaunchParams const &params, const char *RAJA_UNUSED_ARG(kernel_name), BODY const &body)
  {

    LaunchContext ctx;

    char *kernel_local_mem = new char[params.shared_mem_size];
    ctx.shared_mem_ptr = kernel_local_mem;

    body(ctx);

    delete[] kernel_local_mem;
    ctx.shared_mem_ptr = nullptr;

    return resources::EventProxy<resources::Resource>(res);
  }

  template<typename ForallParam, typename BODY>
  /*  Q - do need to handle the case of when it is empty
  concepts::enable_if_t<void, 
                        expt::type_traits::is_ForallParamPack<ForallParam>,
                         concepts::negate<expt::type_traits::is_ForallParamPack_empty<ForallParam>>>
  */  
  static void 
  exec(LaunchParams const &params, ForallParam f_params, BODY const &body)
  {

    std::cout<<"using new reducers sequential policy "<<std::endl;
    expt::ParamMultiplexer::init<seq_exec>(f_params);

    LaunchContext ctx;
    expt::invoke_body(f_params, body, ctx);

    expt::ParamMultiplexer::resolve<seq_exec>(f_params);
  }

};


template <typename SEGMENT>
struct LoopExecute<seq_exec, SEGMENT> {

  RAJA_SUPPRESS_HD_WARN
  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    for (int i = 0; i < len; i++) {

      body(*(segment.begin() + i));
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    for (int i = 0; i < len; i++) {
      body(*(segment.begin() + i));
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    // block stride loop
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int j = 0; j < len1; j++) {
      for (int i = 0; i < len0; i++) {

        body(*(segment0.begin() + i), *(segment1.begin() + j));
      }
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    // block stride loop
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int k = 0; k < len2; k++) {
      for (int j = 0; j < len1; j++) {
        for (int i = 0; i < len0; i++) {
          body(*(segment0.begin() + i),
               *(segment1.begin() + j),
               *(segment2.begin() + k));
        }
      }
    }
  }

};


template <typename SEGMENT>
struct LoopICountExecute<seq_exec, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();
    for (int i = 0; i < len; i++) {
      body(*(segment.begin() + i), i);
    }
  }

    template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {

    // block stride loop
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int j = 0; j < len1; j++) {
      for (int i = 0; i < len0; i++) {

        body(*(segment0.begin() + i), *(segment1.begin() + j), i, j);
      }
    }
  }

  template <typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {

    // block stride loop
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int k = 0; k < len2; k++) {
      for (int j = 0; j < len1; j++) {
        for (int i = 0; i < len0; i++) {
          body(*(segment0.begin() + i),
               *(segment1.begin() + j),
               *(segment2.begin() + k), i, j, k);
        }
      }
    }
  }

};

//Tile Execute + variants

template <typename SEGMENT>
struct TileExecute<seq_exec, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_HOST_DEVICE RAJA_INLINE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = 0; tx < len; tx += tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }

};

template <typename SEGMENT>
struct TileTCountExecute<seq_exec, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_HOST_DEVICE RAJA_INLINE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = 0, bx=0; tx < len; tx += tile_size, bx++)
    {
      body(segment.slice(tx, tile_size), bx);
    }
  }

};

}  // namespace RAJA
#endif
