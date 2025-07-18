/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::launch::openmp
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_launch_openmp_HPP
#define RAJA_pattern_launch_openmp_HPP

#include "RAJA/pattern/launch/launch_core.hpp"
#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

template<>
struct LaunchExecute<RAJA::omp_launch_t>
{

  template<typename BODY, typename ReduceParams>
  static concepts::enable_if_t<
      resources::EventProxy<resources::Resource>,
      RAJA::expt::type_traits::is_ForallParamPack<ReduceParams>,
      RAJA::expt::type_traits::is_ForallParamPack_empty<ReduceParams>>
  exec(RAJA::resources::Resource res,
       LaunchParams const& params,
       BODY const& body,
       ReduceParams& RAJA_UNUSED_ARG(launch_reducers))
  {
    RAJA::region<RAJA::omp_parallel_region>([&]() {
      LaunchContext ctx;

      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      ctx.shared_mem_ptr = (char*)malloc(params.shared_mem_size);

      loop_body.get_priv()(ctx);

      free(ctx.shared_mem_ptr);
      ctx.shared_mem_ptr = nullptr;
    });

    return resources::EventProxy<resources::Resource>(res);
  }

  template<typename ReduceParams, typename BODY>
  static concepts::enable_if_t<
      resources::EventProxy<resources::Resource>,
      RAJA::expt::type_traits::is_ForallParamPack<ReduceParams>,
      concepts::negate<
          RAJA::expt::type_traits::is_ForallParamPack_empty<ReduceParams>>>
  exec(RAJA::resources::Resource res,
       LaunchParams const& launch_params,
       BODY const& body,
       ReduceParams& f_params)
  {
    using EXEC_POL = RAJA::omp_launch_t;
    EXEC_POL pol {};

    expt::ParamMultiplexer::parampack_init(pol, f_params);

    // reducer object must be named f_params as expected by macro below
    RAJA_OMP_DECLARE_REDUCTION_COMBINE;

#pragma omp parallel reduction(combine : f_params)
    {

      LaunchContext ctx;

      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

      ctx.shared_mem_ptr = (char*)malloc(launch_params.shared_mem_size);

      expt::invoke_body(f_params, loop_body.get_priv(), ctx);

      free(ctx.shared_mem_ptr);
      ctx.shared_mem_ptr = nullptr;
    }

    expt::ParamMultiplexer::parampack_resolve(pol, f_params);

    return resources::EventProxy<resources::Resource>(res);
  }
};

template<typename SEGMENT>
struct LoopExecute<omp_parallel_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();
    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);
#pragma omp for
      for (int i = 0; i < len; i++)
      {

        loop_body.get_priv()(*(segment.begin() + i));
      }
    });
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {

          loop_body.get_priv()(*(segment0.begin() + i),
                               *(segment1.begin() + j));
        }
      }
    });
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for
      for (int k = 0; k < len2; k++)
      {
        for (int j = 0; j < len1; j++)
        {
          for (int i = 0; i < len0; i++)
          {
            loop_body.get_priv()(*(segment0.begin() + i),
                                 *(segment1.begin() + j),
                                 *(segment2.begin() + k));
          }
        }
      }
    });
  }
};

template<typename SEGMENT>
struct LoopExecute<omp_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();
#pragma omp for
    for (int i = 0; i < len; i++)
    {

      body(*(segment.begin() + i));
    }
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp for
    for (int j = 0; j < len1; j++)
    {
      for (int i = 0; i < len0; i++)
      {

        body(*(segment0.begin() + i), *(segment1.begin() + j));
      }
    }
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp for
    for (int k = 0; k < len2; k++)
    {
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {
          body(*(segment0.begin() + i), *(segment1.begin() + j),
               *(segment2.begin() + k));
        }
      }
    }
  }
};

//
// Return local index
//
template<typename SEGMENT>
struct LoopICountExecute<omp_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();

#pragma omp for
    for (int i = 0; i < len; i++)
    {
      body(*(segment.begin() + i), i);
    }
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp for
    for (int j = 0; j < len1; j++)
    {
      for (int i = 0; i < len0; i++)
      {

        body(*(segment0.begin() + i), *(segment1.begin() + j), i, j);
      }
    }
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

#pragma omp for
    for (int k = 0; k < len2; k++)
    {
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {
          body(*(segment0.begin() + i), *(segment1.begin() + j),
               *(segment2.begin() + k), i, j, k);
        }
      }
    }
  }
};

// policy for perfectly nested loops
struct omp_parallel_nested_for_exec;

template<typename SEGMENT>
struct LoopExecute<omp_parallel_nested_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for RAJA_COLLAPSE(2)
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {

          loop_body.get_priv()(*(segment0.begin() + i),
                               *(segment1.begin() + j));
        }
      }
    });
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for RAJA_COLLAPSE(3)
      for (int k = 0; k < len2; k++)
      {
        for (int j = 0; j < len1; j++)
        {
          for (int i = 0; i < len0; i++)
          {
            loop_body.get_priv()(*(segment0.begin() + i),
                                 *(segment1.begin() + j),
                                 *(segment2.begin() + k));
          }
        }
      }
    });
  }
};

// Return local index
template<typename SEGMENT>
struct LoopICountExecute<omp_parallel_nested_for_exec, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {

    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for RAJA_COLLAPSE(2)
      for (int j = 0; j < len1; j++)
      {
        for (int i = 0; i < len0; i++)
        {

          loop_body.get_priv()(*(segment0.begin() + i), *(segment1.begin() + j),
                               i, j);
        }
      }
    });
  }

  template<typename BODY>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {

    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for RAJA_COLLAPSE(3)
      for (int k = 0; k < len2; k++)
      {
        for (int j = 0; j < len1; j++)
        {
          for (int i = 0; i < len0; i++)
          {
            loop_body.get_priv()(*(segment0.begin() + i),
                                 *(segment1.begin() + j),
                                 *(segment2.begin() + k), i, j, k);
          }
        }
      }
    });
  }
};

template<typename SEGMENT>
struct TileExecute<omp_parallel_for_exec, SEGMENT>
{

  template<typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp for
      for (int i = 0; i < len; i += tile_size)
      {
        loop_body.get_priv()(segment.slice(i, tile_size));
      }
    });
  }
};

template<typename SEGMENT>
struct TileTCountExecute<omp_parallel_for_exec, SEGMENT>
{

  template<typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    const int len      = segment.end() - segment.begin();
    const int numTiles = (len - 1) / tile_size + 1;

    RAJA::region<RAJA::omp_parallel_region>([&]() {
      using RAJA::internal::thread_privatize;
      auto loop_body = thread_privatize(body);

#pragma omp parallel for
      for (int i = 0; i < numTiles; i++)
      {
        const int i_tile_size = i * tile_size;
        loop_body.get_priv()(segment.slice(i_tile_size, tile_size), i);
      }
    });
  }
};

template<typename SEGMENT>
struct TileExecute<omp_for_exec, SEGMENT>
{

  template<typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    int len = segment.end() - segment.begin();
#pragma omp for
    for (int i = 0; i < len; i += tile_size)
    {
      body(segment.slice(i, tile_size));
    }
  }
};

template<typename SEGMENT>
struct TileTCountExecute<omp_for_exec, SEGMENT>
{

  template<typename BODY, typename TILE_T>
  static RAJA_INLINE RAJA_HOST_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {

    const int len      = segment.end() - segment.begin();
    const int numTiles = (len - 1) / tile_size + 1;

#pragma omp for
    for (int i = 0; i < numTiles; i++)
    {
      const int i_tile_size = i * tile_size;
      body(segment.slice(i_tile_size, tile_size), i);
    }
  }
};

}  // namespace RAJA
#endif
