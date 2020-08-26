/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::hip
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_hip_HPP
#define RAJA_pattern_teams_hip_HPP

#include "RAJA/pattern/teams/teams_core.hpp"
#include "RAJA/policy/hip/policy.hpp"

namespace RAJA
{

namespace expt
{

template <bool async, int num_threads = 0>
struct hip_launch_t {
};

template <typename BODY>
__global__ void launch_global_fcn(LaunchContext ctx, BODY body)
{
  body(ctx);
}


template <bool async>
struct LaunchExecute<RAJA::expt::hip_launch_t<async, 0>> {
  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    dim3 blocks;
    dim3 threads;

    blocks.x = ctx.teams.value[0];
    blocks.y = ctx.teams.value[1];
    blocks.z = ctx.teams.value[2];

    threads.x = ctx.threads.value[0];
    threads.y = ctx.threads.value[1];
    threads.z = ctx.threads.value[2];
    launch_global_fcn<<<blocks, threads>>>(ctx, body);

    if (!async) {
      hipDeviceSynchronize();
    }
  }
};


template <typename BODY, int num_threads>
__launch_bounds__(num_threads, 1) __global__
    void launch_global_fcn_fixed(LaunchContext ctx, BODY body)
{
  body(ctx);
}


template <bool async, int nthreads>
struct LaunchExecute<RAJA::expt::hip_launch_t<async, nthreads>> {
  template <typename BODY>
  static void exec(LaunchContext const &ctx, BODY const &body)
  {
    dim3 blocks;
    dim3 threads;

    blocks.x = ctx.teams.value[0];
    blocks.y = ctx.teams.value[1];
    blocks.z = ctx.teams.value[2];

    threads.x = ctx.threads.value[0];
    threads.y = ctx.threads.value[1];
    threads.z = ctx.threads.value[2];
    launch_global_fcn_fixed<nthreads><<<blocks, threads>>>(ctx, body);

    if (!async) {
      hipDeviceSynchronize();
    }
  }
};

/*
  HIP thread loops with block strides
*/

template <typename SEGMENT>
struct LoopExecute<hip_thread_x_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = threadIdx.x; tx < len; tx += blockDim.x) {
      body(*(segment.begin() + tx));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_thread_y_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int ty = threadIdx.y; ty < len; ty += blockDim.y) {
      body(*(segment.begin() + ty));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_thread_z_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tz = threadIdx.z; tz < len; tz += blockDim.z) {
      body(*(segment.begin() + tz));
    }
  }
};

/*
  HIP thread direct mappings
*/

template <typename SEGMENT>
struct LoopExecute<hip_thread_x_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = threadIdx.x;
      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_thread_y_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int ty = threadIdx.y;
      if (ty < len) body(*(segment.begin() + ty));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_thread_z_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tz = threadIdx.z;
      if (tz < len) body(*(segment.begin() + tz));
    }
  }
};

/*
  HIP block loops with grid strides
*/
template <typename SEGMENT>
struct LoopExecute<hip_block_x_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = blockIdx.x; bx < len; bx += gridDim.x) {
      body(*(segment.begin() + bx));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_block_y_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int by = blockIdx.y; by < len; by += gridDim.y) {
      body(*(segment.begin() + by));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_block_z_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bz = blockIdx.z; bz < len; bz += gridDim.z) {
      body(*(segment.begin() + bz));
    }
  }
};

/*
  HIP block direct mappings
*/

template <typename SEGMENT>
struct LoopExecute<hip_block_x_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = blockIdx.x;
      if (bx < len) body(*(segment.begin() + bx));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_block_y_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int by = blockIdx.y;
      if (by < len) body(*(segment.begin() + by));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_block_z_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bz = blockIdx.z;
      if (bz < len) body(*(segment.begin() + bz));
    }
  }
};


// perfectly nested hip direct policies
struct hip_block_xy_nested_direct;
struct hip_block_xyz_nested_direct;

struct hip_thread_xy_nested_direct;
struct hip_thread_xyz_nested_direct;


template <typename SEGMENT>
struct LoopExecute<hip_block_xy_nested_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = blockIdx.x;
      const int ty = blockIdx.y;
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_thread_xy_nested_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = threadIdx.x;
      const int ty = threadIdx.y;
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};


template <typename SEGMENT>
struct LoopExecute<hip_block_xyz_nested_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = blockIdx.x;
      const int ty = blockIdx.y;
      const int tz = blockIdx.z;
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment2.begin() + tz));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_thread_xyz_nested_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = threadIdx.x;
      const int ty = threadIdx.y;
      const int tz = threadIdx.z;
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment2.begin() + tz));
    }
  }
};

// perfectly nested hip loop policies
struct hip_block_xy_nested_loop;
struct hip_block_xyz_nested_loop;

struct hip_thread_xy_nested_loop;
struct hip_thread_xyz_nested_loop;

template <typename SEGMENT>
struct LoopExecute<hip_block_xy_nested_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      for (int by = blockIdx.y; by < len1; by += gridDim.y) {
        for (int bx = blockIdx.x; bx < len0; bx += gridDim.x) {
          body(*(segment0.begin() + bx), *(segment1.begin() + by));
        }
      }
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<hip_thread_xy_nested_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      for (int ty = threadIdx.y; ty < len1; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < len0; tx += blockDim.x) {
          body(*(segment0.begin() + tx), *(segment1.begin() + ty));
        }
      }
    }
  }
};


template <typename SEGMENT>
struct LoopExecute<hip_block_xyz_nested_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int bz = blockIdx.z; bz < len2; bz += gridDim.z) {
      for (int by = blockIdx.y; by < len1; by += gridDim.y) {
        for (int bx = blockIdx.x; bx < len0; bx += gridDim.x) {
          body(*(segment0.begin() + bx),
               *(segment1.begin() + by),
               *(segment2.begin() + bz));
        }
      }
    }
  }
};


template <typename SEGMENT>
struct LoopExecute<hip_thread_xyz_nested_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int bz = threadIdx.z; bz < len2; bz += blockDim.z) {
      for (int by = threadIdx.y; by < len1; by += blockDim.y) {
        for (int bx = threadIdx.x; bx < len0; bx += blockDim.x) {
          body(*(segment0.begin() + bx),
               *(segment1.begin() + by),
               *(segment2.begin() + bz));
        }
      }
    }
  }
};

}  // namespace expt

}  // namespace RAJA
#endif
