
/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::cuda
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_cuda_HPP
#define RAJA_pattern_teams_cuda_HPP

#include "RAJA/pattern/teams/teams_core.hpp"


namespace RAJA
{

template <bool async, int num_threads = 0>
struct cuda_launch_t {
};

template <typename BODY>
__global__ void launch_global_fcn(LaunchContext ctx, BODY body)
{
  // printf("Entering global function\n");
  body(ctx);
  // printf("Leaving global function\n");
}


template <bool async>
struct LaunchExecute<RAJA::cuda_launch_t<async, 0>> {
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
    //printf("thread block dim %d %d %d \n", threads.x, threads.y, threads.z);
    //printf("grid block dim %d %d %d \n", blocks.x, blocks.y, blocks.z);
    launch_global_fcn<<<blocks, threads>>>(ctx, body);

    if (!async) {
      cudaDeviceSynchronize();
    }
  }
};


template <typename BODY, int num_threads>
__launch_bounds__(num_threads, 1) __global__
    void launch_global_fcn_fixed(LaunchContext ctx, BODY body)
{
  // printf("Entering global function\n");
  body(ctx);
  // printf("Leaving global function\n");
}


template <bool async, int nthreads>
struct LaunchExecute<RAJA::cuda_launch_t<async, nthreads>> {
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
      cudaDeviceSynchronize();
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_thread_x_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_thread_y_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = threadIdx.y; i < len; i += blockDim.y) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_thread_z_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = threadIdx.z; i < len; i += blockDim.z) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_x_loop, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();

    for (int i = blockIdx.x; i < len; i += gridDim.x) {
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_x_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();
    {
      const int i = blockIdx.x;
      body(*(segment.begin() + i));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_y_direct, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    int len = segment.end() - segment.begin();
    {
      const int i = blockIdx.y;
      body(*(segment.begin() + i));
    }
  }
};

// collapsed cuda policies

template <typename SEGMENT>
struct LoopExecute<cuda_block_xyz_direct<2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    int len1 = segment1.end() - segment1.begin();
    int len0 = segment0.end() - segment0.begin();
    {
      const int i = blockIdx.x;
      const int j = blockIdx.y;
      body(*(segment0.begin() + i), *(segment1.begin() + j));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_thread_xyz_direct<2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    int len1 = segment1.end() - segment1.begin();
    int len0 = segment0.end() - segment0.begin();
    {
      const int i = threadIdx.x;
      const int j = threadIdx.y;
      body(*(segment0.begin() + i), *(segment1.begin() + j));
    }
  }
};

template <typename SEGMENT>
struct LoopExecute<cuda_block_xyz_direct<3>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    int len2 = segment2.end() - segment2.begin();
    int len1 = segment1.end() - segment1.begin();
    int len0 = segment0.end() - segment0.begin();
    {
      const int i = blockIdx.x;
      const int j = blockIdx.y;
      const int k = blockIdx.z;
      body(*(segment0.begin() + i),
           *(segment1.begin() + j),
           *(segment2.begin() + k));
    }
  }
};


}  // namespace RAJA
#endif
