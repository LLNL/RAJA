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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_hip_HPP
#define RAJA_pattern_teams_hip_HPP

#include "RAJA/pattern/teams/teams_core.hpp"
#include "RAJA/pattern/detail/privatizer.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#include "RAJA/util/resource.hpp"

namespace RAJA
{

namespace expt
{

template <typename BODY>
__global__ void launch_global_fcn(LaunchContext ctx, BODY body_in)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body = privatizer.get_priv();
  body(ctx);
}

template <bool async>
struct LaunchExecute<RAJA::expt::hip_launch_t<async, 0>> {

  template <typename BODY_IN>
  static void exec(LaunchContext const &ctx, BODY_IN &&body_in)
  {
    using BODY = camp::decay<BODY_IN>;

    auto func = launch_global_fcn<BODY>;

    resources::Hip hip_res = resources::Hip::get_default();

    //
    // Compute the number of blocks and threads
    //

    hip_dim_t gridSize{ static_cast<hip_dim_member_t>(ctx.teams.value[0]),
                        static_cast<hip_dim_member_t>(ctx.teams.value[1]),
                        static_cast<hip_dim_member_t>(ctx.teams.value[2]) };

    hip_dim_t blockSize{ static_cast<hip_dim_member_t>(ctx.threads.value[0]),
                         static_cast<hip_dim_member_t>(ctx.threads.value[1]),
                         static_cast<hip_dim_member_t>(ctx.threads.value[2]) };

    // Only launch kernel if we have something to iterate over
    constexpr hip_dim_member_t zero = 0;
    if ( gridSize.x  > zero && gridSize.y  > zero && gridSize.z  > zero &&
         blockSize.x > zero && blockSize.y > zero && blockSize.z > zero ) {

      RAJA_FT_BEGIN;

      //
      // Setup shared memory buffers
      //
      size_t shmem = 0;

      {
        //
        // Privatize the loop_body, using make_launch_body to setup reductions
        //
        BODY body = RAJA::hip::make_launch_body(
            gridSize, blockSize, shmem, hip_res, std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void *args[] = {(void*)&ctx, (void*)&body};
        RAJA::hip::launch((const void*)func, gridSize, blockSize, args, shmem, hip_res, async, ctx.kernel_name);
      }

      RAJA_FT_END;
    }

  }

  template <typename BODY_IN>
  static resources::EventProxy<resources::Resource>
  exec(RAJA::resources::Resource res, LaunchContext const &ctx, BODY_IN &&body_in)
  {
    using BODY = camp::decay<BODY_IN>;

    auto func = launch_global_fcn<BODY>;

    resources::Hip hip_res = res.get<RAJA::resources::Hip>();

    //
    // Compute the number of blocks and threads
    //

    hip_dim_t gridSize{ static_cast<hip_dim_member_t>(ctx.teams.value[0]),
                        static_cast<hip_dim_member_t>(ctx.teams.value[1]),
                        static_cast<hip_dim_member_t>(ctx.teams.value[2]) };

    hip_dim_t blockSize{ static_cast<hip_dim_member_t>(ctx.threads.value[0]),
                         static_cast<hip_dim_member_t>(ctx.threads.value[1]),
                         static_cast<hip_dim_member_t>(ctx.threads.value[2]) };

    // Only launch kernel if we have something to iterate over
    constexpr hip_dim_member_t zero = 0;
    if ( gridSize.x  > zero && gridSize.y  > zero && gridSize.z  > zero &&
         blockSize.x > zero && blockSize.y > zero && blockSize.z > zero ) {

      RAJA_FT_BEGIN;

      //
      // Setup shared memory buffers
      //
      size_t shmem = 0;

      {
        //
        // Privatize the loop_body, using make_launch_body to setup reductions
        //
        BODY body = RAJA::hip::make_launch_body(
            gridSize, blockSize, shmem, hip_res, std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void *args[] = {(void*)&ctx, (void*)&body};
        RAJA::hip::launch((const void*)func, gridSize, blockSize, args, shmem, hip_res, async, ctx.kernel_name);
      }

      RAJA_FT_END;
    }

    return resources::EventProxy<resources::Resource>(res);
  }

};

template <typename BODY, int num_threads>
__launch_bounds__(num_threads, 1) __global__
static void launch_global_fcn_fixed(LaunchContext ctx, BODY body_in)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body = privatizer.get_priv();
  body(ctx);
}

template <bool async, int nthreads>
struct LaunchExecute<RAJA::expt::hip_launch_t<async, nthreads>> {

  template <typename BODY_IN>
  static void exec(LaunchContext const &ctx, BODY_IN &&body_in)
  {
    using BODY = camp::decay<BODY_IN>;

    auto func = launch_global_fcn_fixed<BODY, nthreads>;

    resources::Hip hip_res = resources::Hip::get_default();

    //
    // Compute the number of blocks and threads
    //

    hip_dim_t gridSize{ static_cast<hip_dim_member_t>(ctx.teams.value[0]),
                        static_cast<hip_dim_member_t>(ctx.teams.value[1]),
                        static_cast<hip_dim_member_t>(ctx.teams.value[2]) };

    hip_dim_t blockSize{ static_cast<hip_dim_member_t>(ctx.threads.value[0]),
                         static_cast<hip_dim_member_t>(ctx.threads.value[1]),
                         static_cast<hip_dim_member_t>(ctx.threads.value[2]) };

    // Only launch kernel if we have something to iterate over
    constexpr hip_dim_member_t zero = 0;
    if ( gridSize.x  > zero && gridSize.y  > zero && gridSize.z  > zero &&
         blockSize.x > zero && blockSize.y > zero && blockSize.z > zero ) {

      RAJA_FT_BEGIN;

      //
      // Setup shared memory buffers
      //
      size_t shmem = 0;

      {
        //
        // Privatize the loop_body, using make_launch_body to setup reductions
        //
        BODY body = RAJA::hip::make_launch_body(
            gridSize, blockSize, shmem, hip_res, std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void *args[] = {(void*)&ctx, (void*)&body};
        RAJA::hip::launch((const void*)func, gridSize, blockSize, args, shmem, hip_res, async, ctx.kernel_name);
      }

      RAJA_FT_END;
    }

  }

  template <typename BODY_IN>
  static resources::EventProxy<resources::Resource>
  exec(RAJA::resources::Resource res, LaunchContext const &ctx, BODY_IN &&body_in)
  {
    using BODY = camp::decay<BODY_IN>;

    auto func = launch_global_fcn<BODY>;

    resources::Hip hip_res = res.get<RAJA::resources::Hip>();

    //
    // Compute the number of blocks and threads
    //

    hip_dim_t gridSize{ static_cast<hip_dim_member_t>(ctx.teams.value[0]),
                        static_cast<hip_dim_member_t>(ctx.teams.value[1]),
                        static_cast<hip_dim_member_t>(ctx.teams.value[2]) };

    hip_dim_t blockSize{ static_cast<hip_dim_member_t>(ctx.threads.value[0]),
                         static_cast<hip_dim_member_t>(ctx.threads.value[1]),
                         static_cast<hip_dim_member_t>(ctx.threads.value[2]) };

    // Only launch kernel if we have something to iterate over
    constexpr hip_dim_member_t zero = 0;
    if ( gridSize.x  > zero && gridSize.y  > zero && gridSize.z  > zero &&
         blockSize.x > zero && blockSize.y > zero && blockSize.z > zero ) {

      RAJA_FT_BEGIN;

      //
      // Setup shared memory buffers
      //
      size_t shmem = 0;

      {
        //
        // Privatize the loop_body, using make_launch_body to setup reductions
        //
        BODY body = RAJA::hip::make_launch_body(
            gridSize, blockSize, shmem, hip_res, std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void *args[] = {(void*)&ctx, (void*)&body};
        RAJA::hip::launch((const void*)func, gridSize, blockSize, args, shmem, hip_res, async, ctx.kernel_name);
      }

      RAJA_FT_END;
    }

    return resources::EventProxy<resources::Resource>(res);
  }

};

/*
   HIP global thread mapping
*/
template<int ... DIM>
struct hip_global_thread;

using hip_global_thread_x = hip_global_thread<0>;
using hip_global_thread_y = hip_global_thread<1>;
using hip_global_thread_z = hip_global_thread<2>;

template <typename SEGMENT, int DIM>
struct LoopExecute<hip_global_thread<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_hip_dim<DIM>(threadIdx) +
        internal::get_hip_dim<DIM>(blockDim)*internal::get_hip_dim<DIM>(blockIdx);

      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};

using hip_global_thread_xy = hip_global_thread<0,1>;
using hip_global_thread_xz = hip_global_thread<0,2>;
using hip_global_thread_yx = hip_global_thread<1,0>;
using hip_global_thread_yz = hip_global_thread<1,2>;
using hip_global_thread_zx = hip_global_thread<2,0>;
using hip_global_thread_zy = hip_global_thread<2,1>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<hip_global_thread<DIM0, DIM1>, SEGMENT> {

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
      const int tx = internal::get_hip_dim<DIM0>(threadIdx) +
        internal::get_hip_dim<DIM0>(blockDim)*internal::get_hip_dim<DIM0>(blockIdx);

      const int ty = internal::get_hip_dim<DIM1>(threadIdx) +
        internal::get_hip_dim<DIM1>(blockDim)*internal::get_hip_dim<DIM1>(blockIdx);

      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

using hip_global_thread_xyz = hip_global_thread<0,1,2>;
using hip_global_thread_xzy = hip_global_thread<0,2,1>;
using hip_global_thread_yxz = hip_global_thread<1,0,2>;
using hip_global_thread_yzx = hip_global_thread<1,2,0>;
using hip_global_thread_zxy = hip_global_thread<2,0,1>;
using hip_global_thread_zyx = hip_global_thread<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<hip_global_thread<DIM0, DIM1, DIM2>, SEGMENT> {

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
      const int tx = internal::get_hip_dim<DIM0>(threadIdx) +
        internal::get_hip_dim<DIM0>(blockDim)*internal::get_hip_dim<DIM0>(blockIdx);

      const int ty = internal::get_hip_dim<DIM1>(threadIdx) +
        internal::get_hip_dim<DIM1>(blockDim)*internal::get_hip_dim<DIM1>(blockIdx);

      const int tz = internal::get_hip_dim<DIM2>(threadIdx) +
        internal::get_hip_dim<DIM2>(blockDim)*internal::get_hip_dim<DIM2>(blockIdx);

      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment1.begin() + ty));
    }
  }
};

/*
Reshape threads in a block into a 1D iteration space
*/
template<int ... dim>
struct hip_flatten_block_threads_direct{};

using hip_flatten_block_threads_xy_direct = hip_flatten_block_threads_direct<0,1>;
using hip_flatten_block_threads_xz_direct = hip_flatten_block_threads_direct<0,2>;
using hip_flatten_block_threads_yx_direct = hip_flatten_block_threads_direct<1,0>;
using hip_flatten_block_threads_yz_direct = hip_flatten_block_threads_direct<1,2>;
using hip_flatten_block_threads_zx_direct = hip_flatten_block_threads_direct<2,0>;
using hip_flatten_block_threads_zy_direct = hip_flatten_block_threads_direct<2,1>;

using hip_flatten_block_threads_xyz_direct = hip_flatten_block_threads_direct<0,1,2>;
using hip_flatten_block_threads_xzy_direct = hip_flatten_block_threads_direct<0,2,1>;
using hip_flatten_block_threads_yxz_direct = hip_flatten_block_threads_direct<1,0,2>;
using hip_flatten_block_threads_yzx_direct = hip_flatten_block_threads_direct<1,2,0>;
using hip_flatten_block_threads_zxy_direct = hip_flatten_block_threads_direct<2,0,1>;
using hip_flatten_block_threads_zyx_direct = hip_flatten_block_threads_direct<2,1,0>;

template<int ... dim>
struct hip_flatten_block_threads_loop{};

using hip_flatten_block_threads_xy_loop = hip_flatten_block_threads_loop<0,1>;
using hip_flatten_block_threads_xz_loop = hip_flatten_block_threads_loop<0,2>;
using hip_flatten_block_threads_yx_loop = hip_flatten_block_threads_loop<1,0>;
using hip_flatten_block_threads_yz_loop = hip_flatten_block_threads_loop<1,2>;
using hip_flatten_block_threads_zx_loop = hip_flatten_block_threads_loop<2,0>;
using hip_flatten_block_threads_zy_loop = hip_flatten_block_threads_loop<2,1>;

using hip_flatten_block_threads_xyz_loop = hip_flatten_block_threads_loop<0,1,2>;
using hip_flatten_block_threads_xzy_loop = hip_flatten_block_threads_loop<0,2,1>;
using hip_flatten_block_threads_yxz_loop = hip_flatten_block_threads_loop<1,0,2>;
using hip_flatten_block_threads_yzx_loop = hip_flatten_block_threads_loop<1,2,0>;
using hip_flatten_block_threads_zxy_loop = hip_flatten_block_threads_loop<2,0,1>;
using hip_flatten_block_threads_zyx_loop = hip_flatten_block_threads_loop<2,1,0>;

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<hip_flatten_block_threads_direct<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_hip_dim<DIM0>(threadIdx);
      const int ty = internal::get_hip_dim<DIM1>(threadIdx);
      const int bx = internal::get_hip_dim<DIM0>(blockDim);
      const int tid = tx + bx*ty;

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<hip_flatten_block_threads_loop<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = internal::get_hip_dim<DIM0>(threadIdx);
    const int ty = internal::get_hip_dim<DIM1>(threadIdx);

    const int bx = internal::get_hip_dim<DIM0>(blockDim);
    const int by = internal::get_hip_dim<DIM1>(blockDim);

    for(int tid = tx + bx*ty; tid < len; tid += bx*by) {
      body(*(segment.begin() + tid));
    }

  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<hip_flatten_block_threads_direct<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_hip_dim<DIM0>(threadIdx);
      const int ty = internal::get_hip_dim<DIM1>(threadIdx);
      const int tz = internal::get_hip_dim<DIM2>(threadIdx);
      const int bx = internal::get_hip_dim<DIM0>(blockDim);
      const int by = internal::get_hip_dim<DIM1>(blockDim);
      const int tid = tx + bx*(ty + by*tz);

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<hip_flatten_block_threads_loop<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = internal::get_hip_dim<DIM0>(threadIdx);
    const int ty = internal::get_hip_dim<DIM1>(threadIdx);
    const int tz = internal::get_hip_dim<DIM2>(threadIdx);
    const int bx = internal::get_hip_dim<DIM0>(blockDim);
    const int by = internal::get_hip_dim<DIM1>(blockDim);
    const int bz = internal::get_hip_dim<DIM2>(blockDim);

    for(int tid = tx + bx*(ty + by*tz); tid < len; tid += bx*by*bz) {
      body(*(segment.begin() + tid));
    }

  }
};


/*
  HIP thread loops with block strides
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<hip_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_hip_dim<DIM>(threadIdx);
         tx < len;
         tx += internal::get_hip_dim<DIM>(blockDim) )
    {
      body(*(segment.begin() + tx));
    }
  }
};

/*
  HIP thread direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<hip_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_hip_dim<DIM>(threadIdx);
      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};


/*
  HIP block loops with grid strides
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<hip_block_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_hip_dim<DIM>(blockIdx);
         bx < len;
         bx += internal::get_hip_dim<DIM>(gridDim) ) {
      body(*(segment.begin() + bx));
    }
  }
};

/*
  HIP block direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<hip_block_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = internal::get_hip_dim<DIM>(blockIdx);
      if (bx < len) body(*(segment.begin() + bx));
    }
  }
};

/*
  HIP thread loops with block strides + Return Index
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<hip_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_hip_dim<DIM>(threadIdx);
         tx < len;
         tx += internal::get_hip_dim<DIM>(blockDim) )
    {
      body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  HIP thread direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<hip_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_hip_dim<DIM>(threadIdx);
      if (tx < len) body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  HIP block loops with grid strides
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<hip_block_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_hip_dim<DIM>(blockIdx);
         bx < len;
         bx += internal::get_hip_dim<DIM>(gridDim) ) {
      body(*(segment.begin() + bx), bx);
    }
  }
};

/*
  HIP block direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<hip_block_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = internal::get_hip_dim<DIM>(blockIdx);
      if (bx < len) body(*(segment.begin() + bx), bx);
    }
  }
};

// perfectly nested hip direct policies
using hip_block_xy_nested_direct = hip_block_xyz_direct<0,1>;
using hip_block_xz_nested_direct = hip_block_xyz_direct<0,2>;
using hip_block_yx_nested_direct = hip_block_xyz_direct<1,0>;
using hip_block_yz_nested_direct = hip_block_xyz_direct<1,2>;
using hip_block_zx_nested_direct = hip_block_xyz_direct<2,0>;
using hip_block_zy_nested_direct = hip_block_xyz_direct<2,1>;

using hip_block_xyz_nested_direct = hip_block_xyz_direct<0,1,2>;
using hip_block_xzy_nested_direct = hip_block_xyz_direct<0,2,1>;
using hip_block_yxz_nested_direct = hip_block_xyz_direct<1,0,2>;
using hip_block_yzx_nested_direct = hip_block_xyz_direct<1,2,0>;
using hip_block_zxy_nested_direct = hip_block_xyz_direct<2,0,1>;
using hip_block_zyx_nested_direct = hip_block_xyz_direct<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<hip_block_xyz_direct<DIM0, DIM1>, SEGMENT> {

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
      const int tx = internal::get_hip_dim<DIM0>(blockIdx);
      const int ty = internal::get_hip_dim<DIM1>(blockIdx);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<hip_block_xyz_direct<DIM0, DIM1, DIM2>, SEGMENT> {

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
      const int tx = internal::get_hip_dim<DIM0>(blockIdx);
      const int ty = internal::get_hip_dim<DIM1>(blockIdx);
      const int tz = internal::get_hip_dim<DIM2>(blockIdx);
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment2.begin() + tz));
    }
  }
};

/*
  Perfectly nested hip direct policies
  Return local index
*/
template <typename SEGMENT, int DIM0, int DIM1>
struct LoopICountExecute<hip_block_xyz_direct<DIM0, DIM1>, SEGMENT> {

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
      const int tx = internal::get_hip_dim<DIM0>(blockIdx);
      const int ty = internal::get_hip_dim<DIM1>(blockIdx);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty),
             tx, ty);
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<hip_block_xyz_direct<DIM0, DIM1, DIM2>, SEGMENT> {

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
      const int tx = internal::get_hip_dim<DIM0>(blockIdx);
      const int ty = internal::get_hip_dim<DIM1>(blockIdx);
      const int tz = internal::get_hip_dim<DIM2>(blockIdx);
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment2.begin() + tz), tx, ty, tz);
    }
  }
};

// perfectly nested hip loop policies
using hip_block_xy_nested_loop = hip_block_xyz_loop<0,1>;
using hip_block_xz_nested_loop = hip_block_xyz_loop<0,2>;
using hip_block_yx_nested_loop = hip_block_xyz_loop<1,0>;
using hip_block_yz_nested_loop = hip_block_xyz_loop<1,2>;
using hip_block_zx_nested_loop = hip_block_xyz_loop<2,0>;
using hip_block_zy_nested_loop = hip_block_xyz_loop<2,1>;

using hip_block_xyz_nested_loop = hip_block_xyz_loop<0,1,2>;
using hip_block_xzy_nested_loop = hip_block_xyz_loop<0,2,1>;
using hip_block_yxz_nested_loop = hip_block_xyz_loop<1,0,2>;
using hip_block_yzx_nested_loop = hip_block_xyz_loop<1,2,0>;
using hip_block_zxy_nested_loop = hip_block_xyz_loop<2,0,1>;
using hip_block_zyx_nested_loop = hip_block_xyz_loop<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<hip_block_xyz_loop<DIM0, DIM1>, SEGMENT> {

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

      for (int bx = internal::get_hip_dim<DIM0>(blockIdx);
           bx < len0;
           bx += internal::get_hip_dim<DIM0>(gridDim))
      {
        for (int by = internal::get_hip_dim<DIM1>(blockIdx);
             by < len1;
             by += internal::get_hip_dim<DIM1>(gridDim))
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by));
        }
      }
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<hip_block_xyz_loop<DIM0, DIM1, DIM2>, SEGMENT> {

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

    for (int bx = internal::get_hip_dim<DIM0>(blockIdx);
         bx < len0;
         bx += internal::get_hip_dim<DIM0>(gridDim))
    {

      for (int by = internal::get_hip_dim<DIM1>(blockIdx);
           by < len1;
           by += internal::get_hip_dim<DIM1>(gridDim))
      {

        for (int bz = internal::get_hip_dim<DIM2>(blockIdx);
             bz < len2;
             bz += internal::get_hip_dim<DIM2>(gridDim))
        {

          body(*(segment0.begin() + bx),
               *(segment1.begin() + by),
               *(segment2.begin() + bz));
        }
      }
    }
  }
};

/*
  perfectly nested hip loop policies + returns local index
*/
template <typename SEGMENT, int DIM0, int DIM1>
struct LoopICountExecute<hip_block_xyz_loop<DIM0, DIM1>, SEGMENT> {

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

      for (int bx = internal::get_hip_dim<DIM0>(blockIdx);
           bx < len0;
           bx += internal::get_hip_dim<DIM0>(gridDim))
      {
        for (int by = internal::get_hip_dim<DIM1>(blockIdx);
             by < len1;
             by += internal::get_hip_dim<DIM1>(gridDim))
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by), bx, by);
        }
      }
    }
  }
};


template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<hip_block_xyz_loop<DIM0, DIM1, DIM2>, SEGMENT> {

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

    for (int bx = internal::get_hip_dim<DIM0>(blockIdx);
         bx < len0;
         bx += internal::get_hip_dim<DIM0>(gridDim))
    {

      for (int by = internal::get_hip_dim<DIM1>(blockIdx);
           by < len1;
           by += internal::get_hip_dim<DIM1>(gridDim))
      {

        for (int bz = internal::get_hip_dim<DIM2>(blockIdx);
             bz < len2;
             bz += internal::get_hip_dim<DIM2>(gridDim))
        {

          body(*(segment0.begin() + bx),
               *(segment1.begin() + by),
               *(segment2.begin() + bz), bx, by, bz);
        }
      }
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<hip_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_hip_dim<DIM>(threadIdx) * tile_size;
         tx < len;
         tx += internal::get_hip_dim<DIM>(blockDim) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<hip_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_hip_dim<DIM>(threadIdx) * tile_size;
    if(tx < len)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<hip_block_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_hip_dim<DIM>(blockIdx) * tile_size;

         tx < len;

         tx += internal::get_hip_dim<DIM>(gridDim) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<hip_block_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_hip_dim<DIM>(blockIdx) * tile_size;
    if(tx < len){
      body(segment.slice(tx, tile_size));
    }
  }
};

//Tile execute + return index
template <typename SEGMENT, int DIM>
struct TileICountExecute<hip_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_hip_dim<DIM>(threadIdx) * tile_size;
         tx < len;
         tx += internal::get_hip_dim<DIM>(blockDim) * tile_size)
    {
      body(segment.slice(tx, tile_size), tx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<hip_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_hip_dim<DIM>(threadIdx) * tile_size;
    if(tx < len)
    {
      body(segment.slice(tx, tile_size), tx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<hip_block_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_hip_dim<DIM>(blockIdx) * tile_size;

         bx < len;

         bx += internal::get_hip_dim<DIM>(gridDim) * tile_size)
    {
      body(segment.slice(bx, tile_size), bx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<hip_block_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int bx = internal::get_hip_dim<DIM>(blockIdx) * tile_size;
    if(bx < len){
      body(segment.slice(bx, tile_size), bx/tile_size);
    }
  }
};

}  // namespace expt

}  // namespace RAJA
#endif
