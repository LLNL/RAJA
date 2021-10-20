/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::Teams::sycl
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_teams_sycl_HPP
#define RAJA_pattern_teams_sycl_HPP

#include "RAJA/pattern/teams/teams_core.hpp"
#include "RAJA/pattern/detail/privatizer.hpp"
#include "RAJA/policy/sycl/policy.hpp"
#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#include "RAJA/util/resource.hpp"

#include <CL/sycl.hpp>

namespace RAJA
{

namespace expt
{

template <bool async>
struct LaunchExecute<RAJA::expt::sycl_launch_t<async, 0>> {

  template <typename BODY_IN>
  static void exec(LaunchContext const &ctx, BODY_IN &&body_in)
  {

    cl::sycl::queue* q = ::RAJA::sycl::detail::getQueue();

    resources::Sycl sycl_res = resources::Sycl::get_default();
    
    // Global resource was not set, use the resource that was passed to forall
    // Determine if the default SYCL res is being used
    if (!q) { 
      q = sycl_res.get_queue();
    }

    const ::sycl::range<3> gridSize(ctx.teams.value[0],
			      ctx.teams.value[1],
			      ctx.teams.value[2]);

    const ::sycl::range<3> blockSize(ctx.threads.value[0],
			      ctx.threads.value[1],
			      ctx.threads.value[2]);


    q->submit([&](cl::sycl::handler& h) {

    h.parallel_for
      (cl::sycl::nd_range<3>{gridSize, blockSize},
       [=] (cl::sycl::nd_item<3> itm) {

	 ctx.setup_loc_id(itm.get_local_id(0),
		    itm.get_local_id(1),
		    itm.get_local_id(2));
	 
	 ctx.setup_group_id(itm.get_group(0),
			    itm.get_group(1),
			    itm.get_group(2));

	 body_in(ctx);
	 
       });                        
	 
    });

    if (!async) { q->wait(); }
    
  }

#if 0  
  template <typename BODY_IN>
  static resources::EventProxy<resources::Resource>
  exec(RAJA::resources::Resource res, LaunchContext const &ctx, BODY_IN &&body_in)
  {
    using BODY = camp::decay<BODY_IN>;

    auto func = launch_global_fcn<BODY>;

    /*Get the concrete resource */
    resources::Sycl sycl_res = res.get<RAJA::resources::Sycl>();

    //
    // Compute the number of blocks and threads
    //

    sycl_dim_t gridSize{ static_cast<sycl_dim_member_t>(ctx.teams.value[0]),
                         static_cast<sycl_dim_member_t>(ctx.teams.value[1]),
                         static_cast<sycl_dim_member_t>(ctx.teams.value[2]) };

    sycl_dim_t blockSize{ static_cast<sycl_dim_member_t>(ctx.threads.value[0]),
                          static_cast<sycl_dim_member_t>(ctx.threads.value[1]),
                          static_cast<sycl_dim_member_t>(ctx.threads.value[2]) };

    // Only launch kernel if we have something to iterate over
    constexpr sycl_dim_member_t zero = 0;
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
        BODY body = RAJA::sycl::make_launch_body(
            gridSize, blockSize, shmem, sycl_res, std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void *args[] = {(void*)&ctx, (void*)&body};
        {
          RAJA::sycl::launch((const void*)func, gridSize, blockSize, args, shmem, sycl_res, async, ctx.kernel_name);
        }
      }

      RAJA_FT_END;
    }

    return resources::EventProxy<resources::Resource>(res);
  }
#endif
};

#if 0
template <typename BODY, int num_threads>
__launch_bounds__(num_threads, 1) __global__
    void launch_global_fcn_fixed(LaunchContext ctx, BODY body_in)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body = privatizer.get_priv();
  body(ctx);
}


template <bool async, int nthreads>
struct LaunchExecute<RAJA::expt::sycl_launch_t<async, nthreads>> {

  template <typename BODY_IN>
  static void exec(LaunchContext const &ctx, BODY_IN &&body_in)
  {
    using BODY = camp::decay<BODY_IN>;

    auto func = launch_global_fcn_fixed<BODY, nthreads>;

    resources::Sycl sycl_res = resources::Sycl::get_default();

    //
    // Compute the number of blocks and threads
    //

    sycl_dim_t gridSize{ static_cast<sycl_dim_member_t>(ctx.teams.value[0]),
                         static_cast<sycl_dim_member_t>(ctx.teams.value[1]),
                         static_cast<sycl_dim_member_t>(ctx.teams.value[2]) };

    sycl_dim_t blockSize{ static_cast<sycl_dim_member_t>(ctx.threads.value[0]),
                          static_cast<sycl_dim_member_t>(ctx.threads.value[1]),
                          static_cast<sycl_dim_member_t>(ctx.threads.value[2]) };

    // Only launch kernel if we have something to iterate over
    constexpr sycl_dim_member_t zero = 0;
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
        BODY body = RAJA::sycl::make_launch_body(
            gridSize, blockSize, shmem, sycl_res, std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void *args[] = {(void*)&ctx, (void*)&body};
        RAJA::sycl::launch((const void*)func, gridSize, blockSize, args, shmem, sycl_res, async, ctx.kernel_name);
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

    /*Get the concrete resource */
    resources::Sycl sycl_res = res.get<RAJA::resources::Sycl>();

    //
    // Compute the number of blocks and threads
    //

    sycl_dim_t gridSize{ static_cast<sycl_dim_member_t>(ctx.teams.value[0]),
                         static_cast<sycl_dim_member_t>(ctx.teams.value[1]),
                         static_cast<sycl_dim_member_t>(ctx.teams.value[2]) };

    sycl_dim_t blockSize{ static_cast<sycl_dim_member_t>(ctx.threads.value[0]),
                          static_cast<sycl_dim_member_t>(ctx.threads.value[1]),
                          static_cast<sycl_dim_member_t>(ctx.threads.value[2]) };

    // Only launch kernel if we have something to iterate over
    constexpr sycl_dim_member_t zero = 0;
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
        BODY body = RAJA::sycl::make_launch_body(
            gridSize, blockSize, shmem, sycl_res, std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void *args[] = {(void*)&ctx, (void*)&body};
        {
          RAJA::sycl::launch((const void*)func, gridSize, blockSize, args, shmem, sycl_res, async, ctx.kernel_name);
        }
      }

      RAJA_FT_END;
    }

    return resources::EventProxy<resources::Resource>(res);
  }

};

#endif
  
/*
   SYCL global thread mapping
*/
template<int ... DIM>
struct sycl_global_thread;

using sycl_global_id_x = sycl_global_thread<0>;
using sycl_global_id_y = sycl_global_thread<1>;
using sycl_global_id_z = sycl_global_thread<2>;

template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_global_thread<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
     LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM>(ctx.loc_id) +
        ctx.threads.value[DIM]*
	internal::get_sycl_dim<DIM>(ctx.group_id);

      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};
  
using sycl_global_id_xy = sycl_global_thread<0,1>;
using sycl_global_id_xz = sycl_global_thread<0,2>;
using sycl_global_id_yx = sycl_global_thread<1,0>;
using sycl_global_id_yz = sycl_global_thread<1,2>;
using sycl_global_id_zx = sycl_global_thread<2,0>;
using sycl_global_id_zy = sycl_global_thread<2,1>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_global_thread<DIM0, DIM1>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(ctx.loc_id) +
        ctx.threads.value[DIM0]*internal::get_sycl_dim<DIM0>(ctx.group_id);

      const int ty = internal::get_sycl_dim<DIM1>(ctx.loc_id) +
        ctx.threads.value[DIM1]*internal::get_sycl_dim<DIM1>(ctx.group_id);

      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

using sycl_global_id_xyz = sycl_global_thread<0,1,2>;
using sycl_global_id_xzy = sycl_global_thread<0,2,1>;
using sycl_global_id_yxz = sycl_global_thread<1,0,2>;
using sycl_global_id_yzx = sycl_global_thread<1,2,0>;
using sycl_global_id_zxy = sycl_global_thread<2,0,1>;
using sycl_global_id_zyx = sycl_global_thread<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_global_thread<DIM0, DIM1, DIM2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(ctx.loc_id) +
        ctx.threads.value[DIM0]*internal::get_sycl_dim<DIM0>(ctx.group_id);

      const int ty = internal::get_sycl_dim<DIM1>(ctx.loc_id) +
        ctx.threads.value[DIM1]*internal::get_sycl_dim<DIM1>(ctx.group_id);

      const int tz = internal::get_sycl_dim<DIM2>(ctx.loc_id) +
        ctx.threads.value[DIM2]*internal::get_sycl_dim<DIM2>(ctx.group_id);

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
struct sycl_flatten_group_threads_direct{};

using sycl_flatten_group_threads_xy_direct = sycl_flatten_group_threads_direct<0,1>;
using sycl_flatten_group_threads_xz_direct = sycl_flatten_group_threads_direct<0,2>;
using sycl_flatten_group_threads_yx_direct = sycl_flatten_group_threads_direct<1,0>;
using sycl_flatten_group_threads_yz_direct = sycl_flatten_group_threads_direct<1,2>;
using sycl_flatten_group_threads_zx_direct = sycl_flatten_group_threads_direct<2,0>;
using sycl_flatten_group_threads_zy_direct = sycl_flatten_group_threads_direct<2,1>;

using sycl_flatten_group_threads_123_direct = sycl_flatten_group_threads_direct<0,1,2>;
using sycl_flatten_group_threads_xzy_direct = sycl_flatten_group_threads_direct<0,2,1>;
using sycl_flatten_group_threads_yxz_direct = sycl_flatten_group_threads_direct<1,0,2>;
using sycl_flatten_group_threads_yzx_direct = sycl_flatten_group_threads_direct<1,2,0>;
using sycl_flatten_group_threads_zxy_direct = sycl_flatten_group_threads_direct<2,0,1>;
using sycl_flatten_group_threads_zyx_direct = sycl_flatten_group_threads_direct<2,1,0>;

template<int ... dim>
struct sycl_flatten_group_threads_loop{};

using sycl_flatten_group_threads_xy_loop = sycl_flatten_group_threads_loop<0,1>;
using sycl_flatten_group_threads_xz_loop = sycl_flatten_group_threads_loop<0,2>;
using sycl_flatten_group_threads_yx_loop = sycl_flatten_group_threads_loop<1,0>;
using sycl_flatten_group_threads_yz_loop = sycl_flatten_group_threads_loop<1,2>;
using sycl_flatten_group_threads_zx_loop = sycl_flatten_group_threads_loop<2,0>;
using sycl_flatten_group_threads_zy_loop = sycl_flatten_group_threads_loop<2,1>;

using sycl_flatten_group_threads_123_loop = sycl_flatten_group_threads_loop<0,1,2>;
using sycl_flatten_group_threads_xzy_loop = sycl_flatten_group_threads_loop<0,2,1>;
using sycl_flatten_group_threads_yxz_loop = sycl_flatten_group_threads_loop<1,0,2>;
using sycl_flatten_group_threads_yzx_loop = sycl_flatten_group_threads_loop<1,2,0>;
using sycl_flatten_group_threads_zxy_loop = sycl_flatten_group_threads_loop<2,0,1>;
using sycl_flatten_group_threads_zyx_loop = sycl_flatten_group_threads_loop<2,1,0>;

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_flatten_group_threads_direct<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(ctx.loc_id);
      const int ty = internal::get_sycl_dim<DIM1>(ctx.loc_id);
      const int bx = internal::get_sycl_dim<DIM0>(ctx.group_id);
      const int tid = tx + bx*ty;

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_flatten_group_threads_loop<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = internal::get_sycl_dim<DIM0>(ctx.loc_id);
    const int ty = internal::get_sycl_dim<DIM1>(ctx.loc_id);
    const int bx = internal::get_sycl_dim<DIM0>(ctx.group_id);
    const int by = internal::get_sycl_dim<DIM1>(ctx.group_id);
    const int tid = tx + bx*ty;

    for(int tid = tx + bx*ty; tid < len; tid += bx*by) {
      body(*(segment.begin() + tid));
    }

  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_flatten_group_threads_direct<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(ctx.loc_id);
      const int ty = internal::get_sycl_dim<DIM1>(ctx.loc_id);
      const int tz = internal::get_sycl_dim<DIM2>(ctx.loc_id);
      const int bx = internal::get_sycl_dim<DIM0>(ctx.group_id);
      const int by = internal::get_sycl_dim<DIM1>(ctx.group_id);
      const int tid = tx + bx*(ty + by*tz);

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_flatten_group_threads_loop<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = internal::get_sycl_dim<DIM0>(ctx.loc_id);
    const int ty = internal::get_sycl_dim<DIM1>(ctx.loc_id);
    const int tz = internal::get_sycl_dim<DIM2>(ctx.loc_id);
    const int bx = internal::get_sycl_dim<DIM0>(ctx.group_id);
    const int by = internal::get_sycl_dim<DIM1>(ctx.group_id);
    const int bz = internal::get_sycl_dim<DIM2>(ctx.group_id);

    for(int tid = tx + bx*(ty + by*tz); tid < len; tid += bx*by*bz) {
      body(*(segment.begin() + tid));
    }

  }
};

/*
  SYCL thread loops with block strides
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_local_123_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(ctx.loc_id);
         tx < len;
         tx += internal::get_sycl_dim<DIM>(ctx.threads.value[DIM]) )
    {
      body(*(segment.begin() + tx));
    }
  }
};

#if 1

/*
  SYCL thread direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_local_123_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM>(ctx.loc_id);
      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};


/*
  SYCL block loops with grid strides
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_group_123_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_sycl_dim<DIM>(ctx.group_id);
         bx < len;
         bx += ctx.teams.value[DIM] ) {
      body(*(segment.begin() + bx));
    }
  }
};

/*
  SYCL block direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_group_123_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = internal::get_sycl_dim<DIM>(ctx.group_id);
      if (bx < len) body(*(segment.begin() + bx));
    }
  }
};

/*
  SYCL thread loops with block strides + Return Index
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_local_123_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(ctx.loc_id);
         tx < len;
         tx += ctx.threads.value[DIM] )
    {
      body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  SYCL thread direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_local_123_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM>(ctx.loc_id);
      if (tx < len) body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  SYCL block loops with grid strides
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_group_123_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_sycl_dim<DIM>(ctx.group_id);
         bx < len;
         bx += ctx.teams.value[DIM] ) {
      body(*(segment.begin() + bx), bx);
    }
  }
};

/*
  SYCL block direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_group_123_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = internal::get_sycl_dim<DIM>(ctx.group_id);
      if (bx < len) body(*(segment.begin() + bx), bx);
    }
  }
};

// perfectly nested sycl direct policies
using sycl_group_xy_nested_direct = sycl_group_123_direct<0,1>;
using sycl_group_xz_nested_direct = sycl_group_123_direct<0,2>;
using sycl_group_yx_nested_direct = sycl_group_123_direct<1,0>;
using sycl_group_yz_nested_direct = sycl_group_123_direct<1,2>;
using sycl_group_zx_nested_direct = sycl_group_123_direct<2,0>;
using sycl_group_zy_nested_direct = sycl_group_123_direct<2,1>;

using sycl_group_123_nested_direct = sycl_group_123_direct<0,1,2>;
using sycl_group_xzy_nested_direct = sycl_group_123_direct<0,2,1>;
using sycl_group_yxz_nested_direct = sycl_group_123_direct<1,0,2>;
using sycl_group_yzx_nested_direct = sycl_group_123_direct<1,2,0>;
using sycl_group_zxy_nested_direct = sycl_group_123_direct<2,0,1>;
using sycl_group_zyx_nested_direct = sycl_group_123_direct<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_group_123_direct<DIM0, DIM1>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(ctx.group_id);
      const int ty = internal::get_sycl_dim<DIM1>(ctx.group_id);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_group_123_direct<DIM0, DIM1, DIM2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(ctx.group_id);
      const int ty = internal::get_sycl_dim<DIM1>(ctx.group_id);
      const int tz = internal::get_sycl_dim<DIM2>(ctx.group_id);
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment2.begin() + tz));
    }
  }
};

/*
  Perfectly nested sycl direct policies
  Return local index
*/
template <typename SEGMENT, int DIM0, int DIM1>
struct LoopICountExecute<sycl_group_123_direct<DIM0, DIM1>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(ctx.group_id);
      const int ty = internal::get_sycl_dim<DIM1>(ctx.group_id);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty),
             tx, ty);
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<sycl_group_123_direct<DIM0, DIM1, DIM2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(ctx.group_id);
      const int ty = internal::get_sycl_dim<DIM1>(ctx.group_id);
      const int tz = internal::get_sycl_dim<DIM2>(ctx.group_id);
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment2.begin() + tz), tx, ty, tz);
    }
  }
};

// perfectly nested sycl loop policies
using sycl_group_xy_nested_loop = sycl_group_123_loop<0,1>;
using sycl_group_xz_nested_loop = sycl_group_123_loop<0,2>;
using sycl_group_yx_nested_loop = sycl_group_123_loop<1,0>;
using sycl_group_yz_nested_loop = sycl_group_123_loop<1,2>;
using sycl_group_zx_nested_loop = sycl_group_123_loop<2,0>;
using sycl_group_zy_nested_loop = sycl_group_123_loop<2,1>;

using sycl_group_123_nested_loop = sycl_group_123_loop<0,1,2>;
using sycl_group_xzy_nested_loop = sycl_group_123_loop<0,2,1>;
using sycl_group_yxz_nested_loop = sycl_group_123_loop<1,0,2>;
using sycl_group_yzx_nested_loop = sycl_group_123_loop<1,2,0>;
using sycl_group_zxy_nested_loop = sycl_group_123_loop<2,0,1>;
using sycl_group_zyx_nested_loop = sycl_group_123_loop<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_group_123_loop<DIM0, DIM1>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {

      for (int bx = internal::get_sycl_dim<DIM0>(ctx.group_id);
           bx < len0;
           bx += ctx.teams.value[DIM0])
      {
        for (int by = internal::get_sycl_dim<DIM1>(ctx.group_id);
             by < len1;
             by += ctx.teams.value[DIM1])
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by));
        }
      }
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_group_123_loop<DIM0, DIM1, DIM2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int bx = internal::get_sycl_dim<DIM0>(ctx.group_id);
         bx < len0;
         bx += ctx.teams.value[DIM0])
    {

      for (int by = internal::get_sycl_dim<DIM1>(ctx.group_id);
           by < len1;
           by += ctx.teams.value[DIM1])
      {

        for (int bz = internal::get_sycl_dim<DIM2>(ctx.group_id);
             bz < len2;
             bz += ctx.teams.value[DIM2])
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
  perfectly nested sycl loop policies + returns local index
*/
template <typename SEGMENT, int DIM0, int DIM1>
struct LoopICountExecute<sycl_group_123_loop<DIM0, DIM1>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      BODY const &body)
  {
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();
    {

      for (int bx = internal::get_sycl_dim<DIM0>(ctx.group_id);
           bx < len0;
           bx += ctx.teams.value[DIM0])
      {
        for (int by = internal::get_sycl_dim<DIM1>(ctx.group_id);
             by < len1;
             by += ctx.teams.value[DIM1])
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by), bx, by);
        }
      }
    }
  }
};


template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<sycl_group_123_loop<DIM0, DIM1, DIM2>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment0,
      SEGMENT const &segment1,
      SEGMENT const &segment2,
      BODY const &body)
  {
    const int len2 = segment2.end() - segment2.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len0 = segment0.end() - segment0.begin();

    for (int bx = internal::get_sycl_dim<DIM0>(ctx.group_id);
         bx < len0;
         bx += ctx.teams.value[DIM0])
    {

      for (int by = internal::get_sycl_dim<DIM1>(ctx.group_id);
           by < len1;
           by += ctx.teams.value[DIM1])
      {

        for (int bz = internal::get_sycl_dim<DIM2>(ctx.group_id);
             bz < len2;
             bz += ctx.teams.value[DIM2])
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
struct TileExecute<sycl_local_123_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(ctx.loc_id) * tile_size;
         tx < len;
         tx += ctx.threads.value[DIM] * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<sycl_local_123_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_sycl_dim<DIM>(ctx.loc_id) * tile_size;
    if(tx < len)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<sycl_group_123_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(ctx.group_id) * tile_size;

         tx < len;

         tx += ctx.teams.value[DIM] * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<sycl_group_123_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_sycl_dim<DIM>(ctx.group_id) * tile_size;
    if(tx < len){
      body(segment.slice(tx, tile_size));
    }
  }
};

//Tile execute + return index
template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_local_123_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(ctx.loc_id) * tile_size;
         tx < len;
         tx += ctx.threads.value[DIM] * tile_size)
    {
      body(segment.slice(tx, tile_size), tx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_local_123_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_sycl_dim<DIM>(ctx.loc_id) * tile_size;
    if(tx < len)
    {
      body(segment.slice(tx, tile_size), tx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_group_123_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_sycl_dim<DIM>(ctx.group_id) * tile_size;

         bx < len;

         bx += ctx.teams.value[DIM] * tile_size)
    {
      body(segment.slice(bx, tile_size), bx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_group_123_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int bx = internal::get_sycl_dim<DIM>(ctx.group_id) * tile_size;
    if(bx < len){
      body(segment.slice(bx, tile_size), bx/tile_size);
    }
  }
};
#endif

}  // namespace expt

}  // namespace RAJA

#endif
