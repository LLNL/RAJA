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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
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
//#include "RAJA/policy/sycl/raja_syclerrchk.hpp"
#include "RAJA/util/resource.hpp"

namespace RAJA
{

namespace expt
{

template <bool async> //switch 1 -> 0, but what should it be ? Ask R. Chen?
struct LaunchExecute<RAJA::expt::sycl_launch_t<async, 0>> {
// sycl_launch_t num_threads set to 1, but not used in launch of kernel

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

    std::cout<<"block size values "<<ctx.threads.value[0]<<" "<<ctx.threads.value[1]<<" "<<ctx.threads.value[2]<<std::endl;
    const ::sycl::range<3> blockSize(ctx.threads.value[0],
				     ctx.threads.value[1],
				     ctx.threads.value[2]);

    std::cout<<"number of blocks "<<ctx.teams.value[0]<<" "<<ctx.teams.value[1]<<" "<<ctx.teams.value[2]<<std::endl;
    const ::sycl::range<3> gridSize(ctx.threads.value[0] * ctx.teams.value[0],
				    ctx.threads.value[1] * ctx.teams.value[1],
				    ctx.threads.value[2] * ctx.teams.value[2]);


    q->submit([&](cl::sycl::handler& h) {

	auto s_vec = cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> (ctx.shared_mem_size, h);

	h.parallel_for
	  (cl::sycl::nd_range<3>(gridSize, blockSize),
	   [=] (cl::sycl::nd_item<3> itm) {

	    ctx.itm = &itm;

	    ctx.setup_loc_id(itm.get_local_id(0),
			     itm.get_local_id(1),
			     itm.get_local_id(2));

	    ctx.setup_group_id(itm.get_group(0),
			       itm.get_group(1),
			       itm.get_group(2));

	    //Point to shared memory
	    ctx.shared_mem_ptr = s_vec.get_pointer().get();

	    body_in(ctx);

	  });

      });

    if (!async) { q->wait(); }


  }

//Need to rework...
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

//Need to rework ...

#if 0
template <typename BODY, int num_threads, size_t BLOCKS_PER_SM>
__launch_bounds__(num_threads, BLOCKS_PER_SM) __global__
    void launch_global_fcn_fixed(LaunchContext ctx, BODY body_in)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body = privatizer.get_priv();
  body(ctx);
}

template <bool async, int nthreads, size_t BLOCKS_PER_SM>
struct LaunchExecute<RAJA::policy::sycl::expt::sycl_launch_explicit_t<async, nthreads, BLOCKS_PER_SM>> {

  template <typename BODY_IN>
  static void exec(LaunchContext const &ctx, BODY_IN &&body_in)
  {
    using BODY = camp::decay<BODY_IN>;

    auto func = launch_global_fcn_fixed<BODY, nthreads, BLOCKS_PER_SM>;

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

    auto func = launch_global_fcn_fixed<BODY, nthreads, BLOCKS_PER_SM>;

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


//Rework of the sycl policies

/*
  SYCL block direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_group_012_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const ctx,
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
  SYCL thread direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_local_012_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const ctx,
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

//================================================
//TODO rework rest of the sycl policies . . .
//================================================

#if 0
/*
   SYCL global thread mapping
*/
template<int ... DIM>
struct sycl_global_thread;

using sycl_global_thread_x = sycl_global_thread<0>;
using sycl_global_thread_y = sycl_global_thread<1>;
using sycl_global_thread_z = sycl_global_thread<2>;

template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_global_thread<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM>(threadIdx) +
        internal::get_sycl_dim<DIM>(blockDim)*internal::get_sycl_dim<DIM>(blockIdx);

      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};

using sycl_global_thread_xy = sycl_global_thread<0,1>;
using sycl_global_thread_xz = sycl_global_thread<0,2>;
using sycl_global_thread_yx = sycl_global_thread<1,0>;
using sycl_global_thread_yz = sycl_global_thread<1,2>;
using sycl_global_thread_zx = sycl_global_thread<2,0>;
using sycl_global_thread_zy = sycl_global_thread<2,1>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_global_thread<DIM0, DIM1>, SEGMENT> {

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
      const int tx = internal::get_sycl_dim<DIM0>(threadIdx) +
        internal::get_sycl_dim<DIM0>(blockDim)*internal::get_sycl_dim<DIM0>(blockIdx);

      const int ty = internal::get_sycl_dim<DIM1>(threadIdx) +
        internal::get_sycl_dim<DIM1>(blockDim)*internal::get_sycl_dim<DIM1>(blockIdx);

      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

using sycl_global_thread_xyz = sycl_global_thread<0,1,2>;
using sycl_global_thread_xzy = sycl_global_thread<0,2,1>;
using sycl_global_thread_yxz = sycl_global_thread<1,0,2>;
using sycl_global_thread_yzx = sycl_global_thread<1,2,0>;
using sycl_global_thread_zxy = sycl_global_thread<2,0,1>;
using sycl_global_thread_zyx = sycl_global_thread<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_global_thread<DIM0, DIM1, DIM2>, SEGMENT> {

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
      const int tx = internal::get_sycl_dim<DIM0>(threadIdx) +
        internal::get_sycl_dim<DIM0>(blockDim)*internal::get_sycl_dim<DIM0>(blockIdx);

      const int ty = internal::get_sycl_dim<DIM1>(threadIdx) +
        internal::get_sycl_dim<DIM1>(blockDim)*internal::get_sycl_dim<DIM1>(blockIdx);

      const int tz = internal::get_sycl_dim<DIM2>(threadIdx) +
        internal::get_sycl_dim<DIM2>(blockDim)*internal::get_sycl_dim<DIM2>(blockIdx);

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
struct sycl_flatten_block_threads_direct{};

using sycl_flatten_block_threads_xy_direct = sycl_flatten_block_threads_direct<0,1>;
using sycl_flatten_block_threads_xz_direct = sycl_flatten_block_threads_direct<0,2>;
using sycl_flatten_block_threads_yx_direct = sycl_flatten_block_threads_direct<1,0>;
using sycl_flatten_block_threads_yz_direct = sycl_flatten_block_threads_direct<1,2>;
using sycl_flatten_block_threads_zx_direct = sycl_flatten_block_threads_direct<2,0>;
using sycl_flatten_block_threads_zy_direct = sycl_flatten_block_threads_direct<2,1>;

using sycl_flatten_block_threads_xyz_direct = sycl_flatten_block_threads_direct<0,1,2>;
using sycl_flatten_block_threads_xzy_direct = sycl_flatten_block_threads_direct<0,2,1>;
using sycl_flatten_block_threads_yxz_direct = sycl_flatten_block_threads_direct<1,0,2>;
using sycl_flatten_block_threads_yzx_direct = sycl_flatten_block_threads_direct<1,2,0>;
using sycl_flatten_block_threads_zxy_direct = sycl_flatten_block_threads_direct<2,0,1>;
using sycl_flatten_block_threads_zyx_direct = sycl_flatten_block_threads_direct<2,1,0>;

template<int ... dim>
struct sycl_flatten_block_threads_loop{};

using sycl_flatten_block_threads_xy_loop = sycl_flatten_block_threads_loop<0,1>;
using sycl_flatten_block_threads_xz_loop = sycl_flatten_block_threads_loop<0,2>;
using sycl_flatten_block_threads_yx_loop = sycl_flatten_block_threads_loop<1,0>;
using sycl_flatten_block_threads_yz_loop = sycl_flatten_block_threads_loop<1,2>;
using sycl_flatten_block_threads_zx_loop = sycl_flatten_block_threads_loop<2,0>;
using sycl_flatten_block_threads_zy_loop = sycl_flatten_block_threads_loop<2,1>;

using sycl_flatten_block_threads_xyz_loop = sycl_flatten_block_threads_loop<0,1,2>;
using sycl_flatten_block_threads_xzy_loop = sycl_flatten_block_threads_loop<0,2,1>;
using sycl_flatten_block_threads_yxz_loop = sycl_flatten_block_threads_loop<1,0,2>;
using sycl_flatten_block_threads_yzx_loop = sycl_flatten_block_threads_loop<1,2,0>;
using sycl_flatten_block_threads_zxy_loop = sycl_flatten_block_threads_loop<2,0,1>;
using sycl_flatten_block_threads_zyx_loop = sycl_flatten_block_threads_loop<2,1,0>;

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_flatten_block_threads_direct<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(threadIdx);
      const int ty = internal::get_sycl_dim<DIM1>(threadIdx);
      const int bx = internal::get_sycl_dim<DIM0>(blockDim);
      const int tid = tx + bx*ty;

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_flatten_block_threads_loop<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = internal::get_sycl_dim<DIM0>(threadIdx);
    const int ty = internal::get_sycl_dim<DIM1>(threadIdx);

    const int bx = internal::get_sycl_dim<DIM0>(blockDim);
    const int by = internal::get_sycl_dim<DIM1>(blockDim);

    for(int tid = tx + bx*ty; tid < len; tid += bx*by) {
      body(*(segment.begin() + tid));
    }

  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_flatten_block_threads_direct<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM0>(threadIdx);
      const int ty = internal::get_sycl_dim<DIM1>(threadIdx);
      const int tz = internal::get_sycl_dim<DIM2>(threadIdx);
      const int bx = internal::get_sycl_dim<DIM0>(blockDim);
      const int by = internal::get_sycl_dim<DIM1>(blockDim);
      const int tid = tx + bx*(ty + by*tz);

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_flatten_block_threads_loop<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = internal::get_sycl_dim<DIM0>(threadIdx);
    const int ty = internal::get_sycl_dim<DIM1>(threadIdx);
    const int tz = internal::get_sycl_dim<DIM2>(threadIdx);
    const int bx = internal::get_sycl_dim<DIM0>(blockDim);
    const int by = internal::get_sycl_dim<DIM1>(blockDim);
    const int bz = internal::get_sycl_dim<DIM2>(blockDim);

    for(int tid = tx + bx*(ty + by*tz); tid < len; tid += bx*by*bz) {
      body(*(segment.begin() + tid));
    }

  }
};


/*
  SYCL thread loops with block strides
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(threadIdx);
         tx < len;
         tx += internal::get_sycl_dim<DIM>(blockDim) )
    {
      body(*(segment.begin() + tx));
    }
  }
};

/*
  SYCL thread direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM>(threadIdx);
      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};


/*
  SYCL block loops with grid strides
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_block_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_sycl_dim<DIM>(blockIdx);
         bx < len;
         bx += internal::get_sycl_dim<DIM>(gridDim) ) {
      body(*(segment.begin() + bx));
    }
  }
};

/*
  SYCL block direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_block_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = internal::get_sycl_dim<DIM>(blockIdx);
      if (bx < len) body(*(segment.begin() + bx));
    }
  }
};

/*
  SYCL thread loops with block strides + Return Index
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(threadIdx);
         tx < len;
         tx += internal::get_sycl_dim<DIM>(blockDim) )
    {
      body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  SYCL thread direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = internal::get_sycl_dim<DIM>(threadIdx);
      if (tx < len) body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  SYCL block loops with grid strides
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_block_xyz_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_sycl_dim<DIM>(blockIdx);
         bx < len;
         bx += internal::get_sycl_dim<DIM>(gridDim) ) {
      body(*(segment.begin() + bx), bx);
    }
  }
};

/*
  SYCL block direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_block_xyz_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = internal::get_sycl_dim<DIM>(blockIdx);
      if (bx < len) body(*(segment.begin() + bx), bx);
    }
  }
};

// perfectly nested sycl direct policies
using sycl_block_xy_nested_direct = sycl_block_xyz_direct<0,1>;
using sycl_block_xz_nested_direct = sycl_block_xyz_direct<0,2>;
using sycl_block_yx_nested_direct = sycl_block_xyz_direct<1,0>;
using sycl_block_yz_nested_direct = sycl_block_xyz_direct<1,2>;
using sycl_block_zx_nested_direct = sycl_block_xyz_direct<2,0>;
using sycl_block_zy_nested_direct = sycl_block_xyz_direct<2,1>;

using sycl_block_xyz_nested_direct = sycl_block_xyz_direct<0,1,2>;
using sycl_block_xzy_nested_direct = sycl_block_xyz_direct<0,2,1>;
using sycl_block_yxz_nested_direct = sycl_block_xyz_direct<1,0,2>;
using sycl_block_yzx_nested_direct = sycl_block_xyz_direct<1,2,0>;
using sycl_block_zxy_nested_direct = sycl_block_xyz_direct<2,0,1>;
using sycl_block_zyx_nested_direct = sycl_block_xyz_direct<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_block_xyz_direct<DIM0, DIM1>, SEGMENT> {

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
      const int tx = internal::get_sycl_dim<DIM0>(blockIdx);
      const int ty = internal::get_sycl_dim<DIM1>(blockIdx);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_block_xyz_direct<DIM0, DIM1, DIM2>, SEGMENT> {

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
      const int tx = internal::get_sycl_dim<DIM0>(blockIdx);
      const int ty = internal::get_sycl_dim<DIM1>(blockIdx);
      const int tz = internal::get_sycl_dim<DIM2>(blockIdx);
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
struct LoopICountExecute<sycl_block_xyz_direct<DIM0, DIM1>, SEGMENT> {

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
      const int tx = internal::get_sycl_dim<DIM0>(blockIdx);
      const int ty = internal::get_sycl_dim<DIM1>(blockIdx);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty),
             tx, ty);
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<sycl_block_xyz_direct<DIM0, DIM1, DIM2>, SEGMENT> {

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
      const int tx = internal::get_sycl_dim<DIM0>(blockIdx);
      const int ty = internal::get_sycl_dim<DIM1>(blockIdx);
      const int tz = internal::get_sycl_dim<DIM2>(blockIdx);
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment2.begin() + tz), tx, ty, tz);
    }
  }
};

// perfectly nested sycl loop policies
using sycl_block_xy_nested_loop = sycl_block_xyz_loop<0,1>;
using sycl_block_xz_nested_loop = sycl_block_xyz_loop<0,2>;
using sycl_block_yx_nested_loop = sycl_block_xyz_loop<1,0>;
using sycl_block_yz_nested_loop = sycl_block_xyz_loop<1,2>;
using sycl_block_zx_nested_loop = sycl_block_xyz_loop<2,0>;
using sycl_block_zy_nested_loop = sycl_block_xyz_loop<2,1>;

using sycl_block_xyz_nested_loop = sycl_block_xyz_loop<0,1,2>;
using sycl_block_xzy_nested_loop = sycl_block_xyz_loop<0,2,1>;
using sycl_block_yxz_nested_loop = sycl_block_xyz_loop<1,0,2>;
using sycl_block_yzx_nested_loop = sycl_block_xyz_loop<1,2,0>;
using sycl_block_zxy_nested_loop = sycl_block_xyz_loop<2,0,1>;
using sycl_block_zyx_nested_loop = sycl_block_xyz_loop<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_block_xyz_loop<DIM0, DIM1>, SEGMENT> {

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

      for (int bx = internal::get_sycl_dim<DIM0>(blockIdx);
           bx < len0;
           bx += internal::get_sycl_dim<DIM0>(gridDim))
      {
        for (int by = internal::get_sycl_dim<DIM1>(blockIdx);
             by < len1;
             by += internal::get_sycl_dim<DIM1>(gridDim))
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by));
        }
      }
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_block_xyz_loop<DIM0, DIM1, DIM2>, SEGMENT> {

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

    for (int bx = internal::get_sycl_dim<DIM0>(blockIdx);
         bx < len0;
         bx += internal::get_sycl_dim<DIM0>(gridDim))
    {

      for (int by = internal::get_sycl_dim<DIM1>(blockIdx);
           by < len1;
           by += internal::get_sycl_dim<DIM1>(gridDim))
      {

        for (int bz = internal::get_sycl_dim<DIM2>(blockIdx);
             bz < len2;
             bz += internal::get_sycl_dim<DIM2>(gridDim))
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
struct LoopICountExecute<sycl_block_xyz_loop<DIM0, DIM1>, SEGMENT> {

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

      for (int bx = internal::get_sycl_dim<DIM0>(blockIdx);
           bx < len0;
           bx += internal::get_sycl_dim<DIM0>(gridDim))
      {
        for (int by = internal::get_sycl_dim<DIM1>(blockIdx);
             by < len1;
             by += internal::get_sycl_dim<DIM1>(gridDim))
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by), bx, by);
        }
      }
    }
  }
};


template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<sycl_block_xyz_loop<DIM0, DIM1, DIM2>, SEGMENT> {

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

    for (int bx = internal::get_sycl_dim<DIM0>(blockIdx);
         bx < len0;
         bx += internal::get_sycl_dim<DIM0>(gridDim))
    {

      for (int by = internal::get_sycl_dim<DIM1>(blockIdx);
           by < len1;
           by += internal::get_sycl_dim<DIM1>(gridDim))
      {

        for (int bz = internal::get_sycl_dim<DIM2>(blockIdx);
             bz < len2;
             bz += internal::get_sycl_dim<DIM2>(gridDim))
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
struct TileExecute<sycl_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(threadIdx) * tile_size;
         tx < len;
         tx += internal::get_sycl_dim<DIM>(blockDim) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<sycl_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_sycl_dim<DIM>(threadIdx) * tile_size;
    if(tx < len)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<sycl_block_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(blockIdx) * tile_size;

         tx < len;

         tx += internal::get_sycl_dim<DIM>(gridDim) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<sycl_block_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_sycl_dim<DIM>(blockIdx) * tile_size;
    if(tx < len){
      body(segment.slice(tx, tile_size));
    }
  }
};

//Tile execute + return index
template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_thread_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = internal::get_sycl_dim<DIM>(threadIdx) * tile_size;
         tx < len;
         tx += internal::get_sycl_dim<DIM>(blockDim) * tile_size)
    {
      body(segment.slice(tx, tile_size), tx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_thread_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = internal::get_sycl_dim<DIM>(threadIdx) * tile_size;
    if(tx < len)
    {
      body(segment.slice(tx, tile_size), tx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_block_xyz_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = internal::get_sycl_dim<DIM>(blockIdx) * tile_size;

         bx < len;

         bx += internal::get_sycl_dim<DIM>(gridDim) * tile_size)
    {
      body(segment.slice(bx, tile_size), bx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_block_xyz_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int bx = internal::get_sycl_dim<DIM>(blockIdx) * tile_size;
    if(bx < len){
      body(segment.slice(bx, tile_size), bx/tile_size);
    }
  }
};

#endif
}  // namespace expt

}  // namespace RAJA
#endif
