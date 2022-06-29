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

    const ::sycl::range<3> blockSize(ctx.threads.value[0],
				     ctx.threads.value[1],
				     ctx.threads.value[2]);

    const ::sycl::range<3> gridSize(ctx.threads.value[0] * ctx.teams.value[0],
				    ctx.threads.value[1] * ctx.teams.value[1],
				    ctx.threads.value[2] * ctx.teams.value[2]);

        // Only launch kernel if we have something to iterate over
    constexpr size_t zero = 0;
    if ( ctx.threads.value[0]  > zero && ctx.threads.value[1]  > zero && ctx.threads.value[2] > zero &&
         ctx.teams.value[0] > zero && ctx.teams.value[1] > zero && ctx.teams.value[2]> zero ) {

      RAJA_FT_BEGIN

      q->submit([&](cl::sycl::handler& h) {

        auto s_vec = cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write,
                                        cl::sycl::access::target::local> (ctx.shared_mem_size, h);

        h.parallel_for
          (cl::sycl::nd_range<3>(gridSize, blockSize),
           [=] (cl::sycl::nd_item<3> itm) {

             ctx.itm = &itm;

             //Point to shared memory
             ctx.shared_mem_ptr = s_vec.get_pointer().get();

             body_in(ctx);

           });

      });

      RAJA_FT_END;
    }

    if (!async) { q->wait(); }
  }


  template <typename BODY_IN>
  static resources::EventProxy<resources::Resource>
  exec(RAJA::resources::Resource res, LaunchContext const &ctx, BODY_IN &&body_in)
  {

    cl::sycl::queue* q = ::RAJA::sycl::detail::getQueue();

    /*Get the concrete resource */
    resources::Sycl sycl_res = res.get<RAJA::resources::Sycl>();

    // Global resource was not set, use the resource that was passed to forall
    // Determine if the default SYCL res is being used
    if (!q) {
      q = sycl_res.get_queue();
    }

    //
    // Compute the number of blocks and threads
    //

    const ::sycl::range<3> blockSize(ctx.threads.value[0],
				     ctx.threads.value[1],
				     ctx.threads.value[2]);

    const ::sycl::range<3> gridSize(ctx.threads.value[0] * ctx.teams.value[0],
				    ctx.threads.value[1] * ctx.teams.value[1],
				    ctx.threads.value[2] * ctx.teams.value[2]);

    // Only launch kernel if we have something to iterate over
    constexpr size_t zero = 0;
    if ( ctx.threads.value[0]  > zero && ctx.threads.value[1]  > zero && ctx.threads.value[2] > zero &&
         ctx.teams.value[0] > zero && ctx.teams.value[1] > zero && ctx.teams.value[2]> zero ) {

      RAJA_FT_BEGIN;

      q->submit([&](cl::sycl::handler& h) {

        auto s_vec = cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write,
                                        cl::sycl::access::target::local> (ctx.shared_mem_size, h);

        h.parallel_for
          (cl::sycl::nd_range<3>(gridSize, blockSize),
           [=] (cl::sycl::nd_item<3> itm) {

             ctx.itm = &itm;

             //Point to shared memory
             ctx.shared_mem_ptr = s_vec.get_pointer().get();

             body_in(ctx);

           });

      });

      RAJA_FT_END;

    }

    return resources::EventProxy<resources::Resource>(res);
  }


};

//Need to rework ...
//Question: Does SYCL support launch bounds??
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

//================================================
//TODO rework rest of the sycl policies . . .
//================================================

/*
   SYCL global thread mapping
*/
template<int ... DIM>
struct sycl_global_thread;

using sycl_global_thread_0 = sycl_global_thread<0>;
using sycl_global_thread_1 = sycl_global_thread<1>;
using sycl_global_thread_2 = sycl_global_thread<2>;

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
      const int tx =
        ctx.itm->get_group(DIM) * ctx.itm->get_local_range(DIM) +
        ctx.itm->get_local_id(DIM);

      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};

using sycl_global_thread_01 = sycl_global_thread<0,1>;
using sycl_global_thread_02 = sycl_global_thread<0,2>;
using sycl_global_thread_10 = sycl_global_thread<1,0>;
using sycl_global_thread_12 = sycl_global_thread<1,2>;
using sycl_global_thread_20 = sycl_global_thread<2,0>;
using sycl_global_thread_21 = sycl_global_thread<2,1>;

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
      const int tx =
        ctx.itm->get_group(DIM0) * ctx.itm->get_local_range(DIM0) +
        ctx.itm->get_local_id(DIM0);

      const int ty =
        ctx.itm->get_group(DIM1) * ctx.itm->get_local_range(DIM1) +
        ctx.itm->get_local_id(DIM1);


      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};


using sycl_global_thread_012 = sycl_global_thread<0,1,2>;
using sycl_global_thread_021 = sycl_global_thread<0,2,1>;
using sycl_global_thread_102 = sycl_global_thread<1,0,2>;
using sycl_global_thread_120 = sycl_global_thread<1,2,0>;
using sycl_global_thread_201 = sycl_global_thread<2,0,1>;
using sycl_global_thread_210 = sycl_global_thread<2,1,0>;

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
      const int tx =
        ctx.itm->get_group(DIM0) * ctx.itm->get_local_range(DIM0) +
        ctx.itm->get_local_id(DIM0);

      const int ty =
        ctx.itm->get_group(DIM1) * ctx.itm->get_local_range(DIM1) +
        ctx.itm->get_local_id(DIM1);

      const int tz =
        ctx.itm->get_group(DIM2) * ctx.itm->get_local_range(DIM2) +
        ctx.itm->get_local_id(DIM2);

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
struct sycl_flatten_group_local_direct{};

using sycl_flatten_group_local_01_direct = sycl_flatten_group_local_direct<0,1>;
using sycl_flatten_group_local_02_direct = sycl_flatten_group_local_direct<0,2>;
using sycl_flatten_group_local_10_direct = sycl_flatten_group_local_direct<1,0>;
using sycl_flatten_group_local_12_direct = sycl_flatten_group_local_direct<1,2>;
using sycl_flatten_group_local_20_direct = sycl_flatten_group_local_direct<2,0>;
using sycl_flatten_group_local_21_direct = sycl_flatten_group_local_direct<2,1>;

using sycl_flatten_group_local_012_direct = sycl_flatten_group_local_direct<0,1,2>;
using sycl_flatten_group_local_021_direct = sycl_flatten_group_local_direct<0,2,1>;
using sycl_flatten_group_local_102_direct = sycl_flatten_group_local_direct<1,0,2>;
using sycl_flatten_group_local_120_direct = sycl_flatten_group_local_direct<1,2,0>;
using sycl_flatten_group_local_201_direct = sycl_flatten_group_local_direct<2,0,1>;
using sycl_flatten_group_local_210_direct = sycl_flatten_group_local_direct<2,1,0>;

template<int ... dim>
struct sycl_flatten_group_local_loop{};

using sycl_flatten_group_local_01_loop = sycl_flatten_group_local_loop<0,1>;
using sycl_flatten_group_local_02_loop = sycl_flatten_group_local_loop<0,2>;
using sycl_flatten_group_local_10_loop = sycl_flatten_group_local_loop<1,0>;
using sycl_flatten_group_local_12_loop = sycl_flatten_group_local_loop<1,2>;
using sycl_flatten_group_local_20_loop = sycl_flatten_group_local_loop<2,0>;
using sycl_flatten_group_local_21_loop = sycl_flatten_group_local_loop<2,1>;

using sycl_flatten_group_local_012_loop = sycl_flatten_group_local_loop<0,1,2>;
using sycl_flatten_group_local_021_loop = sycl_flatten_group_local_loop<0,2,1>;
using sycl_flatten_group_local_102_loop = sycl_flatten_group_local_loop<1,0,2>;
using sycl_flatten_group_local_120_loop = sycl_flatten_group_local_loop<1,2,0>;
using sycl_flatten_group_local_201_loop = sycl_flatten_group_local_loop<2,0,1>;
using sycl_flatten_group_local_210_loop = sycl_flatten_group_local_loop<2,1,0>;

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_flatten_group_local_direct<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = ctx.itm->get_local_id(DIM0);
      const int ty = ctx.itm->get_local_id(DIM1);
      const int bx = ctx.itm->get_local_range(DIM0);
      const int tid = tx + bx*ty;

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_flatten_group_local_loop<DIM0, DIM1>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = ctx.itm->get_local_id(DIM0);
    const int ty = ctx.itm->get_local_id(DIM1);

    const int bx = ctx.itm->get_local_range(DIM0);
    const int by = ctx.itm->get_local_range(DIM1);

    for(int tid = tx + bx*ty; tid < len; tid += bx*by) {
      body(*(segment.begin() + tid));
    }

  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_flatten_group_local_direct<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();
    {
      const int tx = ctx.itm->get_local_id(DIM0);
      const int ty = ctx.itm->get_local_id(DIM1);
      const int tz = ctx.itm->get_local_id(DIM2);
      const int bx = ctx.itm->get_local_range(DIM0);
      const int by = ctx.itm->get_local_range(DIM1);

      const int tid = tx + bx*(ty + by*tz);

      if (tid < len) body(*(segment.begin() + tid));
    }
  }
};

template<typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_flatten_group_local_loop<DIM0, DIM1, DIM2>, SEGMENT>
{
  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {
    const int len = segment.end() - segment.begin();

    const int tx = ctx.itm->get_local_id(DIM0);
    const int ty = ctx.itm->get_local_id(DIM1);
    const int tz = ctx.itm->get_local_id(DIM2);
    const int bx = ctx.itm->get_local_range(DIM0);
    const int by = ctx.itm->get_local_range(DIM1);
    const int bz = ctx.itm->get_local_range(DIM2);

    for(int tid = tx + bx*(ty + by*tz); tid < len; tid += bx*by*bz) {
      body(*(segment.begin() + tid));
    }

  }
};

/*
  SYCL thread loops with block strides
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_local_012_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_local_id(DIM);
         tx < len;
         tx += ctx.itm->get_local_range(DIM))
    {
      body(*(segment.begin() + tx));
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
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = ctx.itm->get_local_id(DIM);
      if (tx < len) body(*(segment.begin() + tx));
    }
  }
};

/*
  SYCL block loops with grid strides
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_group_012_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = ctx.itm->get_group(DIM);
         bx < len;
         bx += ctx.itm->get_group_range(DIM) ) {
      body(*(segment.begin() + bx));
    }
  }
};

/*
  SYCL block direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopExecute<sycl_group_012_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = ctx.itm->get_group(DIM);
      if (bx < len) body(*(segment.begin() + bx));
    }
  }
};

/*
  SYCL thread loops with block strides + Return Index
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_local_012_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_local_id(DIM);
         tx < len;
         tx += ctx.itm->get_local_range(DIM) )
    {
      body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  SYCL thread direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_local_012_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int tx = ctx.itm->get_local_id(DIM);
      if (tx < len) body(*(segment.begin() + tx), tx);
    }
  }
};

/*
  SYCL block loops with grid strides
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_group_012_loop<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx =  ctx.itm->get_group(DIM);
         bx < len;
         bx += ctx.itm->get_group_range(DIM) ) {
      body(*(segment.begin() + bx), bx);
    }
  }
};

/*
  SYCL block direct mappings
*/
template <typename SEGMENT, int DIM>
struct LoopICountExecute<sycl_group_012_direct<DIM>, SEGMENT> {

  template <typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();
    {
      const int bx = ctx.itm->get_group(DIM);
      if (bx < len) body(*(segment.begin() + bx), bx);
    }
  }
};

// perfectly nested sycl direct policies
using sycl_group_01_nested_direct = sycl_group_012_direct<0,1>;
using sycl_group_02_nested_direct = sycl_group_012_direct<0,2>;
using sycl_group_10_nested_direct = sycl_group_012_direct<1,0>;
using sycl_group_12_nested_direct = sycl_group_012_direct<1,2>;
using sycl_group_20_nested_direct = sycl_group_012_direct<2,0>;
using sycl_group_21_nested_direct = sycl_group_012_direct<2,1>;

using sycl_group_012_nested_direct = sycl_group_012_direct<0,1,2>;
using sycl_group_021_nested_direct = sycl_group_012_direct<0,2,1>;
using sycl_group_102_nested_direct = sycl_group_012_direct<1,0,2>;
using sycl_group_120_nested_direct = sycl_group_012_direct<1,2,0>;
using sycl_group_201_nested_direct = sycl_group_012_direct<2,0,1>;
using sycl_group_210_nested_direct = sycl_group_012_direct<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_group_012_direct<DIM0, DIM1>, SEGMENT> {

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
      const int tx = ctx.itm->get_group(DIM0);
      const int ty = ctx.itm->get_group(DIM1);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty));
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_group_012_direct<DIM0, DIM1, DIM2>, SEGMENT> {

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
      const int tx = ctx.itm->get_group(DIM0);
      const int ty = ctx.itm->get_group(DIM1);
      const int tz = ctx.itm->get_group(DIM2);
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
struct LoopICountExecute<sycl_group_012_direct<DIM0, DIM1>, SEGMENT> {

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
      const int tx =  ctx.itm->get_group(DIM0);
      const int ty =  ctx.itm->get_group(DIM1);
      if (tx < len0 && ty < len1)
        body(*(segment0.begin() + tx), *(segment1.begin() + ty),
             tx, ty);
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<sycl_group_012_direct<DIM0, DIM1, DIM2>, SEGMENT> {

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
      const int tx = ctx.itm->get_group(DIM0);
      const int ty = ctx.itm->get_group(DIM1);
      const int tz = ctx.itm->get_group(DIM2);
      if (tx < len0 && ty < len1 && tz < len2)
        body(*(segment0.begin() + tx),
             *(segment1.begin() + ty),
             *(segment2.begin() + tz), tx, ty, tz);
    }
  }
};

// perfectly nested sycl loop policies
using sycl_group_01_nested_loop = sycl_group_012_loop<0,1>;
using sycl_group_02_nested_loop = sycl_group_012_loop<0,2>;
using sycl_group_10_nested_loop = sycl_group_012_loop<1,0>;
using sycl_group_12_nested_loop = sycl_group_012_loop<1,2>;
using sycl_group_20_nested_loop = sycl_group_012_loop<2,0>;
using sycl_group_21_nested_loop = sycl_group_012_loop<2,1>;

using sycl_group_012_nested_loop = sycl_group_012_loop<0,1,2>;
using sycl_group_021_nested_loop = sycl_group_012_loop<0,2,1>;
using sycl_group_102_nested_loop = sycl_group_012_loop<1,0,2>;
using sycl_group_120_nested_loop = sycl_group_012_loop<1,2,0>;
using sycl_group_201_nested_loop = sycl_group_012_loop<2,0,1>;
using sycl_group_210_nested_loop = sycl_group_012_loop<2,1,0>;

template <typename SEGMENT, int DIM0, int DIM1>
struct LoopExecute<sycl_group_012_loop<DIM0, DIM1>, SEGMENT> {

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

      for (int bx = ctx.itm->get_group(DIM0);
           bx < len0;
           bx += ctx.itm->get_group_range(DIM0))
      {
        for (int by = ctx.itm->get_group(DIM1);
             by < len1;
             bx += ctx.itm->get_group_range(DIM1))
        {
          body(*(segment0.begin() + bx), *(segment1.begin() + by));
        }
      }
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopExecute<sycl_group_012_loop<DIM0, DIM1, DIM2>, SEGMENT> {

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

    for (int bx = ctx.itm->get_group(DIM0);
         bx < len0;
         bx += ctx.itm->get_group_range(DIM0))
    {

      for (int by = ctx.itm->get_group(DIM1);
           by < len1;
           by += ctx.itm->get_group_range(DIM1))
      {

        for (int bz = ctx.itm->get_group(DIM2);
             bz < len2;
             bz += ctx.itm->get_group_range(DIM2))
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
struct LoopICountExecute<sycl_group_012_loop<DIM0, DIM1>, SEGMENT> {

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

      for (int bx = ctx.itm->get_group(DIM0);
           bx < len0;
           bx += ctx.itm->get_group_range(DIM0))
      {
        for (int by = ctx.itm->get_group(DIM0);
             by < len1;
             by += ctx.itm->get_group_range(DIM1))
        {

          body(*(segment0.begin() + bx), *(segment1.begin() + by), bx, by);
        }
      }
    }
  }
};

template <typename SEGMENT, int DIM0, int DIM1, int DIM2>
struct LoopICountExecute<sycl_group_012_loop<DIM0, DIM1, DIM2>, SEGMENT> {

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

    for (int bx = ctx.itm->get_group(DIM0);
         bx < len0;
         bx += ctx.itm->get_group_range(DIM0))
    {

      for (int by = ctx.itm->get_group(DIM0);
           by < len1;
           by += ctx.itm->get_group_range(DIM0))
      {

        for (int bz =  ctx.itm->get_group(DIM0);
             bz < len2;
             bz += ctx.itm->get_group_range(DIM0))
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
struct TileExecute<sycl_local_012_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_local_id(DIM) * tile_size;
         tx < len;
         tx += ctx.itm->get_local_range(DIM) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<sycl_local_012_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = ctx.itm->get_local_id(DIM) * tile_size;
    if(tx < len)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileExecute<sycl_group_012_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_group(DIM)* tile_size;

         tx < len;

         tx += ctx.itm->get_group_range(DIM) * tile_size)
    {
      body(segment.slice(tx, tile_size));
    }
  }
};

template <typename SEGMENT, int DIM>
struct TileExecute<sycl_group_012_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = ctx.itm->get_group(DIM) * tile_size;
    if(tx < len){
      body(segment.slice(tx, tile_size));
    }
  }
};

//Tile execute + return index
template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_local_012_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int tx = ctx.itm->get_local_id(DIM) * tile_size;
         tx < len;
         tx += ctx.itm->get_local_range(DIM) * tile_size)
    {
      body(segment.slice(tx, tile_size), tx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_local_012_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int tx = ctx.itm->get_local_id(DIM) * tile_size;
    if(tx < len)
    {
      body(segment.slice(tx, tile_size), tx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_group_012_loop<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    for (int bx = ctx.itm->get_group(DIM) * tile_size;
         bx < len;
         bx += ctx.itm->get_group_range(DIM) * tile_size)
    {
      body(segment.slice(bx, tile_size), bx/tile_size);
    }
  }
};


template <typename SEGMENT, int DIM>
struct TileICountExecute<sycl_group_012_direct<DIM>, SEGMENT> {

  template <typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      TILE_T tile_size,
      SEGMENT const &segment,
      BODY const &body)
  {

    const int len = segment.end() - segment.begin();

    int bx = ctx.itm->get_group(DIM) * tile_size;
    if(bx < len){
      body(segment.slice(bx, tile_size), bx/tile_size);
    }
  }
};


}  // namespace expt

}  // namespace RAJA
#endif
