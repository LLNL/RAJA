/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for RAJA::launch::hip
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_launch_hip_HPP
#define RAJA_pattern_launch_hip_HPP

#include "RAJA/pattern/launch/launch_core.hpp"
#include "RAJA/pattern/detail/privatizer.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#include "RAJA/util/resource.hpp"

namespace RAJA
{

// internal helper function
namespace detail
{

template<typename T, size_t... I>
constexpr T multiply_impl(const std::array<T, sizeof...(I)>& arr,
                          std::index_sequence<I...>)
{
  return (arr[I] * ...);
}

template<typename T, size_t N>
constexpr T multiplyArray(const std::array<T, N>& arr)
{
  return multiply_impl(arr, std::make_index_sequence<N> {});
}

}  // namespace detail

template<typename BODY>
__global__ void launch_global_fcn(BODY body_in)
{
  LaunchContext ctx;

  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body      = privatizer.get_priv();

  // Set pointer to shared memory
  extern __shared__ char raja_shmem_ptr[];
  ctx.shared_mem_ptr = raja_shmem_ptr;

  body(ctx);
}

template<typename BODY>
__global__ void launch_global_fcn_ctx(BODY body_in, LaunchContext ctx)
{
  //LaunchContext ctx;

  //unravel index
  int tid = threadIdx.x;
  for (int d = ctx.thread_dim.size()-1; d >= 0; --d) {
    ctx.thread_id[d] = tid % ctx.thread_dim[d];
    tid /= ctx.thread_dim[d];
  }

  /*
  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    for(int i=0; i<ctx.threads.size(); ++i) {
      printf("ctx.threads[i] = %d \n", ctx.threads[i]);
    }
  }
  */

  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body      = privatizer.get_priv();

  // Set pointer to shared memory
  extern __shared__ char raja_shmem_ptr[];
  ctx.shared_mem_ptr = raja_shmem_ptr;

  body(ctx);
}

template<typename BODY, typename ReduceParams>
__global__ void launch_new_reduce_global_fcn(BODY body_in,
                                             ReduceParams reduce_params)
{
  LaunchContext ctx;

  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body      = privatizer.get_priv();

  // Set pointer to shared memory
  extern __shared__ char raja_shmem_ptr[];
  ctx.shared_mem_ptr = raja_shmem_ptr;

  RAJA::expt::invoke_body(reduce_params, body, ctx);

  // Using a flatten global policy as we may use all dimensions
  RAJA::expt::ParamMultiplexer::parampack_combine(
      RAJA::hip_flatten_global_xyz_direct {}, reduce_params);
}

template<bool async>
struct LaunchExecute<
    RAJA::policy::hip::hip_launch_t<async, named_usage::unspecified>>
{

  template<size_t ThreadDIM = 3, typename BODY_IN, typename ReduceParams>
  static concepts::enable_if_t<
      resources::EventProxy<resources::Resource>,
      RAJA::expt::type_traits::is_ForallParamPack<ReduceParams>,
      RAJA::expt::type_traits::is_ForallParamPack_empty<ReduceParams>>
  exec(RAJA::resources::Resource res,
       const LaunchParams<ThreadDIM>& params,
       BODY_IN&& body_in,
       ReduceParams& RAJA_UNUSED_ARG(launch_reducers))
  {
    using BODY = camp::decay<BODY_IN>;

    auto func = reinterpret_cast<const void*>(&launch_global_fcn_ctx<BODY>);

    resources::Hip hip_res = res.get<RAJA::resources::Hip>();

    if (params.threads.value.size() > 3)
    {
      std::cout << "threads container is larger than 3 : "
                << params.threads.value.size() << std::endl;
    }


    //
    // Compute the number of blocks and threads
    //

    hip_dim_t gridSize {static_cast<hip_dim_member_t>(params.teams.value[0]),
                        static_cast<hip_dim_member_t>(params.teams.value[1]),
                        static_cast<hip_dim_member_t>(params.teams.value[2])};

    /*
    hip_dim_t blockSize {
        static_cast<hip_dim_member_t>(params.threads.value[0]),
        static_cast<hip_dim_member_t>(params.threads.value[1]),
        static_cast<hip_dim_member_t>(params.threads.value[2])};
    */

    hip_dim_t blockSize;

    if (params.threads.value.size() < 4)
    {
      blockSize =
          hip_dim_t {static_cast<hip_dim_member_t>(params.threads.value[0]),
                      static_cast<hip_dim_member_t>(params.threads.value[1]),
                      static_cast<hip_dim_member_t>(params.threads.value[2])};
    }
    else
    {

      int total_threads = detail::multiplyArray(params.threads.value);
      std::cout << "Total threads" << std::endl;
      blockSize = hip_dim_t {static_cast<hip_dim_member_t>(
                                  detail::multiplyArray(params.threads.value)),
                              static_cast<hip_dim_member_t>(1),
                              static_cast<hip_dim_member_t>(1)};
    }


    // Only launch kernel if we have something to iterate over
    constexpr hip_dim_member_t zero = 0;
    if (gridSize.x > zero && gridSize.y > zero && gridSize.z > zero &&
        blockSize.x > zero && blockSize.y > zero && blockSize.z > zero)
    {

      RAJA_FT_BEGIN;

      {
        size_t shared_mem_size = params.shared_mem_size;
        std::cout << "launching kernel " << std::endl;
        //
        // Privatize the loop_body, using make_launch_body to setup reductions
        //
        BODY body = RAJA::hip::make_launch_body(func, gridSize, blockSize,
                                                shared_mem_size, hip_res,
                                                std::forward<BODY_IN>(body_in));

        //Copy threads over
        LaunchContext ctx;
        ctx.thread_dim = params.threads.value;

        //
        // Launch the kernel
        //
        void* args[] = {(void*)&body, (void*)&ctx};
        RAJA::hip::launch(func, gridSize, blockSize, args, shared_mem_size,
                          hip_res, async);
      }

      RAJA_FT_END;
    }
    else
    {

      std::cout << "did not launch kernel " << std::endl;
    }

    return resources::EventProxy<resources::Resource>(res);
  }

  // Version with explicit reduction parameters..
  template<size_t ThreadDIM = 3, typename BODY_IN, typename ReduceParams>
  static concepts::enable_if_t<
      resources::EventProxy<resources::Resource>,
      RAJA::expt::type_traits::is_ForallParamPack<ReduceParams>,
      concepts::negate<
          RAJA::expt::type_traits::is_ForallParamPack_empty<ReduceParams>>>
  exec(RAJA::resources::Resource res,
       const LaunchParams<ThreadDIM>& launch_params,
       BODY_IN&& body_in,
       ReduceParams& launch_reducers)
  {
    using BODY = camp::decay<BODY_IN>;
    using EXEC_POL =
        RAJA::policy::hip::hip_launch_t<async, named_usage::unspecified>;
    EXEC_POL pol {};

    auto func = reinterpret_cast<const void*>(
        &launch_new_reduce_global_fcn<BODY, camp::decay<ReduceParams>>);

    resources::Hip hip_res = res.get<RAJA::resources::Hip>();

    //
    // Compute the number of blocks and threads
    //

    hip_dim_t gridSize {
        static_cast<hip_dim_member_t>(launch_params.teams.value[0]),
        static_cast<hip_dim_member_t>(launch_params.teams.value[1]),
        static_cast<hip_dim_member_t>(launch_params.teams.value[2])};

    hip_dim_t blockSize {
        static_cast<hip_dim_member_t>(launch_params.threads.value[0]),
        static_cast<hip_dim_member_t>(launch_params.threads.value[1]),
        static_cast<hip_dim_member_t>(launch_params.threads.value[2])};

    // Only launch kernel if we have something to iterate over
    constexpr hip_dim_member_t zero = 0;
    if (gridSize.x > zero && gridSize.y > zero && gridSize.z > zero &&
        blockSize.x > zero && blockSize.y > zero && blockSize.z > zero)
    {

      RAJA_FT_BEGIN;

      size_t shared_mem_size = launch_params.shared_mem_size;
      RAJA::hip::detail::hipInfo launch_info;
      launch_info.gridDim      = gridSize;
      launch_info.blockDim     = blockSize;
      launch_info.dynamic_smem = &shared_mem_size;
      launch_info.res          = hip_res;

      {

        RAJA::expt::ParamMultiplexer::parampack_init(pol, launch_reducers,
                                                     launch_info);

        //
        // Privatize the loop_body, using make_launch_body to setup reductions
        //
        BODY body = RAJA::hip::make_launch_body(func, gridSize, blockSize,
                                                shared_mem_size, hip_res,
                                                std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void* args[] = {(void*)&body, (void*)&launch_reducers};
        RAJA::hip::launch(func, gridSize, blockSize, args, shared_mem_size,
                          hip_res, async);

        RAJA::expt::ParamMultiplexer::parampack_resolve(pol, launch_reducers,
                                                        launch_info);
      }

      RAJA_FT_END;
    }

    return resources::EventProxy<resources::Resource>(res);
  }
};

template<typename BODY, int num_threads>
__launch_bounds__(num_threads, 1) __global__
    void launch_global_fcn_fixed(BODY body_in)
{
  LaunchContext ctx;

  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body      = privatizer.get_priv();

  // Set pointer to shared memory
  extern __shared__ char raja_shmem_ptr[];
  ctx.shared_mem_ptr = raja_shmem_ptr;

  body(ctx);
}

template<typename BODY, int num_threads, typename ReduceParams>
__launch_bounds__(num_threads, 1) __global__
    void launch_new_reduce_global_fcn_fixed(BODY body_in,
                                            ReduceParams reduce_params)
{
  LaunchContext ctx;

  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(body_in);
  auto& body      = privatizer.get_priv();

  // Set pointer to shared memory
  extern __shared__ char raja_shmem_ptr[];
  ctx.shared_mem_ptr = raja_shmem_ptr;

  RAJA::expt::invoke_body(reduce_params, body, ctx);

  // Using a flatten global policy as we may use all dimensions
  RAJA::expt::ParamMultiplexer::parampack_combine(
      RAJA::hip_flatten_global_xyz_direct {}, reduce_params);
}

template<bool async, int nthreads>
struct LaunchExecute<RAJA::policy::hip::hip_launch_t<async, nthreads>>
{

  template<size_t ThreadDIM = 3, typename BODY_IN, typename ReduceParams>
  static concepts::enable_if_t<
      resources::EventProxy<resources::Resource>,
      RAJA::expt::type_traits::is_ForallParamPack<ReduceParams>,
      RAJA::expt::type_traits::is_ForallParamPack_empty<ReduceParams>>
  exec(RAJA::resources::Resource res,
       const LaunchParams<ThreadDIM>& params,
       BODY_IN&& body_in,
       ReduceParams& RAJA_UNUSED_ARG(launch_reducers))
  {
    using BODY = camp::decay<BODY_IN>;

    auto func =
        reinterpret_cast<const void*>(&launch_global_fcn_fixed<BODY, nthreads>);

    resources::Hip hip_res = res.get<RAJA::resources::Hip>();

    //
    // Compute the number of blocks and threads
    //

    hip_dim_t gridSize {static_cast<hip_dim_member_t>(params.teams.value[0]),
                        static_cast<hip_dim_member_t>(params.teams.value[1]),
                        static_cast<hip_dim_member_t>(params.teams.value[2])};

    hip_dim_t blockSize {
        static_cast<hip_dim_member_t>(params.threads.value[0]),
        static_cast<hip_dim_member_t>(params.threads.value[1]),
        static_cast<hip_dim_member_t>(params.threads.value[2])};

    // Only launch kernel if we have something to iterate over
    constexpr hip_dim_member_t zero = 0;
    if (gridSize.x > zero && gridSize.y > zero && gridSize.z > zero &&
        blockSize.x > zero && blockSize.y > zero && blockSize.z > zero)
    {

      RAJA_FT_BEGIN;

      {
        size_t shared_mem_size = params.shared_mem_size;

        //
        // Privatize the loop_body, using make_launch_body to setup reductions
        //
        BODY body = RAJA::hip::make_launch_body(func, gridSize, blockSize,
                                                shared_mem_size, hip_res,
                                                std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void* args[] = {(void*)&body};
        RAJA::hip::launch(func, gridSize, blockSize, args, shared_mem_size,
                          hip_res, async);
      }

      RAJA_FT_END;
    }

    return resources::EventProxy<resources::Resource>(res);
  }

  // Version with explicit reduction parameters..
  template<size_t ThreadDIM = 3, typename BODY_IN, typename ReduceParams>
  static concepts::enable_if_t<
      resources::EventProxy<resources::Resource>,
      RAJA::expt::type_traits::is_ForallParamPack<ReduceParams>,
      concepts::negate<
          RAJA::expt::type_traits::is_ForallParamPack_empty<ReduceParams>>>
  exec(RAJA::resources::Resource res,
       const LaunchParams<ThreadDIM>& launch_params,
       BODY_IN&& body_in,
       ReduceParams& launch_reducers)
  {
    using BODY = camp::decay<BODY_IN>;
    // Use a generic block size policy here to match that used in
    // parampack_combine
    using EXEC_POL =
        RAJA::policy::hip::hip_launch_t<async, named_usage::unspecified>;
    EXEC_POL pol {};

    auto func = reinterpret_cast<const void*>(
        &launch_new_reduce_global_fcn_fixed<BODY, nthreads,
                                            camp::decay<ReduceParams>>);

    resources::Hip hip_res = res.get<RAJA::resources::Hip>();

    //
    // Compute the number of blocks and threads
    //

    hip_dim_t gridSize {
        static_cast<hip_dim_member_t>(launch_params.teams.value[0]),
        static_cast<hip_dim_member_t>(launch_params.teams.value[1]),
        static_cast<hip_dim_member_t>(launch_params.teams.value[2])};

    hip_dim_t blockSize {
        static_cast<hip_dim_member_t>(launch_params.threads.value[0]),
        static_cast<hip_dim_member_t>(launch_params.threads.value[1]),
        static_cast<hip_dim_member_t>(launch_params.threads.value[2])};

    // Only launch kernel if we have something to iterate over
    constexpr hip_dim_member_t zero = 0;
    if (gridSize.x > zero && gridSize.y > zero && gridSize.z > zero &&
        blockSize.x > zero && blockSize.y > zero && blockSize.z > zero)
    {

      RAJA_FT_BEGIN;

      size_t shared_mem_size = launch_params.shared_mem_size;
      RAJA::hip::detail::hipInfo launch_info;
      launch_info.gridDim      = gridSize;
      launch_info.blockDim     = blockSize;
      launch_info.dynamic_smem = &shared_mem_size;
      launch_info.res          = hip_res;

      {

        RAJA::expt::ParamMultiplexer::parampack_init(pol, launch_reducers,
                                                     launch_info);

        //
        // Privatize the loop_body, using make_launch_body to setup reductions
        //
        BODY body = RAJA::hip::make_launch_body(func, gridSize, blockSize,
                                                shared_mem_size, hip_res,
                                                std::forward<BODY_IN>(body_in));

        //
        // Launch the kernel
        //
        void* args[] = {(void*)&body, (void*)&launch_reducers};
        RAJA::hip::launch(func, gridSize, blockSize, args, shared_mem_size,
                          hip_res, async);

        RAJA::expt::ParamMultiplexer::parampack_resolve(pol, launch_reducers,
                                                        launch_info);
      }

      RAJA_FT_END;
    }

    return resources::EventProxy<resources::Resource>(res);
  }
};


/*
  Arbitrary dimension thread indexing
*/
template<size_t dim>
struct hip_loop_dim_exec;

template<typename SEGMENT, size_t DIM>
struct LoopExecute<hip_loop_dim_exec<DIM>, SEGMENT>
{

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const &ctx,
      SEGMENT const& segment,
      BODY const& body)
  {
    const int i = ctx.thread_id[DIM];

    body(*(segment.begin() + i));
  }
};


/*
   HIP generic loop implementations
*/
template<typename SEGMENT, typename IndexMapper>
struct LoopExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t i = IndexMapper::template index<diff_t>();

    body(*(segment.begin() + i));
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct LoopExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const diff_t i0 = IndexMapper0::template index<diff_t>();
    const diff_t i1 = IndexMapper1::template index<diff_t>();

    body(*(segment0.begin() + i0), *(segment1.begin() + i1));
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct LoopExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1,
                                   IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const diff_t i0 = IndexMapper0::template index<diff_t>();
    const diff_t i1 = IndexMapper1::template index<diff_t>();
    const diff_t i2 = IndexMapper2::template index<diff_t>();

    body(*(segment0.begin() + i0), *(segment1.begin() + i1),
         *(segment2.begin() + i2));
  }
};

template<typename SEGMENT, typename IndexMapper>
struct LoopExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t len = segment.end() - segment.begin();
    const diff_t i   = IndexMapper::template index<diff_t>();

    if (i < len)
    {
      body(*(segment.begin() + i));
    }
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct LoopExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();

    const diff_t i0 = IndexMapper0::template index<diff_t>();
    const diff_t i1 = IndexMapper1::template index<diff_t>();

    if (i0 < len0 && i1 < len1)
    {
      body(*(segment0.begin() + i0), *(segment1.begin() + i1));
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct LoopExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1,
                                   IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len2 = segment2.end() - segment2.begin();

    const diff_t i0 = IndexMapper0::template index<diff_t>();
    const diff_t i1 = IndexMapper1::template index<diff_t>();
    const diff_t i2 = IndexMapper2::template index<diff_t>();

    if (i0 < len0 && i1 < len1 && i2 < len2)
    {
      body(*(segment0.begin() + i0), *(segment1.begin() + i1),
           *(segment2.begin() + i2));
    }
  }
};

template<typename SEGMENT, typename IndexMapper>
struct LoopExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t len      = segment.end() - segment.begin();
    const diff_t i_init   = IndexMapper::template index<diff_t>();
    const diff_t i_stride = IndexMapper::template size<diff_t>();

    for (diff_t i = i_init; i < len; i += i_stride)
    {
      body(*(segment.begin() + i));
    }
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct LoopExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();

    const diff_t i0_init = IndexMapper0::template index<diff_t>();
    const diff_t i1_init = IndexMapper1::template index<diff_t>();

    const diff_t i0_stride = IndexMapper0::template size<diff_t>();
    const diff_t i1_stride = IndexMapper1::template size<diff_t>();

    for (diff_t i0 = i0_init; i0 < len0; i0 += i0_stride)
    {

      for (diff_t i1 = i1_init; i1 < len1; i1 += i1_stride)
      {

        body(*(segment0.begin() + i0), *(segment1.begin() + i1));
      }
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct LoopExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1,
        IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len2 = segment2.end() - segment2.begin();

    const diff_t i0_init = IndexMapper0::template index<diff_t>();
    const diff_t i1_init = IndexMapper1::template index<diff_t>();
    const diff_t i2_init = IndexMapper2::template index<diff_t>();

    const diff_t i0_stride = IndexMapper0::template size<diff_t>();
    const diff_t i1_stride = IndexMapper1::template size<diff_t>();
    const diff_t i2_stride = IndexMapper2::template size<diff_t>();

    for (diff_t i0 = i0_init; i0 < len0; i0 += i0_stride)
    {

      for (diff_t i1 = i1_init; i1 < len1; i1 += i1_stride)
      {

        for (diff_t i2 = i2_init; i2 < len2; i2 += i2_stride)
        {

          body(*(segment0.begin() + i0), *(segment1.begin() + i1),
               *(segment2.begin() + i2));
        }
      }
    }
  }
};

/*
   HIP generic loop_icount implementations
*/
template<typename SEGMENT, typename IndexMapper>
struct LoopICountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t i = IndexMapper::template index<diff_t>();

    body(*(segment.begin() + i), i);
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct LoopICountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const diff_t i0 = IndexMapper0::template index<diff_t>();
    const diff_t i1 = IndexMapper1::template index<diff_t>();

    body(*(segment0.begin() + i0), *(segment1.begin() + i1), i0, i1);
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct LoopICountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1,
                                   IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const diff_t i0 = IndexMapper0::template index<diff_t>();
    const diff_t i1 = IndexMapper1::template index<diff_t>();
    const diff_t i2 = IndexMapper2::template index<diff_t>();

    body(*(segment0.begin() + i0), *(segment1.begin() + i1),
         *(segment2.begin() + i2), i0, i1, i2);
  }
};

template<typename SEGMENT, typename IndexMapper>
struct LoopICountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t len = segment.end() - segment.begin();
    const diff_t i   = IndexMapper::template index<diff_t>();

    if (i < len)
    {
      body(*(segment.begin() + i), i);
    }
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct LoopICountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();

    const diff_t i0 = IndexMapper0::template index<diff_t>();
    const diff_t i1 = IndexMapper1::template index<diff_t>();

    if (i0 < len0 && i1 < len1)
    {
      body(*(segment0.begin() + i0), *(segment1.begin() + i1), i0, i1);
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct LoopICountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1,
                                   IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len2 = segment2.end() - segment2.begin();

    const diff_t i0 = IndexMapper0::template index<diff_t>();
    const diff_t i1 = IndexMapper1::template index<diff_t>();
    const diff_t i2 = IndexMapper2::template index<diff_t>();

    if (i0 < len0 && i1 < len1 && i2 < len2)
    {
      body(*(segment0.begin() + i0), *(segment1.begin() + i1),
           *(segment2.begin() + i2), i0, i1, i2);
    }
  }
};

template<typename SEGMENT, typename IndexMapper>
struct LoopICountExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t len      = segment.end() - segment.begin();
    const diff_t i_init   = IndexMapper::template index<diff_t>();
    const diff_t i_stride = IndexMapper::template size<diff_t>();

    for (diff_t i = i_init; i < len; i += i_stride)
    {
      body(*(segment.begin() + i), i);
    }
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct LoopICountExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();

    const diff_t i0_init = IndexMapper0::template index<diff_t>();
    const diff_t i1_init = IndexMapper1::template index<diff_t>();

    const diff_t i0_stride = IndexMapper0::template size<diff_t>();
    const diff_t i1_stride = IndexMapper1::template size<diff_t>();

    for (diff_t i0 = i0_init; i0 < len0; i0 += i0_stride)
    {

      for (diff_t i1 = i1_init; i1 < len1; i1 += i1_stride)
      {

        body(*(segment0.begin() + i0), *(segment1.begin() + i1), i0, i1);
      }
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct LoopICountExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1,
        IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const int len0 = segment0.end() - segment0.begin();
    const int len1 = segment1.end() - segment1.begin();
    const int len2 = segment2.end() - segment2.begin();

    const diff_t i0_init = IndexMapper0::template index<diff_t>();
    const diff_t i1_init = IndexMapper1::template index<diff_t>();
    const diff_t i2_init = IndexMapper2::template index<diff_t>();

    const diff_t i0_stride = IndexMapper0::template size<diff_t>();
    const diff_t i1_stride = IndexMapper1::template size<diff_t>();
    const diff_t i2_stride = IndexMapper2::template size<diff_t>();

    for (diff_t i0 = i0_init; i0 < len0; i0 += i0_stride)
    {

      for (diff_t i1 = i1_init; i1 < len1; i1 += i1_stride)
      {

        for (diff_t i2 = i2_init; i2 < len2; i2 += i2_stride)
        {

          body(*(segment0.begin() + i0), *(segment1.begin() + i1),
               *(segment2.begin() + i2), i0, i1, i2);
        }
      }
    }
  }
};

/*
   HIP generic flattened loop implementations
*/
template<typename SEGMENT, kernel_sync_requirement sync, typename IndexMapper0>
struct LoopExecute<RAJA::policy::hip::hip_flatten_indexer<
                       RAJA::iteration_mapping::DirectUnchecked,
                       sync,
                       IndexMapper0>,
                   SEGMENT>
    : LoopExecute<RAJA::policy::hip::hip_indexer<
                      RAJA::iteration_mapping::DirectUnchecked,
                      sync,
                      IndexMapper0>,
                  SEGMENT>
{};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct LoopExecute<RAJA::policy::hip::hip_flatten_indexer<
                       RAJA::iteration_mapping::DirectUnchecked,
                       kernel_sync_requirement::none,
                       IndexMapper0,
                       IndexMapper1>,
                   SEGMENT>
{
  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const int i0 = IndexMapper0::template index<diff_t>();
    const int i1 = IndexMapper1::template index<diff_t>();

    const diff_t i0_stride = IndexMapper0::template size<diff_t>();

    const int i = i0 + i0_stride * i1;

    body(*(segment.begin() + i));
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct LoopExecute<RAJA::policy::hip::hip_flatten_indexer<
                       RAJA::iteration_mapping::DirectUnchecked,
                       kernel_sync_requirement::none,
                       IndexMapper0,
                       IndexMapper1,
                       IndexMapper2>,
                   SEGMENT>
{
  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const int i0 = IndexMapper0::template index<diff_t>();
    const int i1 = IndexMapper1::template index<diff_t>();
    const int i2 = IndexMapper2::template index<diff_t>();

    const diff_t i0_stride = IndexMapper0::template size<diff_t>();
    const diff_t i1_stride = IndexMapper1::template size<diff_t>();

    const int i = i0 + i0_stride * (i1 + i1_stride * i2);

    body(*(segment.begin() + i));
  }
};

template<typename SEGMENT, kernel_sync_requirement sync, typename IndexMapper0>
struct LoopExecute<
    RAJA::policy::hip::hip_flatten_indexer<RAJA::iteration_mapping::Direct,
                                           sync,
                                           IndexMapper0>,
    SEGMENT>
    : LoopExecute<
          RAJA::policy::hip::
              hip_indexer<RAJA::iteration_mapping::Direct, sync, IndexMapper0>,
          SEGMENT>
{};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct LoopExecute<
    RAJA::policy::hip::hip_flatten_indexer<RAJA::iteration_mapping::Direct,
                                           kernel_sync_requirement::none,
                                           IndexMapper0,
                                           IndexMapper1>,
    SEGMENT>
{
  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const int len = segment.end() - segment.begin();

    const int i0 = IndexMapper0::template index<diff_t>();
    const int i1 = IndexMapper1::template index<diff_t>();

    const diff_t i0_stride = IndexMapper0::template size<diff_t>();

    const int i = i0 + i0_stride * i1;

    if (i < len)
    {
      body(*(segment.begin() + i));
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct LoopExecute<
    RAJA::policy::hip::hip_flatten_indexer<RAJA::iteration_mapping::Direct,
                                           kernel_sync_requirement::none,
                                           IndexMapper0,
                                           IndexMapper1,
                                           IndexMapper2>,
    SEGMENT>
{
  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const int len = segment.end() - segment.begin();

    const int i0 = IndexMapper0::template index<diff_t>();
    const int i1 = IndexMapper1::template index<diff_t>();
    const int i2 = IndexMapper2::template index<diff_t>();

    const diff_t i0_stride = IndexMapper0::template size<diff_t>();
    const diff_t i1_stride = IndexMapper1::template size<diff_t>();

    const int i = i0 + i0_stride * (i1 + i1_stride * i2);

    if (i < len)
    {
      body(*(segment.begin() + i));
    }
  }
};

template<typename SEGMENT, kernel_sync_requirement sync, typename IndexMapper0>
struct LoopExecute<
    RAJA::policy::hip::hip_flatten_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        sync,
        IndexMapper0>,
    SEGMENT>
    : LoopExecute<
          RAJA::policy::hip::hip_indexer<
              RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
              sync,
              IndexMapper0>,
          SEGMENT>
{};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct LoopExecute<
    RAJA::policy::hip::hip_flatten_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1>,
    SEGMENT>
{
  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const int len = segment.end() - segment.begin();

    const int i0 = IndexMapper0::template index<diff_t>();
    const int i1 = IndexMapper1::template index<diff_t>();

    const int i0_stride = IndexMapper0::template size<diff_t>();
    const int i1_stride = IndexMapper1::template size<diff_t>();

    for (int i = i0 + i0_stride * i1; i < len; i += i0_stride * i1_stride)
    {
      body(*(segment.begin() + i));
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct LoopExecute<
    RAJA::policy::hip::hip_flatten_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1,
        IndexMapper2>,
    SEGMENT>
{
  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      SEGMENT const& segment,
      BODY const& body)
  {
    const int len = segment.end() - segment.begin();

    const int i0 = IndexMapper0::template index<diff_t>();
    const int i1 = IndexMapper1::template index<diff_t>();
    const int i2 = IndexMapper2::template index<diff_t>();

    const int i0_stride = IndexMapper0::template size<diff_t>();
    const int i1_stride = IndexMapper1::template size<diff_t>();
    const int i2_stride = IndexMapper2::template size<diff_t>();

    for (int i = i0 + i0_stride * (i1 + i1_stride * i2); i < len;
         i += i0_stride * i1_stride * i2_stride)
    {
      body(*(segment.begin() + i));
    }
  }
};

/*
   HIP generic tile implementations
*/
template<typename SEGMENT, typename IndexMapper>
struct TileExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t i =
        IndexMapper::template index<diff_t>() * static_cast<diff_t>(tile_size);

    body(segment.slice(i, static_cast<diff_t>(tile_size)));
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct TileExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const diff_t i0 = IndexMapper0::template index<diff_t>() *
                      static_cast<diff_t>(tile_size0);
    const diff_t i1 = IndexMapper1::template index<diff_t>() *
                      static_cast<diff_t>(tile_size1);

    body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
         segment1.slice(i1, static_cast<diff_t>(tile_size1)));
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct TileExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1,
                                   IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      TILE_T tile_size2,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const diff_t i0 = IndexMapper0::template index<diff_t>() *
                      static_cast<diff_t>(tile_size0);
    const diff_t i1 = IndexMapper1::template index<diff_t>() *
                      static_cast<diff_t>(tile_size1);
    const diff_t i2 = IndexMapper2::template index<diff_t>() *
                      static_cast<diff_t>(tile_size2);

    body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
         segment1.slice(i1, static_cast<diff_t>(tile_size1)),
         segment2.slice(i2, static_cast<diff_t>(tile_size2)));
  }
};

template<typename SEGMENT, typename IndexMapper>
struct TileExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t len = segment.end() - segment.begin();
    const diff_t i =
        IndexMapper::template index<diff_t>() * static_cast<diff_t>(tile_size);

    if (i < len)
    {
      body(segment.slice(i, static_cast<diff_t>(tile_size)));
    }
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct TileExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const diff_t len0 = segment0.end() - segment0.begin();
    const diff_t len1 = segment1.end() - segment1.begin();

    const diff_t i0 = IndexMapper0::template index<diff_t>() *
                      static_cast<diff_t>(tile_size0);
    const diff_t i1 = IndexMapper1::template index<diff_t>() *
                      static_cast<diff_t>(tile_size1);

    if (i0 < len0 && i1 < len1)
    {
      body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
           segment1.slice(i1, static_cast<diff_t>(tile_size1)));
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct TileExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1,
                                   IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      TILE_T tile_size2,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const diff_t len0 = segment0.end() - segment0.begin();
    const diff_t len1 = segment1.end() - segment1.begin();
    const diff_t len2 = segment2.end() - segment2.begin();

    const diff_t i0 = IndexMapper0::template index<diff_t>() *
                      static_cast<diff_t>(tile_size0);
    const diff_t i1 = IndexMapper1::template index<diff_t>() *
                      static_cast<diff_t>(tile_size1);
    const diff_t i2 = IndexMapper2::template index<diff_t>() *
                      static_cast<diff_t>(tile_size2);

    if (i0 < len0 && i1 < len1 && i2 < len2)
    {
      body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
           segment1.slice(i1, static_cast<diff_t>(tile_size1)),
           segment2.slice(i2, static_cast<diff_t>(tile_size2)));
    }
  }
};

template<typename SEGMENT, typename IndexMapper>
struct TileExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t len = segment.end() - segment.begin();
    const diff_t i_init =
        IndexMapper::template index<diff_t>() * static_cast<diff_t>(tile_size);
    const diff_t i_stride =
        IndexMapper::template size<diff_t>() * static_cast<diff_t>(tile_size);

    for (diff_t i = i_init; i < len; i += i_stride)
    {
      body(segment.slice(i, static_cast<diff_t>(tile_size)));
    }
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct TileExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const diff_t len0 = segment0.end() - segment0.begin();
    const diff_t len1 = segment1.end() - segment1.begin();

    const diff_t i0_init = IndexMapper0::template index<diff_t>() *
                           static_cast<diff_t>(tile_size0);
    const diff_t i1_init = IndexMapper1::template index<diff_t>() *
                           static_cast<diff_t>(tile_size1);

    const diff_t i0_stride =
        IndexMapper0::template size<diff_t>() * static_cast<diff_t>(tile_size0);
    const diff_t i1_stride =
        IndexMapper1::template size<diff_t>() * static_cast<diff_t>(tile_size1);

    for (diff_t i0 = i0_init; i0 < len0; i0 += i0_stride)
    {
      for (diff_t i1 = i1_init; i1 < len1; i1 += i1_stride)
      {
        body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
             segment1.slice(i1, static_cast<diff_t>(tile_size1)));
      }
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct TileExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1,
        IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      TILE_T tile_size2,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const diff_t len0 = segment0.end() - segment0.begin();
    const diff_t len1 = segment1.end() - segment1.begin();
    const diff_t len2 = segment2.end() - segment2.begin();

    const diff_t i0_init = IndexMapper0::template index<diff_t>() *
                           static_cast<diff_t>(tile_size0);
    const diff_t i1_init = IndexMapper1::template index<diff_t>() *
                           static_cast<diff_t>(tile_size1);
    const diff_t i2_init = IndexMapper2::template index<diff_t>() *
                           static_cast<diff_t>(tile_size2);

    const diff_t i0_stride =
        IndexMapper0::template size<diff_t>() * static_cast<diff_t>(tile_size0);
    const diff_t i1_stride =
        IndexMapper1::template size<diff_t>() * static_cast<diff_t>(tile_size1);
    const diff_t i2_stride =
        IndexMapper2::template size<diff_t>() * static_cast<diff_t>(tile_size2);

    for (diff_t i0 = i0_init; i0 < len0; i0 += i0_stride)
    {
      for (diff_t i1 = i1_init; i1 < len1; i1 += i1_stride)
      {
        for (diff_t i2 = i2_init; i2 < len2; i2 += i2_stride)
        {
          body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
               segment1.slice(i1, static_cast<diff_t>(tile_size1)),
               segment2.slice(i2, static_cast<diff_t>(tile_size2)));
        }
      }
    }
  }
};

/*
   HIP generic tile_tcount implementations
*/
template<typename SEGMENT, typename IndexMapper>
struct TileTCountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t t = IndexMapper::template index<diff_t>();
    const diff_t i = t * static_cast<diff_t>(tile_size);

    body(segment.slice(i, static_cast<diff_t>(tile_size)), t);
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct TileTCountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const diff_t t0 = IndexMapper0::template index<diff_t>();
    const diff_t t1 = IndexMapper1::template index<diff_t>();

    const diff_t i0 = t0 * static_cast<diff_t>(tile_size0);
    const diff_t i1 = t1 * static_cast<diff_t>(tile_size1);

    body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
         segment1.slice(i1, static_cast<diff_t>(tile_size1)), t0, t1);
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct TileTCountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::DirectUnchecked,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1,
                                   IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      TILE_T tile_size2,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const diff_t t0 = IndexMapper0::template index<diff_t>();
    const diff_t t1 = IndexMapper1::template index<diff_t>();
    const diff_t t2 = IndexMapper2::template index<diff_t>();

    const diff_t i0 = t0 * static_cast<diff_t>(tile_size0);
    const diff_t i1 = t1 * static_cast<diff_t>(tile_size1);
    const diff_t i2 = t2 * static_cast<diff_t>(tile_size2);

    body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
         segment1.slice(i1, static_cast<diff_t>(tile_size1)),
         segment2.slice(i2, static_cast<diff_t>(tile_size2)), t0, t1, t2);
  }
};

template<typename SEGMENT, typename IndexMapper>
struct TileTCountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t len = segment.end() - segment.begin();
    const diff_t t   = IndexMapper::template index<diff_t>();
    const diff_t i   = t * static_cast<diff_t>(tile_size);

    if (i < len)
    {
      body(segment.slice(i, static_cast<diff_t>(tile_size)), t);
    }
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct TileTCountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const diff_t len0 = segment0.end() - segment0.begin();
    const diff_t len1 = segment1.end() - segment1.begin();

    const diff_t t0 = IndexMapper0::template index<diff_t>();
    const diff_t t1 = IndexMapper1::template index<diff_t>();

    const diff_t i0 = t0 * static_cast<diff_t>(tile_size0);
    const diff_t i1 = t1 * static_cast<diff_t>(tile_size1);

    if (i0 < len0 && i1 < len1)
    {
      body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
           segment1.slice(i1, static_cast<diff_t>(tile_size1)), t0, t1);
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct TileTCountExecute<
    RAJA::policy::hip::hip_indexer<RAJA::iteration_mapping::Direct,
                                   kernel_sync_requirement::none,
                                   IndexMapper0,
                                   IndexMapper1,
                                   IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      TILE_T tile_size2,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const diff_t len0 = segment0.end() - segment0.begin();
    const diff_t len1 = segment1.end() - segment1.begin();
    const diff_t len2 = segment2.end() - segment2.begin();

    const diff_t t0 = IndexMapper0::template index<diff_t>();
    const diff_t t1 = IndexMapper1::template index<diff_t>();
    const diff_t t2 = IndexMapper2::template index<diff_t>();

    const diff_t i0 = t0 * static_cast<diff_t>(tile_size0);
    const diff_t i1 = t1 * static_cast<diff_t>(tile_size1);
    const diff_t i2 = t2 * static_cast<diff_t>(tile_size2);

    if (i0 < len0 && i1 < len1 && i2 < len2)
    {
      body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
           segment1.slice(i1, static_cast<diff_t>(tile_size1)),
           segment2.slice(i2, static_cast<diff_t>(tile_size2)), t0, t1, t2);
    }
  }
};

template<typename SEGMENT, typename IndexMapper>
struct TileTCountExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size,
      SEGMENT const& segment,
      BODY const& body)
  {
    const diff_t len      = segment.end() - segment.begin();
    const diff_t t_init   = IndexMapper::template index<diff_t>();
    const diff_t i_init   = t_init * static_cast<diff_t>(tile_size);
    const diff_t t_stride = IndexMapper::template size<diff_t>();
    const diff_t i_stride = t_stride * static_cast<diff_t>(tile_size);

    for (diff_t i = i_init, t = t_init; i < len; i += i_stride, t += t_stride)
    {
      body(segment.slice(i, static_cast<diff_t>(tile_size)), t);
    }
  }
};

template<typename SEGMENT, typename IndexMapper0, typename IndexMapper1>
struct TileTCountExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      BODY const& body)
  {
    const diff_t len0 = segment0.end() - segment0.begin();
    const diff_t len1 = segment1.end() - segment1.begin();

    const diff_t t0_init = IndexMapper0::template index<diff_t>();
    const diff_t t1_init = IndexMapper1::template index<diff_t>();

    const diff_t i0_init = t0_init * static_cast<diff_t>(tile_size0);
    const diff_t i1_init = t1_init * static_cast<diff_t>(tile_size1);

    const diff_t t0_stride = IndexMapper0::template size<diff_t>();
    const diff_t t1_stride = IndexMapper1::template size<diff_t>();

    const diff_t i0_stride = t0_stride * static_cast<diff_t>(tile_size0);
    const diff_t i1_stride = t1_stride * static_cast<diff_t>(tile_size1);

    for (diff_t i0 = i0_init, t0 = t0_init; i0 < len0;
         i0 += i0_stride, t0 += t0_stride)
    {
      for (diff_t i1 = i1_init, t1 = t1_init; i1 < len1;
           i1 += i1_stride, t1 += t1_stride)
      {
        body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
             segment1.slice(i1, static_cast<diff_t>(tile_size1)), t0, t1);
      }
    }
  }
};

template<typename SEGMENT,
         typename IndexMapper0,
         typename IndexMapper1,
         typename IndexMapper2>
struct TileTCountExecute<
    RAJA::policy::hip::hip_indexer<
        RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
        kernel_sync_requirement::none,
        IndexMapper0,
        IndexMapper1,
        IndexMapper2>,
    SEGMENT>
{

  using diff_t = typename std::iterator_traits<
      typename SEGMENT::iterator>::difference_type;

  template<typename TILE_T, typename BODY>
  static RAJA_INLINE RAJA_DEVICE void exec(
      LaunchContext const RAJA_UNUSED_ARG(&ctx),
      TILE_T tile_size0,
      TILE_T tile_size1,
      TILE_T tile_size2,
      SEGMENT const& segment0,
      SEGMENT const& segment1,
      SEGMENT const& segment2,
      BODY const& body)
  {
    const diff_t len0 = segment0.end() - segment0.begin();
    const diff_t len1 = segment1.end() - segment1.begin();
    const diff_t len2 = segment2.end() - segment2.begin();

    const diff_t t0_init = IndexMapper0::template index<diff_t>();
    const diff_t t1_init = IndexMapper1::template index<diff_t>();
    const diff_t t2_init = IndexMapper2::template index<diff_t>();

    const diff_t i0_init = t0_init * static_cast<diff_t>(tile_size0);
    const diff_t i1_init = t1_init * static_cast<diff_t>(tile_size1);
    const diff_t i2_init = t2_init * static_cast<diff_t>(tile_size2);

    const diff_t t0_stride = IndexMapper0::template size<diff_t>();
    const diff_t t1_stride = IndexMapper1::template size<diff_t>();
    const diff_t t2_stride = IndexMapper2::template size<diff_t>();

    const diff_t i0_stride = t0_stride * static_cast<diff_t>(tile_size0);
    const diff_t i1_stride = t1_stride * static_cast<diff_t>(tile_size1);
    const diff_t i2_stride = t2_stride * static_cast<diff_t>(tile_size2);

    for (diff_t i0 = i0_init, t0 = t0_init; i0 < len0;
         i0 += i0_stride, t0 += t0_stride)
    {
      for (diff_t i1 = i1_init, t1 = t1_init; i1 < len1;
           i1 += i1_stride, t1 += t1_stride)
      {
        for (diff_t i2 = i2_init, t2 = t2_init; i2 < len2;
             i2 += i2_stride, t2 += t2_stride)
        {
          body(segment0.slice(i0, static_cast<diff_t>(tile_size0)),
               segment1.slice(i1, static_cast<diff_t>(tile_size1)),
               segment2.slice(i2, static_cast<diff_t>(tile_size2)), t0, t1, t2);
        }
      }
    }
  }
};

}  // namespace RAJA
#endif
