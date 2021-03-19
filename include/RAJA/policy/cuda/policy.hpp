/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA CUDA policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_cuda_HPP
#define RAJA_policy_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_CUDA_ACTIVE)

#include <utility>

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

#if defined(RAJA_ENABLE_CLANG_CUDA)
using cuda_dim_t = uint3;
#else
using cuda_dim_t = dim3;
#endif

using cuda_dim_member_t = camp::decay<decltype(std::declval<cuda_dim_t>().x)>;


//
/////////////////////////////////////////////////////////////////////
//
// Execution policies
//
/////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///

namespace detail
{
template <bool Async>
struct get_launch {
  static constexpr RAJA::Launch value = RAJA::Launch::async;
};

template <>
struct get_launch<false> {
  static constexpr RAJA::Launch value = RAJA::Launch::sync;
};
}  // end namespace detail

namespace policy
{
namespace cuda
{

template <size_t BLOCK_SIZE, bool Async = false>
struct cuda_exec : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::cuda,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::cuda> {
};



//
// NOTE: There is no Index set segment iteration policy for CUDA
//

///
/// WorkGroup execution policies
///
template <size_t BLOCK_SIZE, bool Async = false>
struct cuda_work : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::cuda,
                       RAJA::Pattern::workgroup_exec,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::cuda> {
};

struct unordered_cuda_loop_y_block_iter_x_threadblock_average
    : public RAJA::make_policy_pattern_platform_t<
                       RAJA::Policy::cuda,
                       RAJA::Pattern::workgroup_order,
                       RAJA::Platform::cuda> {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction reduction policies
///
///////////////////////////////////////////////////////////////////////
///

template <bool maybe_atomic>
struct cuda_reduce_base
    : public RAJA::
          make_policy_pattern_launch_platform_t<RAJA::Policy::cuda,
                                                RAJA::Pattern::reduce,
                                                detail::get_launch<false>::value,
                                                RAJA::Platform::cuda> {
};

using cuda_reduce = cuda_reduce_base<false>;

using cuda_reduce_atomic = cuda_reduce_base<true>;


// Policy for RAJA::statement::Reduce that reduces threads in a block
// down to threadIdx 0
struct cuda_block_reduce{};

// Policy for RAJA::statement::Reduce that reduces threads in a warp
// down to the first lane of the warp
struct cuda_warp_reduce{};

// Policy to map work directly to threads within a warp
// Maximum iteration count is WARP_SIZE
// Cannot be used in conjunction with cuda_thread_x_*
// Multiple warps have to be created by using cuda_thread_{yz}_*
struct cuda_warp_direct{};

// Policy to map work to threads within a warp using a warp-stride loop
// Cannot be used in conjunction with cuda_thread_x_*
// Multiple warps have to be created by using cuda_thread_{yz}_*
struct cuda_warp_loop{};



// Policy to map work to threads within a warp using a bit mask
// Cannot be used in conjunction with cuda_thread_x_*
// Multiple warps have to be created by using cuda_thread_{yz}_*
// Since we are masking specific threads, multiple nested
// cuda_warp_masked
// can be used to create complex thread interleaving patterns
template<typename Mask>
struct cuda_warp_masked_direct {};

// Policy to map work to threads within a warp using a bit mask
// Cannot be used in conjunction with cuda_thread_x_*
// Multiple warps have to be created by using cuda_thread_{yz}_*
// Since we are masking specific threads, multiple nested
// cuda_warp_masked
// can be used to create complex thread interleaving patterns
template<typename Mask>
struct cuda_warp_masked_loop {};


template<typename Mask>
struct cuda_thread_masked_direct {};

template<typename Mask>
struct cuda_thread_masked_loop {};



//
// Operations in the included files are parametrized using the following
// values for CUDA warp size and max block size.
//
constexpr const RAJA::Index_type WARP_SIZE = 32;
constexpr const RAJA::Index_type MAX_BLOCK_SIZE = 1024;
constexpr const RAJA::Index_type MAX_WARPS = MAX_BLOCK_SIZE / WARP_SIZE;
static_assert(WARP_SIZE >= MAX_WARPS,
              "RAJA Assumption Broken: WARP_SIZE < MAX_WARPS");
static_assert(MAX_BLOCK_SIZE % WARP_SIZE == 0,
              "RAJA Assumption Broken: MAX_BLOCK_SIZE not "
              "a multiple of WARP_SIZE");

struct cuda_synchronize : make_policy_pattern_launch_t<Policy::cuda,
                                                       Pattern::synchronize,
                                                       Launch::sync> {
};

}  // end namespace cuda
}  // end namespace policy

using policy::cuda::cuda_exec;

template <size_t BLOCK_SIZE>
using cuda_exec_async = policy::cuda::cuda_exec<BLOCK_SIZE, true>;

using policy::cuda::cuda_work;

template <size_t BLOCK_SIZE>
using cuda_work_async = policy::cuda::cuda_work<BLOCK_SIZE, true>;

using policy::cuda::unordered_cuda_loop_y_block_iter_x_threadblock_average;

using policy::cuda::cuda_reduce_base;
using policy::cuda::cuda_reduce;
using policy::cuda::cuda_reduce_atomic;

using policy::cuda::cuda_block_reduce;
using policy::cuda::cuda_warp_reduce;

using policy::cuda::cuda_warp_direct;
using policy::cuda::cuda_warp_loop;

using policy::cuda::cuda_warp_masked_direct;
using policy::cuda::cuda_warp_masked_loop;

using policy::cuda::cuda_thread_masked_direct;
using policy::cuda::cuda_thread_masked_loop;

using policy::cuda::cuda_synchronize;




/*!
 * Maps segment indices to CUDA threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * For example, a segment of size 2000 will not fit, and trigger a runtime
 * error.
 */
template<int ... dim>
struct cuda_thread_xyz_direct{};

using cuda_thread_x_direct = cuda_thread_xyz_direct<0>;
using cuda_thread_y_direct = cuda_thread_xyz_direct<1>;
using cuda_thread_z_direct = cuda_thread_xyz_direct<2>;


/*!
 * Maps segment indices to CUDA threads.
 * Uses block-stride looping to exceed the maximum number of physical threads
 */
template<int ... dim>
struct cuda_thread_xyz_loop{};

using cuda_thread_x_loop = cuda_thread_xyz_loop<0>;
using cuda_thread_y_loop = cuda_thread_xyz_loop<1>;
using cuda_thread_z_loop = cuda_thread_xyz_loop<2>;

/*!
 * Maps segment indices to CUDA blocks.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical blocks to fit all of the direct map requests.
 */
template<int ... dim>
struct cuda_block_xyz_direct{};

using cuda_block_x_direct = cuda_block_xyz_direct<0>;
using cuda_block_y_direct = cuda_block_xyz_direct<1>;
using cuda_block_z_direct = cuda_block_xyz_direct<2>;


/*!
 * Maps segment indices to CUDA blocks.
 * Uses grid-stride looping to exceed the maximum number of blocks
 */
template<int ... dim>
struct cuda_block_xyz_loop{};

using cuda_block_x_loop = cuda_block_xyz_loop<0>;
using cuda_block_y_loop = cuda_block_xyz_loop<1>;
using cuda_block_z_loop = cuda_block_xyz_loop<2>;




namespace internal{

template<int dim>
struct CudaDimHelper;

template<>
struct CudaDimHelper<0>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline
  static
  constexpr
  auto get(dim_t const &d) ->
    decltype(d.x)
  {
    return d.x;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline
  static
  void set(dim_t &d, int value)
  {
    d.x = value;
  }
};

template<>
struct CudaDimHelper<1>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline
  static
  constexpr
  auto get(dim_t const &d) ->
    decltype(d.x)
  {
    return d.y;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline
  static
  void set(dim_t &d, int value)
  {
    d.y = value;
  }
};

template<>
struct CudaDimHelper<2>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline
  static
  constexpr
  auto get(dim_t const &d) ->
    decltype(d.x)
  {
    return d.z;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline
  static
  void set(dim_t &d, int value)
  {
    d.z = value;
  }
};

template<int dim, typename dim_t>
RAJA_HOST_DEVICE
constexpr
auto get_cuda_dim(dim_t const &d) ->
  decltype(d.x)
{
  return CudaDimHelper<dim>::get(d);
}

template<int dim, typename dim_t>
RAJA_HOST_DEVICE
void set_cuda_dim(dim_t &d, int value)
{
  return CudaDimHelper<dim>::set(d, value);
}
} // namespace internal


}  // namespace RAJA

#endif  // RAJA_ENABLE_CUDA
#endif
