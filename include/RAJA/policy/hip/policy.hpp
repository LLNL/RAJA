/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA HIP policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_hip_HPP
#define RAJA_policy_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_HIP_ACTIVE)

#include <utility>
#include "hip/hip_runtime.h"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/loop/policy.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

using hip_dim_t = dim3;
using hip_dim_member_t = camp::decay<decltype(std::declval<hip_dim_t>().x)>;


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
namespace hip
{

template <size_t BLOCK_SIZE, bool Async = false>
struct hip_exec : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::hip,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::hip> {
};

template <bool Async, int num_threads = 0>
struct hip_launch_t : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::hip,
                       RAJA::Pattern::region,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::hip> {
};


//
// NOTE: There is no Index set segment iteration policy for HIP
//

///
/// WorkGroup execution policies
///
template <size_t BLOCK_SIZE, bool Async = false>
struct hip_work : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::hip,
                       RAJA::Pattern::workgroup_exec,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::hip> {
};

/// execute the enqueued loops in an unordered fashion by mapping loops to
/// blocks in the y direction and loop iterations to threads in the x direction
/// with the size of the x direction being the average of the iteration counts
/// of all the loops
struct unordered_hip_loop_y_block_iter_x_threadblock_average
    : public RAJA::make_policy_pattern_platform_t<
                       RAJA::Policy::hip,
                       RAJA::Pattern::workgroup_order,
                       RAJA::Platform::hip> {
};


///
///////////////////////////////////////////////////////////////////////
///
/// Reduction reduction policies
///
///////////////////////////////////////////////////////////////////////
///

template <bool maybe_atomic>
struct hip_reduce_base
    : public RAJA::
          make_policy_pattern_launch_platform_t<RAJA::Policy::hip,
                                                RAJA::Pattern::reduce,
                                                detail::get_launch<false>::value,
                                                RAJA::Platform::hip> {
};

using hip_reduce = hip_reduce_base<false>;

using hip_reduce_atomic = hip_reduce_base<true>;


// Policy for RAJA::statement::Reduce that reduces threads in a block
// down to threadIdx 0
struct hip_block_reduce{};

// Policy for RAJA::statement::Reduce that reduces threads in a warp
// down to the first lane of the warp
struct hip_warp_reduce{};

// Policy to map work directly to threads within a warp
// Maximum iteration count is WARP_SIZE
// Cannot be used in conjunction with hip_thread_x_*
// Multiple warps have to be created by using hip_thread_{yz}_*
struct hip_warp_direct{};

// Policy to map work to threads within a warp using a warp-stride loop
// Cannot be used in conjunction with hip_thread_x_*
// Multiple warps have to be created by using hip_thread_{yz}_*
struct hip_warp_loop{};



// Policy to map work to threads within a warp using a bit mask
// Cannot be used in conjunction with hip_thread_x_*
// Multiple warps have to be created by using hip_thread_{yz}_*
// Since we are masking specific threads, multiple nested
// hip_warp_masked
// can be used to create complex thread interleaving patterns
template<typename Mask>
struct hip_warp_masked_direct {};

// Policy to map work to threads within a warp using a bit mask
// Cannot be used in conjunction with hip_thread_x_*
// Multiple warps have to be created by using hip_thread_{yz}_*
// Since we are masking specific threads, multiple nested
// hip_warp_masked
// can be used to create complex thread interleaving patterns
template<typename Mask>
struct hip_warp_masked_loop {};


template<typename Mask>
struct hip_thread_masked_direct {};

template<typename Mask>
struct hip_thread_masked_loop {};




//
// Operations in the included files are parametrized using the following
// values for HIP warp size and max block size.
//
#if defined(__HIP_PLATFORM_HCC__)
constexpr const RAJA::Index_type WARP_SIZE = 64;
#elif defined(__HIP_PLATFORM_NVCC__)
constexpr const RAJA::Index_type WARP_SIZE = 32;
#endif

constexpr const RAJA::Index_type MAX_BLOCK_SIZE = 1024;
constexpr const RAJA::Index_type MAX_WARPS = MAX_BLOCK_SIZE / WARP_SIZE;
static_assert(WARP_SIZE >= MAX_WARPS,
              "RAJA Assumption Broken: WARP_SIZE < MAX_WARPS");
static_assert(MAX_BLOCK_SIZE % WARP_SIZE == 0,
              "RAJA Assumption Broken: MAX_BLOCK_SIZE not "
              "a multiple of WARP_SIZE");

struct hip_synchronize : make_policy_pattern_launch_t<Policy::hip,
                                                       Pattern::synchronize,
                                                       Launch::sync> {
};

/*!
 * Hip atomic policy for using hip atomics on the device and
 * the provided host_policy on the host
 */
template<typename host_policy>
struct hip_atomic_explicit{};

/*!
 * Default hip atomic policy uses hip atomics on the device and non-atomics
 * on the host
 */
using hip_atomic = hip_atomic_explicit<loop_atomic>;

}  // end namespace hip
}  // end namespace policy

using policy::hip::hip_exec;

template <size_t BLOCK_SIZE>
using hip_exec_async = policy::hip::hip_exec<BLOCK_SIZE, true>;

using policy::hip::hip_work;

template <size_t BLOCK_SIZE>
using hip_work_async = policy::hip::hip_work<BLOCK_SIZE, true>;

using policy::hip::hip_atomic;
using policy::hip::hip_atomic_explicit;

using policy::hip::unordered_hip_loop_y_block_iter_x_threadblock_average;

using policy::hip::hip_reduce_base;
using policy::hip::hip_reduce;
using policy::hip::hip_reduce_atomic;

using policy::hip::hip_block_reduce;
using policy::hip::hip_warp_reduce;

using policy::hip::hip_warp_direct;
using policy::hip::hip_warp_loop;

using policy::hip::hip_warp_masked_direct;
using policy::hip::hip_warp_masked_loop;

using policy::hip::hip_thread_masked_direct;
using policy::hip::hip_thread_masked_loop;

using policy::hip::hip_synchronize;

using policy::hip::hip_launch_t;

/*!
 * Maps segment indices to HIP threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * For example, a segment of size 2000 will not fit, and trigger a runtime
 * error.
 */
template<int ... dim>
struct hip_thread_xyz_direct{};

using hip_thread_x_direct = hip_thread_xyz_direct<0>;
using hip_thread_y_direct = hip_thread_xyz_direct<1>;
using hip_thread_z_direct = hip_thread_xyz_direct<2>;


/*!
 * Maps segment indices to HIP threads.
 * Uses block-stride looping to exceed the maximum number of physical threads
 */
template<int ... dim>
struct hip_thread_xyz_loop{};

using hip_thread_x_loop = hip_thread_xyz_loop<0>;
using hip_thread_y_loop = hip_thread_xyz_loop<1>;
using hip_thread_z_loop = hip_thread_xyz_loop<2>;


/*!
 * Maps segment indices to HIP blocks.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical blocks to fit all of the direct map requests.
 */
template<int ... dim>
struct hip_block_xyz_direct{};

using hip_block_x_direct = hip_block_xyz_direct<0>;
using hip_block_y_direct = hip_block_xyz_direct<1>;
using hip_block_z_direct = hip_block_xyz_direct<2>;


/*!
 * Maps segment indices to HIP blocks.
 * Uses grid-stride looping to exceed the maximum number of blocks
 */
template<int ... dim>
struct hip_block_xyz_loop{};

using hip_block_x_loop = hip_block_xyz_loop<0>;
using hip_block_y_loop = hip_block_xyz_loop<1>;
using hip_block_z_loop = hip_block_xyz_loop<2>;




namespace internal{

template<int dim>
struct HipDimHelper;

template<>
struct HipDimHelper<0>{

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
struct HipDimHelper<1>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline
  static
  constexpr
  auto get(dim_t const &d) ->
    decltype(d.y)
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
struct HipDimHelper<2>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline
  static
  constexpr
  auto get(dim_t const &d) ->
    decltype(d.z)
  {
    return d.z;
  }

  template<typename dim_t>
  inline
  static
  RAJA_HOST_DEVICE
  void set(dim_t &d, int value)
  {
    d.z = value;
  }
};

template<int dim, typename dim_t>
RAJA_HOST_DEVICE
constexpr
auto get_hip_dim(dim_t const &d) ->
  decltype(HipDimHelper<dim>::get(d))
{
  return HipDimHelper<dim>::get(d);
}

template<int dim, typename dim_t>
RAJA_HOST_DEVICE
void set_hip_dim(dim_t &d, int value)
{
  return HipDimHelper<dim>::set(d, value);
}
} // namespace internal


}  // namespace RAJA

#endif  // RAJA_ENABLE_HIP
#endif
