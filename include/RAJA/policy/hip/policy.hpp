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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
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

template <typename _Indexer, bool Async = false>
struct hip_exec : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::hip,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::hip>
                , _Indexer {
  using Indexer = _Indexer;
  using Indexer::Indexer;
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


namespace internal{

RAJA_INLINE
int get_size(hip_dim_t dims)
{
  if(dims.x == 0 && dims.y == 0 && dims.z == 0){
    return 0;
  }
  return (dims.x ? dims.x : 1) *
         (dims.y ? dims.y : 1) *
         (dims.z ? dims.z : 1);
}

struct HipDims {

  hip_dim_t blocks{0,0,0};
  hip_dim_t threads{0,0,0};

  HipDims() = default;
  HipDims(HipDims const&) = default;
  HipDims& operator=(HipDims const&) = default;

  RAJA_INLINE
  HipDims(hip_dim_member_t default_val)
    : blocks{default_val, default_val, default_val}
    , threads{default_val, default_val, default_val}
  { }

  RAJA_INLINE
  int num_blocks() const {
    return get_size(blocks);
  }

  RAJA_INLINE
  int num_threads() const {
    return get_size(threads);
  }

  RAJA_INLINE
  hip_dim_t get_blocks() const {
    if (num_blocks() != 0) {
      return {(blocks.x ? blocks.x : 1),
              (blocks.y ? blocks.y : 1),
              (blocks.z ? blocks.z : 1)};
    } else {
      return blocks;
    }
  }

  RAJA_INLINE
  hip_dim_t get_threads() const {
    if (num_threads() != 0) {
      return {(threads.x ? threads.x : 1),
              (threads.y ? threads.y : 1),
              (threads.z ? threads.z : 1)};
    } else {
      return threads;
    }
  }
};

template<int dim>
struct HipDimHelper;

template<>
struct HipDimHelper<0>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static constexpr
  auto get(dim_t const &d)
  {
    return d.x;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static
  void set(dim_t &d, hip_dim_member_t value)
  {
    d.x = value;
  }
};

template<>
struct HipDimHelper<1>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static constexpr
  auto get(dim_t const &d)
  {
    return d.y;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static
  void set(dim_t &d, hip_dim_member_t value)
  {
    d.y = value;
  }
};

template<>
struct HipDimHelper<2>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static constexpr
  auto get(dim_t const &d)
  {
    return d.z;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static
  void set(dim_t &d, hip_dim_member_t value)
  {
    d.z = value;
  }
};

template<int dim, typename dim_t>
RAJA_HOST_DEVICE
constexpr
auto get_hip_dim(dim_t const &d)
{
  return HipDimHelper<dim>::get(d);
}

template<int dim, typename dim_t>
RAJA_HOST_DEVICE
void set_hip_dim(dim_t &d, hip_dim_member_t value)
{
  return HipDimHelper<dim>::set(d, value);
}


/// Type representing thread indexing within a block
/// block_size is fixed
template<int dim, hip_dim_member_t t_block_size>
struct HipIndexThread
{
  static constexpr hip_dim_member_t block_size = t_block_size;
  static constexpr hip_dim_member_t grid_size = 0;

  RAJA_HOST_DEVICE constexpr
  HipIndexThread(hip_dim_member_t RAJA_UNUSED_ARG(_block_size) = 0)
  { }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto index()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(threadIdx));
  }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static constexpr auto size()
  {
    return static_cast<IdxT>(block_size);
  }
};
/// unless t_block_size is 0 then block_size is dynamic
template<int dim>
struct HipIndexThread<dim, 0>
{
  static constexpr hip_dim_member_t grid_size = 0;

  hip_dim_member_t block_size;

  RAJA_HOST_DEVICE constexpr
  HipIndexThread(hip_dim_member_t _block_size = 0)
    : block_size(_block_size)
  { }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto index()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(threadIdx));
  }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto size()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(blockDim));
  }
};

/// Type representing block indexing within a grid
/// grid_size is fixed
template<int dim, hip_dim_member_t t_grid_size>
struct HipIndexBlock
{
  static constexpr hip_dim_member_t block_size = 0;
  static constexpr hip_dim_member_t grid_size = t_grid_size;

  RAJA_HOST_DEVICE constexpr
  HipIndexBlock(hip_dim_member_t RAJA_UNUSED_ARG(_grid_size) = 0)
  { }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto index()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(blockIdx));
  }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static constexpr auto size()
  {
    return static_cast<IdxT>(grid_size);
  }
};
/// unless t_grid_size is 0 then grid_size is dynamic
template<int dim>
struct HipIndexBlock<dim, 0>
{
  static constexpr hip_dim_member_t block_size = 0;

  hip_dim_member_t grid_size;

  RAJA_HOST_DEVICE constexpr
  HipIndexBlock(hip_dim_member_t _grid_size = 0)
    : grid_size(_grid_size)
  { }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto index()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(blockIdx));
  }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto size()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(gridDim));
  }
};

/// Type representing thread indexing within a grid
template<int dim, hip_dim_member_t t_block_size, hip_dim_member_t t_grid_size>
struct HipIndexGlobal
{
  static constexpr hip_dim_member_t block_size = t_block_size;
  static constexpr hip_dim_member_t grid_size = t_grid_size;

  RAJA_HOST_DEVICE constexpr
  HipIndexGlobal(hip_dim_member_t RAJA_UNUSED_ARG(_block_size) = 0,
                 hip_dim_member_t RAJA_UNUSED_ARG(_grid_size) = 0)
  { }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto index()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(block_size) *
           static_cast<IdxT>(HipDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static constexpr auto size()
  {
    return static_cast<IdxT>(block_size) *
           static_cast<IdxT>(grid_size) ;
  }
};
/// unless t_grid_size is 0 then grid_size is dynamic
template<int dim, hip_dim_member_t t_block_size>
struct HipIndexGlobal<dim, t_block_size, 0> {

  static constexpr hip_dim_member_t block_size = t_block_size;

  hip_dim_member_t grid_size;

  RAJA_HOST_DEVICE constexpr
  HipIndexGlobal(hip_dim_member_t RAJA_UNUSED_ARG(_block_size) = 0,
                 hip_dim_member_t _grid_size = 0)
    : grid_size(_grid_size)
  { }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto index()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(block_size) *
           static_cast<IdxT>(HipDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto size()
  {
    return static_cast<IdxT>(block_size) *
           static_cast<IdxT>(HipDimHelper<dim>::get(gridDim)) ;
  }
};
/// unless t_block_size is 0 then block_size is dynamic
template<int dim, hip_dim_member_t t_grid_size>
struct HipIndexGlobal<dim, 0, t_grid_size>
{
  static constexpr hip_dim_member_t grid_size = t_grid_size;

  hip_dim_member_t block_size = 0;

  RAJA_HOST_DEVICE constexpr
  HipIndexGlobal(hip_dim_member_t _block_size = 0,
                 hip_dim_member_t RAJA_UNUSED_ARG(_grid_size) = 0)
    : block_size(_block_size)
  { }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto index()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(HipDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(HipDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto size()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(grid_size) ;
  }
};
/// unless t_block_size and t_gris_size are 0 then block_size and grid_size are dynamic
template<int dim>
struct HipIndexGlobal<dim, 0, 0>
{
  hip_dim_member_t block_size = 0;
  hip_dim_member_t grid_size = 0;

  RAJA_HOST_DEVICE constexpr
  HipIndexGlobal(hip_dim_member_t _block_size = 0,
                 hip_dim_member_t _grid_size = 0)
    : block_size(_block_size)
    , grid_size(_grid_size)
  { }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto index()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(HipDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(HipDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = hip_dim_member_t >
  RAJA_DEVICE static inline auto size()
  {
    return static_cast<IdxT>(HipDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(HipDimHelper<dim>::get(gridDim)) ;
  }
};


// Indexer used with Hip forall policies
// each thread gets 0 or 1 indices
template<typename IndexMapper>
struct HipForallDirect;
/// each thread gets 0 or more indices
template<typename _IndexMapper>
struct HipForallLoop;

template<typename _IndexMapper>

} // namespace internal

namespace type_traits {

template <typename Indexer>
struct is_hip_direct_indexer : std::false_type {};

template <typename Indexer>
struct is_hip_loop_indexer : std::false_type {};

template <typename IndexMapper>
struct is_hip_block_size_known : std::false_type {};

template<int dim, hip_dim_member_t t_block_size>
struct is_hip_block_size_known<::RAJA::internal::HipIndexThread<dim, t_block_size>> : std::true_type {};
template<int dim>
struct is_hip_block_size_known<::RAJA::internal::HipIndexThread<dim, 0>> : std::false_type {};
template<int dim, hip_dim_member_t t_block_size, hip_dim_member_t t_grid_size>
struct is_hip_block_size_known<::RAJA::internal::HipIndexGlobal<dim, t_block_size, t_grid_size>> : std::true_type {};
template<int dim, hip_dim_member_t t_grid_size>
struct is_hip_block_size_known<::RAJA::internal::HipIndexGlobal<dim, 0, t_grid_size>> : std::false_type {};

} // namespace type_traits


template <size_t BLOCK_SIZE, bool Async = false>
using hip_exec = policy::hip::hip_exec<
    internal::HipForallDirect<internal::HipIndexGlobal<0, BLOCK_SIZE, 0>>,
    Async>;

template <size_t BLOCK_SIZE>
using hip_exec_async = policy::hip::hip_exec<
    internal::HipForallDirect<internal::HipIndexGlobal<0, BLOCK_SIZE, 0>>,
    true>;

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

using hip_thread_x_direct = internal::HipIndexDirect<internal::HipIndexThread<0, 0>>; // hip_thread_xyz_direct<0>;
using hip_thread_y_direct = internal::HipIndexDirect<internal::HipIndexThread<1, 0>>; // hip_thread_xyz_direct<1>;
using hip_thread_z_direct = internal::HipIndexDirect<internal::HipIndexThread<2, 0>>; // hip_thread_xyz_direct<2>;


/*!
 * Maps segment indices to HIP threads.
 * Uses block-stride looping to exceed the maximum number of physical threads
 */
template<int ... dim>
struct hip_thread_xyz_loop{};

using hip_thread_x_loop = internal::HipIndexLoop<internal::HipIndexThread<0, 0>>; // hip_thread_xyz_loop<0>;
using hip_thread_y_loop = internal::HipIndexLoop<internal::HipIndexThread<1, 0>>; // hip_thread_xyz_loop<1>;
using hip_thread_z_loop = internal::HipIndexLoop<internal::HipIndexThread<2, 0>>; // hip_thread_xyz_loop<2>;


/*!
 * Maps segment indices to HIP blocks.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical blocks to fit all of the direct map requests.
 */
template<int ... dim>
struct hip_block_xyz_direct{};

using hip_block_x_direct = internal::HipIndexDirect<internal::HipIndexBlock<0, 0>>; // hip_block_xyz_direct<0>;
using hip_block_y_direct = internal::HipIndexDirect<internal::HipIndexBlock<1, 0>>; // hip_block_xyz_direct<1>;
using hip_block_z_direct = internal::HipIndexDirect<internal::HipIndexBlock<2, 0>>; // hip_block_xyz_direct<2>;


/*!
 * Maps segment indices to HIP blocks.
 * Uses grid-stride looping to exceed the maximum number of blocks
 */
template<int ... dim>
struct hip_block_xyz_loop{};

using hip_block_x_loop = internal::HipIndexLoop<internal::HipIndexBlock<0, 0>>; // hip_block_xyz_loop<0>;
using hip_block_y_loop = internal::HipIndexLoop<internal::HipIndexBlock<1, 0>>; // hip_block_xyz_loop<1>;
using hip_block_z_loop = internal::HipIndexLoop<internal::HipIndexBlock<2, 0>>; // hip_block_xyz_loop<2>;


/*!
 * Maps segment indices to HIP global threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 */
template<size_t BLOCK_SIZE, size_t GRID_SIZE=0>
using hip_global_x_direct = internal::HipIndexDirect<internal::HipIndexGlobal<0, BLOCK_SIZE, GRID_SIZE>>;
template<size_t BLOCK_SIZE, size_t GRID_SIZE=0>
using hip_global_y_direct = internal::HipIndexDirect<internal::HipIndexGlobal<1, BLOCK_SIZE, GRID_SIZE>>;
template<size_t BLOCK_SIZE, size_t GRID_SIZE=0>
using hip_global_z_direct = internal::HipIndexDirect<internal::HipIndexGlobal<2, BLOCK_SIZE, GRID_SIZE>>;


/*!
 * Maps segment indices to HIP global threads.
 * Uses grid-stride looping to exceed the maximum number of global threads
 */
template<size_t BLOCK_SIZE, size_t GRID_SIZE=0>
using hip_global_x_loop = internal::HipIndexLoop<internal::HipIndexGlobal<0, BLOCK_SIZE, GRID_SIZE>>;
template<size_t BLOCK_SIZE, size_t GRID_SIZE=0>
using hip_global_y_loop = internal::HipIndexLoop<internal::HipIndexGlobal<1, BLOCK_SIZE, GRID_SIZE>>;
template<size_t BLOCK_SIZE, size_t GRID_SIZE=0>
using hip_global_z_loop = internal::HipIndexLoop<internal::HipIndexGlobal<2, BLOCK_SIZE, GRID_SIZE>>;

}  // namespace RAJA

#endif  // RAJA_ENABLE_HIP
#endif
