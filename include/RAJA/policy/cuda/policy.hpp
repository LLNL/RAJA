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
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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
#include "RAJA/policy/sequential/policy.hpp"

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

namespace cuda
{

/// Type representing thread and block indexing within a grid
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE>
struct IndexGlobal;

template<typename ...indexers>
struct IndexFlatten;

}  // namespace cuda

namespace policy
{
namespace cuda
{

constexpr const size_t MIN_BLOCKS_PER_SM = 1;
constexpr const size_t MAX_BLOCKS_PER_SM = 32;

template <typename _IterationMapping, kernel_sync_requirement sync, typename ... _IterationGetters>
struct cuda_indexer {};

template <typename _IterationMapping, kernel_sync_requirement sync, typename ... _IterationGetters>
struct cuda_flatten_indexer : public RAJA::make_policy_pattern_launch_platform_t<
  RAJA::Policy::cuda,
  RAJA::Pattern::region,
  detail::get_launch<true /*async */>::value,
  RAJA::Platform::cuda> {
  using IterationGetter = RAJA::cuda::IndexFlatten<_IterationGetters...>;
};

template <typename _IterationMapping, typename _IterationGetter, size_t BLOCKS_PER_SM = policy::cuda::MIN_BLOCKS_PER_SM, bool Async = false>
struct cuda_exec_explicit : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::cuda,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::cuda> {
  using IterationMapping = _IterationMapping;
  using IterationGetter = _IterationGetter;
};

template <bool Async, int num_threads = named_usage::unspecified, size_t BLOCKS_PER_SM = policy::cuda::MIN_BLOCKS_PER_SM>
struct cuda_launch_explicit_t : public RAJA::make_policy_pattern_launch_platform_t<
                                RAJA::Policy::cuda,
                                RAJA::Pattern::region,
                                detail::get_launch<Async>::value,
                                RAJA::Platform::cuda> {
};




//
// NOTE: There is no Index set segment iteration policy for CUDA
//

///
/// WorkGroup execution policies
///
template <size_t BLOCK_SIZE, size_t BLOCKS_PER_SM = policy::cuda::MIN_BLOCKS_PER_SM, bool Async = false>
struct cuda_work_explicit : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::cuda,
                       RAJA::Pattern::workgroup_exec,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::cuda> {
};

/// execute the enqueued loops in an unordered fashion by mapping loops to
/// blocks in the y direction and loop iterations to threads in the x direction
/// with the size of the x direction being the average of the iteration counts
/// of all the loops
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

/*!
 * Cuda atomic policy for using cuda atomics on the device and
 * the provided policy on the host
 */
template<typename host_policy>
struct cuda_atomic_explicit{};

/*!
 * Default cuda atomic policy uses cuda atomics on the device and non-atomics
 * on the host
 */
using cuda_atomic = cuda_atomic_explicit<seq_atomic>;

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


namespace internal
{

RAJA_INLINE
int get_size(cuda_dim_t dims)
{
  if(dims.x == 0 && dims.y == 0 && dims.z == 0){
    return 0;
  }
  return (dims.x ? dims.x : 1) *
         (dims.y ? dims.y : 1) *
         (dims.z ? dims.z : 1);
}

struct CudaDims {

  cuda_dim_t blocks{0,0,0};
  cuda_dim_t threads{0,0,0};

  CudaDims() = default;
  CudaDims(CudaDims const&) = default;
  CudaDims& operator=(CudaDims const&) = default;

  RAJA_INLINE
  CudaDims(cuda_dim_member_t default_val)
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
  cuda_dim_t get_blocks() const {
    if (num_blocks() != 0) {
      return {(blocks.x ? blocks.x : 1),
              (blocks.y ? blocks.y : 1),
              (blocks.z ? blocks.z : 1)};
    } else {
      return blocks;
    }
  }

  RAJA_INLINE
  cuda_dim_t get_threads() const {
    if (num_threads() != 0) {
      return {(threads.x ? threads.x : 1),
              (threads.y ? threads.y : 1),
              (threads.z ? threads.z : 1)};
    } else {
      return threads;
    }
  }
};

template<named_dim dim>
struct CudaDimHelper;

template<>
struct CudaDimHelper<named_dim::x>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static constexpr
  cuda_dim_member_t get(dim_t const &d)
  {
    return d.x;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static
  void set(dim_t &d, cuda_dim_member_t value)
  {
    d.x = value;
  }
};

template<>
struct CudaDimHelper<named_dim::y>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static constexpr
  cuda_dim_member_t get(dim_t const &d)
  {
    return d.y;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static
  void set(dim_t &d, cuda_dim_member_t value)
  {
    d.y = value;
  }
};

template<>
struct CudaDimHelper<named_dim::z>{

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static constexpr
  cuda_dim_member_t get(dim_t const &d)
  {
    return d.z;
  }

  template<typename dim_t>
  RAJA_HOST_DEVICE
  inline static
  void set(dim_t &d, cuda_dim_member_t value)
  {
    d.z = value;
  }
};

template<named_dim dim, typename dim_t>
RAJA_HOST_DEVICE
constexpr
cuda_dim_member_t get_cuda_dim(dim_t const &d)
{
  return CudaDimHelper<dim>::get(d);
}

template<named_dim dim, typename dim_t>
RAJA_HOST_DEVICE
void set_cuda_dim(dim_t &d, cuda_dim_member_t value)
{
  return CudaDimHelper<dim>::set(d, value);
}

} // namespace internal

namespace cuda
{

/// specify block size and grid size for one dimension at runtime
struct IndexSize
{
  cuda_dim_member_t block_size = named_usage::unspecified;
  cuda_dim_member_t grid_size = named_usage::unspecified;

  RAJA_HOST_DEVICE constexpr
  IndexSize(cuda_dim_member_t _block_size = named_usage::unspecified,
            cuda_dim_member_t _grid_size = named_usage::unspecified)
    : block_size(_block_size)
    , grid_size(_grid_size)
  { }
};

/// Type representing thread indexing within a grid
/// It has various specializations that optimize specific patterns

/// useful for global indexing
/// with fixed block size and fixed grid size
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE>
struct IndexGlobal
{
  static_assert(BLOCK_SIZE > 0, "block size must not be negative");
  static_assert(GRID_SIZE > 0, "grid size must not be negative");

  static constexpr int block_size = BLOCK_SIZE;
  static constexpr int grid_size = GRID_SIZE;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(block_size) *
           static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(block_size) *
           static_cast<IdxT>(grid_size) ;
  }
};
/// with fixed block size of 1 and fixed grid size
template<named_dim dim, int GRID_SIZE>
struct IndexGlobal<dim, 1, GRID_SIZE>
{
  static_assert(GRID_SIZE > 0, "grid size must not be negative");

  static constexpr int block_size = 1;
  static constexpr int grid_size = GRID_SIZE;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(grid_size) ;
  }
};
/// with fixed block size and fixed grid size of 1
template<named_dim dim, int BLOCK_SIZE>
struct IndexGlobal<dim, BLOCK_SIZE, 1>
{
  static_assert(BLOCK_SIZE > 0, "block size must not be negative");

  static constexpr int block_size = BLOCK_SIZE;
  static constexpr int grid_size = 1;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(threadIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(block_size) ;
  }
};
/// with fixed block size and fixed grid size of 1
template<named_dim dim>
struct IndexGlobal<dim, 1, 1>
{
  static constexpr int block_size = 1;
  static constexpr int grid_size = 1;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(0) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(1) ;
  }
};

/// with dynamic block size and fixed grid size
template<named_dim dim, int GRID_SIZE>
struct IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>
{
  static_assert(GRID_SIZE > 0, "grid size must not be negative");

  static constexpr int block_size = named_usage::unspecified;
  static constexpr int grid_size = GRID_SIZE;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(grid_size) ;
  }
};
/// with dynamic block size and fixed grid size of 1
template<named_dim dim>
struct IndexGlobal<dim, named_usage::unspecified, 1>
{
  static constexpr int block_size = named_usage::unspecified;
  static constexpr int grid_size = 1;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(threadIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockDim)) ;
  }
};

/// with fixed block size and dynamic grid size
template<named_dim dim, int BLOCK_SIZE>
struct IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>
{
  static_assert(BLOCK_SIZE > 0, "block size must not be negative");

  static constexpr int block_size = BLOCK_SIZE;
  static constexpr int grid_size = named_usage::unspecified;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(block_size) *
           static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(block_size) *
           static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(gridDim)) ;
  }
};
/// with fixed block size of 1 and dynamic grid size
template<named_dim dim>
struct IndexGlobal<dim, 1, named_usage::unspecified>
{
  static constexpr int block_size = 1;
  static constexpr int grid_size = named_usage::unspecified;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(gridDim)) ;
  }
};

/// with dynamic block size and dynamic grid size
template<named_dim dim>
struct IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>
{
  static constexpr int block_size = named_usage::unspecified;
  static constexpr int grid_size = named_usage::unspecified;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(threadIdx)) +
           static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockDim)) *
           static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(gridDim)) ;
  }
};

/// useful for indexing blocks (ignores thread indices)
/// with fixed grid size
template<named_dim dim, int GRID_SIZE>
struct IndexGlobal<dim, named_usage::ignored, GRID_SIZE>
{
  static_assert(GRID_SIZE > 0, "grid size must not be negative");

  static constexpr int block_size = named_usage::ignored;
  static constexpr int grid_size = GRID_SIZE;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(grid_size) ;
  }
};
/// with fixed grid sized of 1
template<named_dim dim>
struct IndexGlobal<dim, named_usage::ignored, 1>
{
  static constexpr int block_size = named_usage::ignored;
  static constexpr int grid_size = 1;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(0) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(1) ;
  }
};
/// with dynamic grid size
template<named_dim dim>
struct IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>
{
  static constexpr int block_size = named_usage::ignored;
  static constexpr int grid_size = named_usage::unspecified;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(gridDim)) ;
  }
};

/// useful for indexing threads (ignores block indices)
/// with fixed block size
template<named_dim dim, int BLOCK_SIZE>
struct IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>
{
  static_assert(BLOCK_SIZE > 0, "block size must not be negative");

  static constexpr int block_size = BLOCK_SIZE;
  static constexpr int grid_size = named_usage::ignored;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(threadIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static constexpr IdxT size()
  {
    return static_cast<IdxT>(block_size) ;
  }
};
/// with fixed block size of 1
template<named_dim dim>
struct IndexGlobal<dim, 1, named_usage::ignored>
{
  static constexpr int block_size = 1;
  static constexpr int grid_size = named_usage::ignored;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(0) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(1) ;
  }
};
/// with dynamic block size
template<named_dim dim>
struct IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>
{
  static constexpr int block_size = named_usage::unspecified;
  static constexpr int grid_size = named_usage::ignored;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(threadIdx)) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(::RAJA::internal::CudaDimHelper<dim>::get(blockDim)) ;
  }
};

/// useful for doing single threaded sequential tasks
/// (ignores thread and block indices)
template<named_dim dim>
struct IndexGlobal<dim, named_usage::ignored, named_usage::ignored>
{
  static constexpr int block_size = named_usage::ignored;
  static constexpr int grid_size = named_usage::ignored;

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {
    return static_cast<IdxT>(0) ;
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return static_cast<IdxT>(1) ;
  }
};

// useful for flatten global index (includes x)
template<typename x_index>
struct IndexFlatten<x_index>
{

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {

    return x_index::template index<IdxT>();
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return  x_index::template size<IdxT>();
  }

};

// useful for flatten global index (includes x,y)
template<typename x_index, typename y_index>
struct IndexFlatten<x_index, y_index>
{

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {

    return x_index::template index<IdxT>() +
      x_index::template size<IdxT>() * ( y_index::template index<IdxT>());

  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return  x_index::template size<IdxT>() * y_index::template size<IdxT> ();
  }

};

// useful for flatten global index (includes x,y,z)
template<typename x_index, typename y_index, typename z_index>
struct IndexFlatten<x_index, y_index, z_index>
{

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT index()
  {

    return x_index::template index<IdxT>() +
      x_index::template size<IdxT>() * ( y_index::template index<IdxT>() +
                                         y_index::template size<IdxT>() * z_index::template index<IdxT>());
  }

  template < typename IdxT = cuda_dim_member_t >
  RAJA_DEVICE static inline IdxT size()
  {
    return  x_index::template size<IdxT>() * y_index::template size<IdxT> () * z_index::template size<IdxT> ();
  }

};


// helper to get just the thread indexing part of IndexGlobal
template < typename index_global >
struct get_index_thread;
///
template < named_dim dim, int BLOCK_SIZE, int GRID_SIZE >
struct get_index_thread<IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>
{
  using type = IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>;
};
///
template <typename x_index, typename y_index, typename z_index>
struct get_index_thread<IndexFlatten<x_index, y_index, z_index>>
{
  using type = IndexFlatten<typename get_index_thread<x_index>::type,
                            typename get_index_thread<y_index>::type,
                            typename get_index_thread<z_index>::type>;
};

// helper to get just the block indexing part of IndexGlobal
template < typename index_global >
struct get_index_block;
///
template < named_dim dim, int BLOCK_SIZE, int GRID_SIZE >
struct get_index_block<IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>
{
  using type = IndexGlobal<dim, named_usage::ignored, GRID_SIZE>;
};
///
template <typename x_index, typename y_index, typename z_index>
struct get_index_block<IndexFlatten<x_index, y_index, z_index>>
{
  using type = IndexFlatten<typename get_index_block<x_index>::type,
                            typename get_index_block<y_index>::type,
                            typename get_index_block<z_index>::type>;
};


template <size_t BLOCK_SIZE=named_usage::unspecified>
using thread_x = IndexGlobal<named_dim::x, BLOCK_SIZE, named_usage::ignored>;
template <size_t BLOCK_SIZE=named_usage::unspecified>
using thread_y = IndexGlobal<named_dim::y, BLOCK_SIZE, named_usage::ignored>;
template <size_t BLOCK_SIZE=named_usage::unspecified>
using thread_z = IndexGlobal<named_dim::z, BLOCK_SIZE, named_usage::ignored>;

template <size_t GRID_SIZE=named_usage::unspecified>
using block_x = IndexGlobal<named_dim::x, named_usage::ignored, GRID_SIZE>;
template <size_t GRID_SIZE=named_usage::unspecified>
using block_y = IndexGlobal<named_dim::y, named_usage::ignored, GRID_SIZE>;
template <size_t GRID_SIZE=named_usage::unspecified>
using block_z = IndexGlobal<named_dim::z, named_usage::ignored, GRID_SIZE>;

template <size_t BLOCK_SIZE, size_t GRID_SIZE=named_usage::unspecified>
using global_x = IndexGlobal<named_dim::x, BLOCK_SIZE, GRID_SIZE>;
template <size_t BLOCK_SIZE, size_t GRID_SIZE=named_usage::unspecified>
using global_y = IndexGlobal<named_dim::y, BLOCK_SIZE, GRID_SIZE>;
template <size_t BLOCK_SIZE, size_t GRID_SIZE=named_usage::unspecified>
using global_z = IndexGlobal<named_dim::z, BLOCK_SIZE, GRID_SIZE>;

} // namespace cuda

// policies usable with forall, scan, and sort
template <size_t BLOCK_SIZE, size_t GRID_SIZE, size_t BLOCKS_PER_SM, bool Async = false>
using cuda_exec_grid_explicit = policy::cuda::cuda_exec_explicit<
    iteration_mapping::StridedLoop, cuda::global_x<BLOCK_SIZE, GRID_SIZE>, BLOCKS_PER_SM, Async>;

template <size_t BLOCK_SIZE, size_t GRID_SIZE, size_t BLOCKS_PER_SM>
using cuda_exec_grid_explicit_async = policy::cuda::cuda_exec_explicit<
    iteration_mapping::StridedLoop, cuda::global_x<BLOCK_SIZE, GRID_SIZE>, BLOCKS_PER_SM, true>;

template <size_t BLOCK_SIZE, size_t GRID_SIZE, bool Async = false>
using cuda_exec_grid = policy::cuda::cuda_exec_explicit<
    iteration_mapping::StridedLoop, cuda::global_x<BLOCK_SIZE, GRID_SIZE>, policy::cuda::MIN_BLOCKS_PER_SM, Async>;

template <size_t BLOCK_SIZE, size_t GRID_SIZE>
using cuda_exec_grid_async = policy::cuda::cuda_exec_explicit<
    iteration_mapping::StridedLoop, cuda::global_x<BLOCK_SIZE, GRID_SIZE>, policy::cuda::MIN_BLOCKS_PER_SM, true>;

template <size_t BLOCK_SIZE, size_t BLOCKS_PER_SM, bool Async = false>
using cuda_exec_explicit = policy::cuda::cuda_exec_explicit<
    iteration_mapping::Direct, cuda::global_x<BLOCK_SIZE>, BLOCKS_PER_SM, Async>;

template <size_t BLOCK_SIZE, size_t BLOCKS_PER_SM>
using cuda_exec_explicit_async = policy::cuda::cuda_exec_explicit<
    iteration_mapping::Direct, cuda::global_x<BLOCK_SIZE>, BLOCKS_PER_SM, true>;

template <size_t BLOCK_SIZE, bool Async = false>
using cuda_exec = policy::cuda::cuda_exec_explicit<
    iteration_mapping::Direct, cuda::global_x<BLOCK_SIZE>, policy::cuda::MIN_BLOCKS_PER_SM, Async>;

template <size_t BLOCK_SIZE>
using cuda_exec_async = policy::cuda::cuda_exec_explicit<
    iteration_mapping::Direct, cuda::global_x<BLOCK_SIZE>, policy::cuda::MIN_BLOCKS_PER_SM, true>;

template <size_t BLOCK_SIZE, size_t BLOCKS_PER_SM, bool Async = false>
using cuda_exec_occ_calc_explicit = policy::cuda::cuda_exec_explicit<
    iteration_mapping::StridedLoop, cuda::global_x<BLOCK_SIZE>, BLOCKS_PER_SM, Async>;

template <size_t BLOCK_SIZE, size_t BLOCKS_PER_SM>
using cuda_exec_occ_calc_explicit_async = policy::cuda::cuda_exec_explicit<
    iteration_mapping::StridedLoop, cuda::global_x<BLOCK_SIZE>, BLOCKS_PER_SM, true>;

template <size_t BLOCK_SIZE, bool Async = false>
using cuda_exec_occ_calc = policy::cuda::cuda_exec_explicit<
    iteration_mapping::StridedLoop, cuda::global_x<BLOCK_SIZE>, policy::cuda::MIN_BLOCKS_PER_SM, Async>;

template <size_t BLOCK_SIZE>
using cuda_exec_occ_calc_async = policy::cuda::cuda_exec_explicit<
    iteration_mapping::StridedLoop, cuda::global_x<BLOCK_SIZE>, policy::cuda::MIN_BLOCKS_PER_SM, true>;

// policies usable with WorkGroup
template <size_t BLOCK_SIZE, size_t BLOCKS_PER_SM = policy::cuda::MIN_BLOCKS_PER_SM, bool Async = false>
using cuda_work_explicit = policy::cuda::cuda_work_explicit<BLOCK_SIZE, BLOCKS_PER_SM, Async>;

template <size_t BLOCK_SIZE, size_t BLOCKS_PER_SM = policy::cuda::MIN_BLOCKS_PER_SM>
using cuda_work_explicit_async = policy::cuda::cuda_work_explicit<BLOCK_SIZE, BLOCKS_PER_SM, true>;

template <size_t BLOCK_SIZE, bool Async = false>
using cuda_work = policy::cuda::cuda_work_explicit<BLOCK_SIZE, policy::cuda::MIN_BLOCKS_PER_SM, Async>;

template <size_t BLOCK_SIZE>
using cuda_work_async = policy::cuda::cuda_work_explicit<BLOCK_SIZE, policy::cuda::MIN_BLOCKS_PER_SM, true>;

using policy::cuda::unordered_cuda_loop_y_block_iter_x_threadblock_average;

// policies usable with atomics
using policy::cuda::cuda_atomic;
using policy::cuda::cuda_atomic_explicit;

// policies usable with reducers
using policy::cuda::cuda_reduce_base;
using policy::cuda::cuda_reduce;
using policy::cuda::cuda_reduce_atomic;

// policies usable with kernel
using policy::cuda::cuda_block_reduce;
using policy::cuda::cuda_warp_reduce;

using cuda_warp_direct = RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    kernel_sync_requirement::none,
    cuda::thread_x<RAJA::policy::cuda::WARP_SIZE>>;
using cuda_warp_loop = RAJA::policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop,
    kernel_sync_requirement::none,
    cuda::thread_x<RAJA::policy::cuda::WARP_SIZE>>;

using policy::cuda::cuda_warp_masked_direct;
using policy::cuda::cuda_warp_masked_loop;

using policy::cuda::cuda_thread_masked_direct;
using policy::cuda::cuda_thread_masked_loop;

// policies usable with synchronize
using policy::cuda::cuda_synchronize;

// policies usable with launch
template <bool Async, int num_threads = named_usage::unspecified, size_t BLOCKS_PER_SM = policy::cuda::MIN_BLOCKS_PER_SM>
using cuda_launch_explicit_t = policy::cuda::cuda_launch_explicit_t<Async, num_threads, BLOCKS_PER_SM>;

template <bool Async, int num_threads = named_usage::unspecified>
using cuda_launch_t = policy::cuda::cuda_launch_explicit_t<Async, num_threads, policy::cuda::MIN_BLOCKS_PER_SM>;


// policies usable with kernel and launch
template < typename ... indexers >
using cuda_indexer_direct = policy::cuda::cuda_indexer<
    iteration_mapping::Direct,
    kernel_sync_requirement::none,
    indexers...>;

template < typename ... indexers >
using cuda_indexer_loop = policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop,
    kernel_sync_requirement::none,
    indexers...>;

template < typename ... indexers >
using cuda_indexer_syncable_loop = policy::cuda::cuda_indexer<
    iteration_mapping::StridedLoop,
    kernel_sync_requirement::sync,
    indexers...>;

template < typename ... indexers >
using cuda_flatten_indexer_direct = policy::cuda::cuda_flatten_indexer<
    iteration_mapping::Direct,
    kernel_sync_requirement::none,
    indexers...>;

template < typename ... indexers >
using cuda_flatten_indexer_loop = policy::cuda::cuda_flatten_indexer<
    iteration_mapping::StridedLoop,
    kernel_sync_requirement::none,
    indexers...>;

/*!
 * Maps segment indices to CUDA threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * For example, a segment of size 2000 will not fit, and trigger a runtime
 * error.
 */
template < named_dim ... dims >
using cuda_thread_direct = cuda_indexer_direct<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;

using cuda_thread_x_direct = cuda_thread_direct<named_dim::x>;
using cuda_thread_y_direct = cuda_thread_direct<named_dim::y>;
using cuda_thread_z_direct = cuda_thread_direct<named_dim::z>;

using cuda_thread_xy_direct = cuda_thread_direct<named_dim::x, named_dim::y>;
using cuda_thread_xz_direct = cuda_thread_direct<named_dim::x, named_dim::z>;
using cuda_thread_yx_direct = cuda_thread_direct<named_dim::y, named_dim::x>;
using cuda_thread_yz_direct = cuda_thread_direct<named_dim::y, named_dim::z>;
using cuda_thread_zx_direct = cuda_thread_direct<named_dim::z, named_dim::x>;
using cuda_thread_zy_direct = cuda_thread_direct<named_dim::z, named_dim::y>;

using cuda_thread_xyz_direct = cuda_thread_direct<named_dim::x, named_dim::y, named_dim::z>;
using cuda_thread_xzy_direct = cuda_thread_direct<named_dim::x, named_dim::z, named_dim::y>;
using cuda_thread_yxz_direct = cuda_thread_direct<named_dim::y, named_dim::x, named_dim::z>;
using cuda_thread_yzx_direct = cuda_thread_direct<named_dim::y, named_dim::z, named_dim::x>;
using cuda_thread_zxy_direct = cuda_thread_direct<named_dim::z, named_dim::x, named_dim::y>;
using cuda_thread_zyx_direct = cuda_thread_direct<named_dim::z, named_dim::y, named_dim::x>;

/*!
 * Maps segment indices to CUDA threads.
 * Uses block-stride looping to exceed the maximum number of physical threads
 */
template < named_dim ... dims >
using cuda_thread_loop = cuda_indexer_loop<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;

template < named_dim ... dims >
using cuda_thread_syncable_loop = cuda_indexer_syncable_loop<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;

using cuda_thread_x_loop = cuda_thread_loop<named_dim::x>;
using cuda_thread_y_loop = cuda_thread_loop<named_dim::y>;
using cuda_thread_z_loop = cuda_thread_loop<named_dim::z>;

using cuda_thread_xy_loop = cuda_thread_loop<named_dim::x, named_dim::y>;
using cuda_thread_xz_loop = cuda_thread_loop<named_dim::x, named_dim::z>;
using cuda_thread_yx_loop = cuda_thread_loop<named_dim::y, named_dim::x>;
using cuda_thread_yz_loop = cuda_thread_loop<named_dim::y, named_dim::z>;
using cuda_thread_zx_loop = cuda_thread_loop<named_dim::z, named_dim::x>;
using cuda_thread_zy_loop = cuda_thread_loop<named_dim::z, named_dim::y>;

using cuda_thread_xyz_loop = cuda_thread_loop<named_dim::x, named_dim::y, named_dim::z>;
using cuda_thread_xzy_loop = cuda_thread_loop<named_dim::x, named_dim::z, named_dim::y>;
using cuda_thread_yxz_loop = cuda_thread_loop<named_dim::y, named_dim::x, named_dim::z>;
using cuda_thread_yzx_loop = cuda_thread_loop<named_dim::y, named_dim::z, named_dim::x>;
using cuda_thread_zxy_loop = cuda_thread_loop<named_dim::z, named_dim::x, named_dim::y>;
using cuda_thread_zyx_loop = cuda_thread_loop<named_dim::z, named_dim::y, named_dim::x>;

/*
 * Maps segment indices to flattened CUDA threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 * Reshapes multiple physical threads into a 1D iteration space
 */
template < named_dim ... dims >
using cuda_flatten_thread_direct = cuda_flatten_indexer_direct<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;

using cuda_flatten_thread_x_direct = cuda_flatten_thread_direct<named_dim::x>;
using cuda_flatten_thread_y_direct = cuda_flatten_thread_direct<named_dim::y>;
using cuda_flatten_thread_z_direct = cuda_flatten_thread_direct<named_dim::z>;

using cuda_flatten_thread_xy_direct = cuda_flatten_thread_direct<named_dim::x, named_dim::y>;
using cuda_flatten_thread_xz_direct = cuda_flatten_thread_direct<named_dim::x, named_dim::z>;
using cuda_flatten_thread_yx_direct = cuda_flatten_thread_direct<named_dim::y, named_dim::x>;
using cuda_flatten_thread_yz_direct = cuda_flatten_thread_direct<named_dim::y, named_dim::z>;
using cuda_flatten_thread_zx_direct = cuda_flatten_thread_direct<named_dim::z, named_dim::x>;
using cuda_flatten_thread_zy_direct = cuda_flatten_thread_direct<named_dim::z, named_dim::y>;

using cuda_flatten_thread_xyz_direct = cuda_flatten_thread_direct<named_dim::x, named_dim::y, named_dim::z>;
using cuda_flatten_thread_xzy_direct = cuda_flatten_thread_direct<named_dim::x, named_dim::z, named_dim::y>;
using cuda_flatten_thread_yxz_direct = cuda_flatten_thread_direct<named_dim::y, named_dim::x, named_dim::z>;
using cuda_flatten_thread_yzx_direct = cuda_flatten_thread_direct<named_dim::y, named_dim::z, named_dim::x>;
using cuda_flatten_thread_zxy_direct = cuda_flatten_thread_direct<named_dim::z, named_dim::x, named_dim::y>;
using cuda_flatten_thread_zyx_direct = cuda_flatten_thread_direct<named_dim::z, named_dim::y, named_dim::x>;

/*
 * Maps segment indices to flattened CUDA threads.
 * Reshapes multiple physical threads into a 1D iteration space
 * Uses block-stride looping to exceed the maximum number of physical threads
 */
template < named_dim ... dims >
using cuda_flatten_thread_loop = cuda_flatten_indexer_loop<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::ignored>...>;

using cuda_flatten_thread_x_loop = cuda_flatten_thread_loop<named_dim::x>;
using cuda_flatten_thread_y_loop = cuda_flatten_thread_loop<named_dim::y>;
using cuda_flatten_thread_z_loop = cuda_flatten_thread_loop<named_dim::z>;

using cuda_flatten_thread_xy_loop = cuda_flatten_thread_loop<named_dim::x, named_dim::y>;
using cuda_flatten_thread_xz_loop = cuda_flatten_thread_loop<named_dim::x, named_dim::z>;
using cuda_flatten_thread_yx_loop = cuda_flatten_thread_loop<named_dim::y, named_dim::x>;
using cuda_flatten_thread_yz_loop = cuda_flatten_thread_loop<named_dim::y, named_dim::z>;
using cuda_flatten_thread_zx_loop = cuda_flatten_thread_loop<named_dim::z, named_dim::x>;
using cuda_flatten_thread_zy_loop = cuda_flatten_thread_loop<named_dim::z, named_dim::y>;

using cuda_flatten_thread_xyz_loop = cuda_flatten_thread_loop<named_dim::x, named_dim::y, named_dim::z>;
using cuda_flatten_thread_xzy_loop = cuda_flatten_thread_loop<named_dim::x, named_dim::z, named_dim::y>;
using cuda_flatten_thread_yxz_loop = cuda_flatten_thread_loop<named_dim::y, named_dim::x, named_dim::z>;
using cuda_flatten_thread_yzx_loop = cuda_flatten_thread_loop<named_dim::y, named_dim::z, named_dim::x>;
using cuda_flatten_thread_zxy_loop = cuda_flatten_thread_loop<named_dim::z, named_dim::x, named_dim::y>;
using cuda_flatten_thread_zyx_loop = cuda_flatten_thread_loop<named_dim::z, named_dim::y, named_dim::x>;


/*!
 * Maps segment indices to CUDA blocks.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical blocks to fit all of the direct map requests.
 */
template < named_dim ... dims >
using cuda_block_direct = cuda_indexer_direct<
    cuda::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;

using cuda_block_x_direct = cuda_block_direct<named_dim::x>;
using cuda_block_y_direct = cuda_block_direct<named_dim::y>;
using cuda_block_z_direct = cuda_block_direct<named_dim::z>;

using cuda_block_xy_direct = cuda_block_direct<named_dim::x, named_dim::y>;
using cuda_block_xz_direct = cuda_block_direct<named_dim::x, named_dim::z>;
using cuda_block_yx_direct = cuda_block_direct<named_dim::y, named_dim::x>;
using cuda_block_yz_direct = cuda_block_direct<named_dim::y, named_dim::z>;
using cuda_block_zx_direct = cuda_block_direct<named_dim::z, named_dim::x>;
using cuda_block_zy_direct = cuda_block_direct<named_dim::z, named_dim::y>;

using cuda_block_xyz_direct = cuda_block_direct<named_dim::x, named_dim::y, named_dim::z>;
using cuda_block_xzy_direct = cuda_block_direct<named_dim::x, named_dim::z, named_dim::y>;
using cuda_block_yxz_direct = cuda_block_direct<named_dim::y, named_dim::x, named_dim::z>;
using cuda_block_yzx_direct = cuda_block_direct<named_dim::y, named_dim::z, named_dim::x>;
using cuda_block_zxy_direct = cuda_block_direct<named_dim::z, named_dim::x, named_dim::y>;
using cuda_block_zyx_direct = cuda_block_direct<named_dim::z, named_dim::y, named_dim::x>;

/*!
 * Maps segment indices to CUDA blocks.
 * Uses grid-stride looping to exceed the maximum number of blocks
 */
template < named_dim ... dims >
using cuda_block_loop = cuda_indexer_loop<
    cuda::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;

template < named_dim ... dims >
using cuda_block_syncable_loop = cuda_indexer_syncable_loop<
    cuda::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;

using cuda_block_x_loop = cuda_block_loop<named_dim::x>;
using cuda_block_y_loop = cuda_block_loop<named_dim::y>;
using cuda_block_z_loop = cuda_block_loop<named_dim::z>;

using cuda_block_xy_loop = cuda_block_loop<named_dim::x, named_dim::y>;
using cuda_block_xz_loop = cuda_block_loop<named_dim::x, named_dim::z>;
using cuda_block_yx_loop = cuda_block_loop<named_dim::y, named_dim::x>;
using cuda_block_yz_loop = cuda_block_loop<named_dim::y, named_dim::z>;
using cuda_block_zx_loop = cuda_block_loop<named_dim::z, named_dim::x>;
using cuda_block_zy_loop = cuda_block_loop<named_dim::z, named_dim::y>;

using cuda_block_xyz_loop = cuda_block_loop<named_dim::x, named_dim::y, named_dim::z>;
using cuda_block_xzy_loop = cuda_block_loop<named_dim::x, named_dim::z, named_dim::y>;
using cuda_block_yxz_loop = cuda_block_loop<named_dim::y, named_dim::x, named_dim::z>;
using cuda_block_yzx_loop = cuda_block_loop<named_dim::y, named_dim::z, named_dim::x>;
using cuda_block_zxy_loop = cuda_block_loop<named_dim::z, named_dim::x, named_dim::y>;
using cuda_block_zyx_loop = cuda_block_loop<named_dim::z, named_dim::y, named_dim::x>;

/*
 * Maps segment indices to flattened CUDA blocks.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical blocks to fit all of the direct map requests.
 * Reshapes multiple physical blocks into a 1D iteration space
 */
template < named_dim ... dims >
using cuda_flatten_block_direct = cuda_flatten_indexer_direct<
    cuda::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;

using cuda_flatten_block_x_direct = cuda_flatten_block_direct<named_dim::x>;
using cuda_flatten_block_y_direct = cuda_flatten_block_direct<named_dim::y>;
using cuda_flatten_block_z_direct = cuda_flatten_block_direct<named_dim::z>;

using cuda_flatten_block_xy_direct = cuda_flatten_block_direct<named_dim::x, named_dim::y>;
using cuda_flatten_block_xz_direct = cuda_flatten_block_direct<named_dim::x, named_dim::z>;
using cuda_flatten_block_yx_direct = cuda_flatten_block_direct<named_dim::y, named_dim::x>;
using cuda_flatten_block_yz_direct = cuda_flatten_block_direct<named_dim::y, named_dim::z>;
using cuda_flatten_block_zx_direct = cuda_flatten_block_direct<named_dim::z, named_dim::x>;
using cuda_flatten_block_zy_direct = cuda_flatten_block_direct<named_dim::z, named_dim::y>;

using cuda_flatten_block_xyz_direct = cuda_flatten_block_direct<named_dim::x, named_dim::y, named_dim::z>;
using cuda_flatten_block_xzy_direct = cuda_flatten_block_direct<named_dim::x, named_dim::z, named_dim::y>;
using cuda_flatten_block_yxz_direct = cuda_flatten_block_direct<named_dim::y, named_dim::x, named_dim::z>;
using cuda_flatten_block_yzx_direct = cuda_flatten_block_direct<named_dim::y, named_dim::z, named_dim::x>;
using cuda_flatten_block_zxy_direct = cuda_flatten_block_direct<named_dim::z, named_dim::x, named_dim::y>;
using cuda_flatten_block_zyx_direct = cuda_flatten_block_direct<named_dim::z, named_dim::y, named_dim::x>;

/*
 * Maps segment indices to flattened CUDA blocks.
 * Reshapes multiple physical blocks into a 1D iteration space
 * Uses block-stride looping to exceed the maximum number of physical blocks
 */
template < named_dim ... dims >
using cuda_flatten_block_loop = cuda_flatten_indexer_loop<
    cuda::IndexGlobal<dims, named_usage::ignored, named_usage::unspecified>...>;

using cuda_flatten_block_x_loop = cuda_flatten_block_loop<named_dim::x>;
using cuda_flatten_block_y_loop = cuda_flatten_block_loop<named_dim::y>;
using cuda_flatten_block_z_loop = cuda_flatten_block_loop<named_dim::z>;

using cuda_flatten_block_xy_loop = cuda_flatten_block_loop<named_dim::x, named_dim::y>;
using cuda_flatten_block_xz_loop = cuda_flatten_block_loop<named_dim::x, named_dim::z>;
using cuda_flatten_block_yx_loop = cuda_flatten_block_loop<named_dim::y, named_dim::x>;
using cuda_flatten_block_yz_loop = cuda_flatten_block_loop<named_dim::y, named_dim::z>;
using cuda_flatten_block_zx_loop = cuda_flatten_block_loop<named_dim::z, named_dim::x>;
using cuda_flatten_block_zy_loop = cuda_flatten_block_loop<named_dim::z, named_dim::y>;

using cuda_flatten_block_xyz_loop = cuda_flatten_block_loop<named_dim::x, named_dim::y, named_dim::z>;
using cuda_flatten_block_xzy_loop = cuda_flatten_block_loop<named_dim::x, named_dim::z, named_dim::y>;
using cuda_flatten_block_yxz_loop = cuda_flatten_block_loop<named_dim::y, named_dim::x, named_dim::z>;
using cuda_flatten_block_yzx_loop = cuda_flatten_block_loop<named_dim::y, named_dim::z, named_dim::x>;
using cuda_flatten_block_zxy_loop = cuda_flatten_block_loop<named_dim::z, named_dim::x, named_dim::y>;
using cuda_flatten_block_zyx_loop = cuda_flatten_block_loop<named_dim::z, named_dim::y, named_dim::x>;


/*!
 * Maps segment indices to CUDA global threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 */
template < named_dim ... dims >
using cuda_global_direct = cuda_indexer_direct<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::unspecified>...>;

using cuda_global_x_direct = cuda_global_direct<named_dim::x>;
using cuda_global_y_direct = cuda_global_direct<named_dim::y>;
using cuda_global_z_direct = cuda_global_direct<named_dim::z>;

using cuda_global_xy_direct = cuda_global_direct<named_dim::x, named_dim::y>;
using cuda_global_xz_direct = cuda_global_direct<named_dim::x, named_dim::z>;
using cuda_global_yx_direct = cuda_global_direct<named_dim::y, named_dim::x>;
using cuda_global_yz_direct = cuda_global_direct<named_dim::y, named_dim::z>;
using cuda_global_zx_direct = cuda_global_direct<named_dim::z, named_dim::x>;
using cuda_global_zy_direct = cuda_global_direct<named_dim::z, named_dim::y>;

using cuda_global_xyz_direct = cuda_global_direct<named_dim::x, named_dim::y, named_dim::z>;
using cuda_global_xzy_direct = cuda_global_direct<named_dim::x, named_dim::z, named_dim::y>;
using cuda_global_yxz_direct = cuda_global_direct<named_dim::y, named_dim::x, named_dim::z>;
using cuda_global_yzx_direct = cuda_global_direct<named_dim::y, named_dim::z, named_dim::x>;
using cuda_global_zxy_direct = cuda_global_direct<named_dim::z, named_dim::x, named_dim::y>;
using cuda_global_zyx_direct = cuda_global_direct<named_dim::z, named_dim::y, named_dim::x>;

/*!
 * Maps segment indices to CUDA global threads.
 * Uses grid-stride looping to exceed the maximum number of global threads
 */
template < named_dim ... dims >
using cuda_global_loop = cuda_indexer_loop<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::unspecified>...>;

template < named_dim ... dims >
using cuda_global_syncable_loop = cuda_indexer_syncable_loop<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::unspecified>...>;

using cuda_global_x_loop = cuda_global_loop<named_dim::x>;
using cuda_global_y_loop = cuda_global_loop<named_dim::y>;
using cuda_global_z_loop = cuda_global_loop<named_dim::z>;

using cuda_global_xy_loop = cuda_global_loop<named_dim::x, named_dim::y>;
using cuda_global_xz_loop = cuda_global_loop<named_dim::x, named_dim::z>;
using cuda_global_yx_loop = cuda_global_loop<named_dim::y, named_dim::x>;
using cuda_global_yz_loop = cuda_global_loop<named_dim::y, named_dim::z>;
using cuda_global_zx_loop = cuda_global_loop<named_dim::z, named_dim::x>;
using cuda_global_zy_loop = cuda_global_loop<named_dim::z, named_dim::y>;

using cuda_global_xyz_loop = cuda_global_loop<named_dim::x, named_dim::y, named_dim::z>;
using cuda_global_xzy_loop = cuda_global_loop<named_dim::x, named_dim::z, named_dim::y>;
using cuda_global_yxz_loop = cuda_global_loop<named_dim::y, named_dim::x, named_dim::z>;
using cuda_global_yzx_loop = cuda_global_loop<named_dim::y, named_dim::z, named_dim::x>;
using cuda_global_zxy_loop = cuda_global_loop<named_dim::z, named_dim::x, named_dim::y>;
using cuda_global_zyx_loop = cuda_global_loop<named_dim::z, named_dim::y, named_dim::x>;

/*
 * Maps segment indices to flattened CUDA global threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical global threads to fit all of the direct map requests.
 * Reshapes multiple physical global threads into a 1D iteration space
 */
template < named_dim ... dims >
using cuda_flatten_global_direct = cuda_flatten_indexer_direct<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::unspecified>...>;

using cuda_flatten_global_x_direct = cuda_flatten_global_direct<named_dim::x>;
using cuda_flatten_global_y_direct = cuda_flatten_global_direct<named_dim::y>;
using cuda_flatten_global_z_direct = cuda_flatten_global_direct<named_dim::z>;

using cuda_flatten_global_xy_direct = cuda_flatten_global_direct<named_dim::x, named_dim::y>;
using cuda_flatten_global_xz_direct = cuda_flatten_global_direct<named_dim::x, named_dim::z>;
using cuda_flatten_global_yx_direct = cuda_flatten_global_direct<named_dim::y, named_dim::x>;
using cuda_flatten_global_yz_direct = cuda_flatten_global_direct<named_dim::y, named_dim::z>;
using cuda_flatten_global_zx_direct = cuda_flatten_global_direct<named_dim::z, named_dim::x>;
using cuda_flatten_global_zy_direct = cuda_flatten_global_direct<named_dim::z, named_dim::y>;

using cuda_flatten_global_xyz_direct = cuda_flatten_global_direct<named_dim::x, named_dim::y, named_dim::z>;
using cuda_flatten_global_xzy_direct = cuda_flatten_global_direct<named_dim::x, named_dim::z, named_dim::y>;
using cuda_flatten_global_yxz_direct = cuda_flatten_global_direct<named_dim::y, named_dim::x, named_dim::z>;
using cuda_flatten_global_yzx_direct = cuda_flatten_global_direct<named_dim::y, named_dim::z, named_dim::x>;
using cuda_flatten_global_zxy_direct = cuda_flatten_global_direct<named_dim::z, named_dim::x, named_dim::y>;
using cuda_flatten_global_zyx_direct = cuda_flatten_global_direct<named_dim::z, named_dim::y, named_dim::x>;

/*
 * Maps segment indices to flattened CUDA global threads.
 * Reshapes multiple physical global threads into a 1D iteration space
 * Uses global thread-stride looping to exceed the maximum number of physical global threads
 */
template < named_dim ... dims >
using cuda_flatten_global_loop = cuda_flatten_indexer_loop<
    cuda::IndexGlobal<dims, named_usage::unspecified, named_usage::unspecified>...>;

using cuda_flatten_global_x_loop = cuda_flatten_global_loop<named_dim::x>;
using cuda_flatten_global_y_loop = cuda_flatten_global_loop<named_dim::y>;
using cuda_flatten_global_z_loop = cuda_flatten_global_loop<named_dim::z>;

using cuda_flatten_global_xy_loop = cuda_flatten_global_loop<named_dim::x, named_dim::y>;
using cuda_flatten_global_xz_loop = cuda_flatten_global_loop<named_dim::x, named_dim::z>;
using cuda_flatten_global_yx_loop = cuda_flatten_global_loop<named_dim::y, named_dim::x>;
using cuda_flatten_global_yz_loop = cuda_flatten_global_loop<named_dim::y, named_dim::z>;
using cuda_flatten_global_zx_loop = cuda_flatten_global_loop<named_dim::z, named_dim::x>;
using cuda_flatten_global_zy_loop = cuda_flatten_global_loop<named_dim::z, named_dim::y>;

using cuda_flatten_global_xyz_loop = cuda_flatten_global_loop<named_dim::x, named_dim::y, named_dim::z>;
using cuda_flatten_global_xzy_loop = cuda_flatten_global_loop<named_dim::x, named_dim::z, named_dim::y>;
using cuda_flatten_global_yxz_loop = cuda_flatten_global_loop<named_dim::y, named_dim::x, named_dim::z>;
using cuda_flatten_global_yzx_loop = cuda_flatten_global_loop<named_dim::y, named_dim::z, named_dim::x>;
using cuda_flatten_global_zxy_loop = cuda_flatten_global_loop<named_dim::z, named_dim::x, named_dim::y>;
using cuda_flatten_global_zyx_loop = cuda_flatten_global_loop<named_dim::z, named_dim::y, named_dim::x>;


/*!
 * Maps segment indices to CUDA global threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical threads to fit all of the direct map requests.
 */
template < int X_BLOCK_SIZE >
using cuda_thread_size_x_direct = cuda_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE >
using cuda_thread_size_y_direct = cuda_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE >
using cuda_thread_size_z_direct = cuda_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_thread_size_xy_direct = cuda_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_thread_size_xz_direct = cuda_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_thread_size_yx_direct = cuda_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_thread_size_yz_direct = cuda_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_thread_size_zx_direct = cuda_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_thread_size_zy_direct = cuda_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_thread_size_xyz_direct = cuda_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_thread_size_xzy_direct = cuda_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_thread_size_yxz_direct = cuda_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_thread_size_yzx_direct = cuda_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_thread_size_zxy_direct = cuda_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_thread_size_zyx_direct = cuda_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;


template < int X_GRID_SIZE >
using cuda_block_size_x_direct = cuda_indexer_direct<cuda::block_x<X_GRID_SIZE>>;
template < int Y_GRID_SIZE >
using cuda_block_size_y_direct = cuda_indexer_direct<cuda::block_y<Y_GRID_SIZE>>;
template < int Z_GRID_SIZE >
using cuda_block_size_z_direct = cuda_indexer_direct<cuda::block_z<Z_GRID_SIZE>>;

template < int X_GRID_SIZE, int Y_GRID_SIZE >
using cuda_block_size_xy_direct = cuda_indexer_direct<cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int X_GRID_SIZE, int Z_GRID_SIZE >
using cuda_block_size_xz_direct = cuda_indexer_direct<cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Y_GRID_SIZE, int X_GRID_SIZE >
using cuda_block_size_yx_direct = cuda_indexer_direct<cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Y_GRID_SIZE, int Z_GRID_SIZE >
using cuda_block_size_yz_direct = cuda_indexer_direct<cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Z_GRID_SIZE, int X_GRID_SIZE >
using cuda_block_size_zx_direct = cuda_indexer_direct<cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Z_GRID_SIZE, int Y_GRID_SIZE >
using cuda_block_size_zy_direct = cuda_indexer_direct<cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;

template < int X_GRID_SIZE, int Y_GRID_SIZE, int Z_GRID_SIZE >
using cuda_block_size_xyz_direct = cuda_indexer_direct<cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int X_GRID_SIZE, int Z_GRID_SIZE, int Y_GRID_SIZE >
using cuda_block_size_xzy_direct = cuda_indexer_direct<cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int Y_GRID_SIZE, int X_GRID_SIZE, int Z_GRID_SIZE >
using cuda_block_size_yxz_direct = cuda_indexer_direct<cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Y_GRID_SIZE, int Z_GRID_SIZE, int X_GRID_SIZE >
using cuda_block_size_yzx_direct = cuda_indexer_direct<cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Z_GRID_SIZE, int X_GRID_SIZE, int Y_GRID_SIZE >
using cuda_block_size_zxy_direct = cuda_indexer_direct<cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int Z_GRID_SIZE, int Y_GRID_SIZE, int X_GRID_SIZE >
using cuda_block_size_zyx_direct = cuda_indexer_direct<cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;


template < int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_x_direct = cuda_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_y_direct = cuda_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_z_direct = cuda_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_xy_direct = cuda_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                     cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_xz_direct = cuda_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                     cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_yx_direct = cuda_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                     cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_yz_direct = cuda_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                     cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_zx_direct = cuda_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                     cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_zy_direct = cuda_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                     cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_xyz_direct = cuda_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_xzy_direct = cuda_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_yxz_direct = cuda_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_yzx_direct = cuda_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_zxy_direct = cuda_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_zyx_direct = cuda_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;

/*!
 * Maps segment indices to CUDA global threads.
 * Uses grid-stride looping to exceed the maximum number of global threads
 */
template < int X_BLOCK_SIZE >
using cuda_thread_size_x_loop = cuda_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE >
using cuda_thread_size_y_loop = cuda_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE >
using cuda_thread_size_z_loop = cuda_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_thread_size_xy_loop = cuda_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_thread_size_xz_loop = cuda_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_thread_size_yx_loop = cuda_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_thread_size_yz_loop = cuda_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_thread_size_zx_loop = cuda_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_thread_size_zy_loop = cuda_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_thread_size_xyz_loop = cuda_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_thread_size_xzy_loop = cuda_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_thread_size_yxz_loop = cuda_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_thread_size_yzx_loop = cuda_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_thread_size_zxy_loop = cuda_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_thread_size_zyx_loop = cuda_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;


template < int X_GRID_SIZE >
using cuda_block_size_x_loop = cuda_indexer_loop<cuda::block_x<X_GRID_SIZE>>;
template < int Y_GRID_SIZE >
using cuda_block_size_y_loop = cuda_indexer_loop<cuda::block_y<Y_GRID_SIZE>>;
template < int Z_GRID_SIZE >
using cuda_block_size_z_loop = cuda_indexer_loop<cuda::block_z<Z_GRID_SIZE>>;

template < int X_GRID_SIZE, int Y_GRID_SIZE >
using cuda_block_size_xy_loop = cuda_indexer_loop<cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int X_GRID_SIZE, int Z_GRID_SIZE >
using cuda_block_size_xz_loop = cuda_indexer_loop<cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Y_GRID_SIZE, int X_GRID_SIZE >
using cuda_block_size_yx_loop = cuda_indexer_loop<cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Y_GRID_SIZE, int Z_GRID_SIZE >
using cuda_block_size_yz_loop = cuda_indexer_loop<cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Z_GRID_SIZE, int X_GRID_SIZE >
using cuda_block_size_zx_loop = cuda_indexer_loop<cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Z_GRID_SIZE, int Y_GRID_SIZE >
using cuda_block_size_zy_loop = cuda_indexer_loop<cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;

template < int X_GRID_SIZE, int Y_GRID_SIZE, int Z_GRID_SIZE >
using cuda_block_size_xyz_loop = cuda_indexer_loop<cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int X_GRID_SIZE, int Z_GRID_SIZE, int Y_GRID_SIZE >
using cuda_block_size_xzy_loop = cuda_indexer_loop<cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int Y_GRID_SIZE, int X_GRID_SIZE, int Z_GRID_SIZE >
using cuda_block_size_yxz_loop = cuda_indexer_loop<cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Y_GRID_SIZE, int Z_GRID_SIZE, int X_GRID_SIZE >
using cuda_block_size_yzx_loop = cuda_indexer_loop<cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Z_GRID_SIZE, int X_GRID_SIZE, int Y_GRID_SIZE >
using cuda_block_size_zxy_loop = cuda_indexer_loop<cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int Z_GRID_SIZE, int Y_GRID_SIZE, int X_GRID_SIZE >
using cuda_block_size_zyx_loop = cuda_indexer_loop<cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;


template < int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_x_loop = cuda_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_y_loop = cuda_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_z_loop = cuda_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_xy_loop = cuda_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                     cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_xz_loop = cuda_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                     cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_yx_loop = cuda_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                     cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_yz_loop = cuda_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                     cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_zx_loop = cuda_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                     cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_zy_loop = cuda_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                     cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_xyz_loop = cuda_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_xzy_loop = cuda_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_yxz_loop = cuda_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_yzx_loop = cuda_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_zxy_loop = cuda_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_global_size_zyx_loop = cuda_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;

/*
 * Maps segment indices to flattened CUDA global threads.
 * This is the lowest overhead mapping, but requires that there are enough
 * physical global threads to fit all of the direct map requests.
 * Reshapes multiple physical global threads into a 1D iteration space
 */
template < int X_BLOCK_SIZE >
using cuda_flatten_thread_size_x_direct = cuda_flatten_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_y_direct = cuda_flatten_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_z_direct = cuda_flatten_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_xy_direct = cuda_flatten_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_xz_direct = cuda_flatten_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_flatten_thread_size_yx_direct = cuda_flatten_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_yz_direct = cuda_flatten_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_flatten_thread_size_zx_direct = cuda_flatten_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_zy_direct = cuda_flatten_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_xyz_direct = cuda_flatten_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_xzy_direct = cuda_flatten_indexer_direct<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_yxz_direct = cuda_flatten_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_flatten_thread_size_yzx_direct = cuda_flatten_indexer_direct<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_zxy_direct = cuda_flatten_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_flatten_thread_size_zyx_direct = cuda_flatten_indexer_direct<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;


template < int X_GRID_SIZE >
using cuda_flatten_block_size_x_direct = cuda_flatten_indexer_direct<cuda::block_x<X_GRID_SIZE>>;
template < int Y_GRID_SIZE >
using cuda_flatten_block_size_y_direct = cuda_flatten_indexer_direct<cuda::block_y<Y_GRID_SIZE>>;
template < int Z_GRID_SIZE >
using cuda_flatten_block_size_z_direct = cuda_flatten_indexer_direct<cuda::block_z<Z_GRID_SIZE>>;

template < int X_GRID_SIZE, int Y_GRID_SIZE >
using cuda_flatten_block_size_xy_direct = cuda_flatten_indexer_direct<cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int X_GRID_SIZE, int Z_GRID_SIZE >
using cuda_flatten_block_size_xz_direct = cuda_flatten_indexer_direct<cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Y_GRID_SIZE, int X_GRID_SIZE >
using cuda_flatten_block_size_yx_direct = cuda_flatten_indexer_direct<cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Y_GRID_SIZE, int Z_GRID_SIZE >
using cuda_flatten_block_size_yz_direct = cuda_flatten_indexer_direct<cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Z_GRID_SIZE, int X_GRID_SIZE >
using cuda_flatten_block_size_zx_direct = cuda_flatten_indexer_direct<cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Z_GRID_SIZE, int Y_GRID_SIZE >
using cuda_flatten_block_size_zy_direct = cuda_flatten_indexer_direct<cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;

template < int X_GRID_SIZE, int Y_GRID_SIZE, int Z_GRID_SIZE >
using cuda_flatten_block_size_xyz_direct = cuda_flatten_indexer_direct<cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int X_GRID_SIZE, int Z_GRID_SIZE, int Y_GRID_SIZE >
using cuda_flatten_block_size_xzy_direct = cuda_flatten_indexer_direct<cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int Y_GRID_SIZE, int X_GRID_SIZE, int Z_GRID_SIZE >
using cuda_flatten_block_size_yxz_direct = cuda_flatten_indexer_direct<cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Y_GRID_SIZE, int Z_GRID_SIZE, int X_GRID_SIZE >
using cuda_flatten_block_size_yzx_direct = cuda_flatten_indexer_direct<cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Z_GRID_SIZE, int X_GRID_SIZE, int Y_GRID_SIZE >
using cuda_flatten_block_size_zxy_direct = cuda_flatten_indexer_direct<cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int Z_GRID_SIZE, int Y_GRID_SIZE, int X_GRID_SIZE >
using cuda_flatten_block_size_zyx_direct = cuda_flatten_indexer_direct<cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;


template < int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_x_direct = cuda_flatten_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_y_direct = cuda_flatten_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_z_direct = cuda_flatten_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_xy_direct = cuda_flatten_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                     cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_xz_direct = cuda_flatten_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                     cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_yx_direct = cuda_flatten_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                     cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_yz_direct = cuda_flatten_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                     cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_zx_direct = cuda_flatten_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                     cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_zy_direct = cuda_flatten_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                     cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_xyz_direct = cuda_flatten_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_xzy_direct = cuda_flatten_indexer_direct<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_yxz_direct = cuda_flatten_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_yzx_direct = cuda_flatten_indexer_direct<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                      cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_zxy_direct = cuda_flatten_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_zyx_direct = cuda_flatten_indexer_direct<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                      cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                      cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;

/*
 * Maps segment indices to flattened CUDA global threads.
 * Reshapes multiple physical global threads into a 1D iteration space
 * Uses global thread-stride looping to exceed the maximum number of physical global threads
 */
template < int X_BLOCK_SIZE >
using cuda_flatten_thread_size_x_loop = cuda_flatten_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_y_loop = cuda_flatten_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_z_loop = cuda_flatten_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_xy_loop = cuda_flatten_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_xz_loop = cuda_flatten_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_flatten_thread_size_yx_loop = cuda_flatten_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_yz_loop = cuda_flatten_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_flatten_thread_size_zx_loop = cuda_flatten_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_zy_loop = cuda_flatten_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_xyz_loop = cuda_flatten_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_xzy_loop = cuda_flatten_indexer_loop<cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE, int Z_BLOCK_SIZE >
using cuda_flatten_thread_size_yxz_loop = cuda_flatten_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_flatten_thread_size_yzx_loop = cuda_flatten_indexer_loop<cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE, int Y_BLOCK_SIZE >
using cuda_flatten_thread_size_zxy_loop = cuda_flatten_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE, int X_BLOCK_SIZE >
using cuda_flatten_thread_size_zyx_loop = cuda_flatten_indexer_loop<cuda::thread_z<Z_BLOCK_SIZE>, cuda::thread_y<Y_BLOCK_SIZE>, cuda::thread_x<X_BLOCK_SIZE>>;


template < int X_GRID_SIZE >
using cuda_flatten_block_size_x_loop = cuda_flatten_indexer_loop<cuda::block_x<X_GRID_SIZE>>;
template < int Y_GRID_SIZE >
using cuda_flatten_block_size_y_loop = cuda_flatten_indexer_loop<cuda::block_y<Y_GRID_SIZE>>;
template < int Z_GRID_SIZE >
using cuda_flatten_block_size_z_loop = cuda_flatten_indexer_loop<cuda::block_z<Z_GRID_SIZE>>;

template < int X_GRID_SIZE, int Y_GRID_SIZE >
using cuda_flatten_block_size_xy_loop = cuda_flatten_indexer_loop<cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int X_GRID_SIZE, int Z_GRID_SIZE >
using cuda_flatten_block_size_xz_loop = cuda_flatten_indexer_loop<cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Y_GRID_SIZE, int X_GRID_SIZE >
using cuda_flatten_block_size_yx_loop = cuda_flatten_indexer_loop<cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Y_GRID_SIZE, int Z_GRID_SIZE >
using cuda_flatten_block_size_yz_loop = cuda_flatten_indexer_loop<cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Z_GRID_SIZE, int X_GRID_SIZE >
using cuda_flatten_block_size_zx_loop = cuda_flatten_indexer_loop<cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Z_GRID_SIZE, int Y_GRID_SIZE >
using cuda_flatten_block_size_zy_loop = cuda_flatten_indexer_loop<cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;

template < int X_GRID_SIZE, int Y_GRID_SIZE, int Z_GRID_SIZE >
using cuda_flatten_block_size_xyz_loop = cuda_flatten_indexer_loop<cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int X_GRID_SIZE, int Z_GRID_SIZE, int Y_GRID_SIZE >
using cuda_flatten_block_size_xzy_loop = cuda_flatten_indexer_loop<cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int Y_GRID_SIZE, int X_GRID_SIZE, int Z_GRID_SIZE >
using cuda_flatten_block_size_yxz_loop = cuda_flatten_indexer_loop<cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>>;
template < int Y_GRID_SIZE, int Z_GRID_SIZE, int X_GRID_SIZE >
using cuda_flatten_block_size_yzx_loop = cuda_flatten_indexer_loop<cuda::block_y<Y_GRID_SIZE>, cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;
template < int Z_GRID_SIZE, int X_GRID_SIZE, int Y_GRID_SIZE >
using cuda_flatten_block_size_zxy_loop = cuda_flatten_indexer_loop<cuda::block_z<Z_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>>;
template < int Z_GRID_SIZE, int Y_GRID_SIZE, int X_GRID_SIZE >
using cuda_flatten_block_size_zyx_loop = cuda_flatten_indexer_loop<cuda::block_z<Z_GRID_SIZE>, cuda::block_y<Y_GRID_SIZE>, cuda::block_x<X_GRID_SIZE>>;


template < int X_BLOCK_SIZE, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_x_loop = cuda_flatten_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_y_loop = cuda_flatten_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_z_loop = cuda_flatten_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_xy_loop = cuda_flatten_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                 cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_xz_loop = cuda_flatten_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                 cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_yx_loop = cuda_flatten_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                 cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_yz_loop = cuda_flatten_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                 cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_zx_loop = cuda_flatten_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                 cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_zy_loop = cuda_flatten_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                 cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;

template < int X_BLOCK_SIZE, int Y_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_xyz_loop = cuda_flatten_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                  cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                  cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int X_BLOCK_SIZE, int Z_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_xzy_loop = cuda_flatten_indexer_loop<cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                  cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                  cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int X_BLOCK_SIZE, int Z_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_yxz_loop = cuda_flatten_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                  cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                  cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>>;
template < int Y_BLOCK_SIZE, int Z_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Y_GRID_SIZE = named_usage::unspecified, int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_yzx_loop = cuda_flatten_indexer_loop<cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                  cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                  cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int X_BLOCK_SIZE, int Y_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_zxy_loop = cuda_flatten_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                  cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>,
                                                                  cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>>;
template < int Z_BLOCK_SIZE, int Y_BLOCK_SIZE, int X_BLOCK_SIZE,
           int Z_GRID_SIZE = named_usage::unspecified, int Y_GRID_SIZE = named_usage::unspecified, int X_GRID_SIZE = named_usage::unspecified >
using cuda_flatten_global_size_zyx_loop = cuda_flatten_indexer_loop<cuda::global_z<Z_BLOCK_SIZE, Z_GRID_SIZE>,
                                                                  cuda::global_y<Y_BLOCK_SIZE, Y_GRID_SIZE>,
                                                                  cuda::global_x<X_BLOCK_SIZE, X_GRID_SIZE>>;


/*
 * Deprecated policies
 */
using cuda_global_thread_x = cuda_global_x_direct;
using cuda_global_thread_y = cuda_global_y_direct;
using cuda_global_thread_z = cuda_global_z_direct;

using cuda_global_thread_xy = cuda_global_xy_direct;
using cuda_global_thread_xz = cuda_global_xz_direct;
using cuda_global_thread_yx = cuda_global_yx_direct;
using cuda_global_thread_yz = cuda_global_yz_direct;
using cuda_global_thread_zx = cuda_global_zx_direct;
using cuda_global_thread_zy = cuda_global_zy_direct;

using cuda_global_thread_xyz = cuda_global_xyz_direct;
using cuda_global_thread_xzy = cuda_global_xzy_direct;
using cuda_global_thread_yxz = cuda_global_yxz_direct;
using cuda_global_thread_yzx = cuda_global_yzx_direct;
using cuda_global_thread_zxy = cuda_global_zxy_direct;
using cuda_global_thread_zyx = cuda_global_zyx_direct;

using cuda_flatten_block_threads_xy_direct = cuda_flatten_thread_xy_direct;
using cuda_flatten_block_threads_xz_direct = cuda_flatten_thread_xz_direct;
using cuda_flatten_block_threads_yx_direct = cuda_flatten_thread_yx_direct;
using cuda_flatten_block_threads_yz_direct = cuda_flatten_thread_yz_direct;
using cuda_flatten_block_threads_zx_direct = cuda_flatten_thread_zx_direct;
using cuda_flatten_block_threads_zy_direct = cuda_flatten_thread_zy_direct;

using cuda_flatten_block_threads_xyz_direct = cuda_flatten_thread_xyz_direct;
using cuda_flatten_block_threads_xzy_direct = cuda_flatten_thread_xzy_direct;
using cuda_flatten_block_threads_yxz_direct = cuda_flatten_thread_yxz_direct;
using cuda_flatten_block_threads_yzx_direct = cuda_flatten_thread_yzx_direct;
using cuda_flatten_block_threads_zxy_direct = cuda_flatten_thread_zxy_direct;
using cuda_flatten_block_threads_zyx_direct = cuda_flatten_thread_zyx_direct;

using cuda_flatten_block_threads_xy_loop = cuda_flatten_thread_xy_loop;
using cuda_flatten_block_threads_xz_loop = cuda_flatten_thread_xz_loop;
using cuda_flatten_block_threads_yx_loop = cuda_flatten_thread_yx_loop;
using cuda_flatten_block_threads_yz_loop = cuda_flatten_thread_yz_loop;
using cuda_flatten_block_threads_zx_loop = cuda_flatten_thread_zx_loop;
using cuda_flatten_block_threads_zy_loop = cuda_flatten_thread_zy_loop;

using cuda_flatten_block_threads_xyz_loop = cuda_flatten_thread_xyz_loop;
using cuda_flatten_block_threads_xzy_loop = cuda_flatten_thread_xzy_loop;
using cuda_flatten_block_threads_yxz_loop = cuda_flatten_thread_yxz_loop;
using cuda_flatten_block_threads_yzx_loop = cuda_flatten_thread_yzx_loop;
using cuda_flatten_block_threads_zxy_loop = cuda_flatten_thread_zxy_loop;
using cuda_flatten_block_threads_zyx_loop = cuda_flatten_thread_zyx_loop;

using cuda_block_xy_nested_direct = cuda_block_xy_direct;
using cuda_block_xz_nested_direct = cuda_block_xz_direct;
using cuda_block_yx_nested_direct = cuda_block_yx_direct;
using cuda_block_yz_nested_direct = cuda_block_yz_direct;
using cuda_block_zx_nested_direct = cuda_block_zx_direct;
using cuda_block_zy_nested_direct = cuda_block_zy_direct;

using cuda_block_xyz_nested_direct = cuda_block_xyz_direct;
using cuda_block_xzy_nested_direct = cuda_block_xzy_direct;
using cuda_block_yxz_nested_direct = cuda_block_yxz_direct;
using cuda_block_yzx_nested_direct = cuda_block_yzx_direct;
using cuda_block_zxy_nested_direct = cuda_block_zxy_direct;
using cuda_block_zyx_nested_direct = cuda_block_zyx_direct;

using cuda_block_xy_nested_loop = cuda_block_xy_loop;
using cuda_block_xz_nested_loop = cuda_block_xz_loop;
using cuda_block_yx_nested_loop = cuda_block_yx_loop;
using cuda_block_yz_nested_loop = cuda_block_yz_loop;
using cuda_block_zx_nested_loop = cuda_block_zx_loop;
using cuda_block_zy_nested_loop = cuda_block_zy_loop;

using cuda_block_xyz_nested_loop = cuda_block_xyz_loop;
using cuda_block_xzy_nested_loop = cuda_block_xzy_loop;
using cuda_block_yxz_nested_loop = cuda_block_yxz_loop;
using cuda_block_yzx_nested_loop = cuda_block_yzx_loop;
using cuda_block_zxy_nested_loop = cuda_block_zxy_loop;
using cuda_block_zyx_nested_loop = cuda_block_zyx_loop;

}  // namespace RAJA

#endif  // RAJA_ENABLE_CUDA
#endif
