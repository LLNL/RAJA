/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via HIP kernel launch.
 *
 *          These methods should work on any platform that supports
 *          HIP devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_hip_HPP
#define RAJA_forall_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <algorithm>
#include "hip/hip_runtime.h"

#include "RAJA/pattern/forall.hpp"

#include "RAJA/pattern/params/forall.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"

#include "RAJA/index/IndexSet.hpp"

#include "RAJA/util/resource.hpp"

namespace RAJA
{
namespace policy
{
namespace hip
{

namespace impl
{

/*!
 ******************************************************************************
 *
 * \brief  Hip kernel block and grid dimension calculator template.
 *
 * \tparam IterationMapping Way of mapping from threads in the kernel to
 *         iterates of the forall loop. For example StridedLoop uses a grid
 *         stride loop to run multiple iterates in a single thread.
 * \tparam IterationGetter Way of getting iteration indices from the underlying
 *         runtime using threadIdx, blockIdx, etc.
 * \tparam UniqueMarker Used in occupancy calculator methods to store and get
 *         data for this specific kernel.
 *
 ******************************************************************************
 */
template<typename IterationMapping, typename IterationGetter, typename Concretizer, typename UniqueMarker>
struct ForallDimensionCalculator;

// The general cases handle fixed BLOCK_SIZE > 0 and/or GRID_SIZE > 0
// there are specializations for named_usage::unspecified
// but named_usage::ignored is not supported so no specializations are provided
// and static_asserts in the general case catch unsupported values
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE, typename Concretizer, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::Direct,
                                 ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>,
                                 Concretizer,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0 or named_usage::unspecified with forall");
  static_assert(GRID_SIZE > 0, "grid size must be > 0 or named_usage::unspecified with forall");

  using IndexGetter = ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT len,
                             const void* RAJA_UNUSED_ARG(func), size_t RAJA_UNUSED_ARG(dynamic_shmem_size))
  {
    const IdxT block_size = static_cast<IdxT>(IndexGetter::block_size);
    const IdxT grid_size = static_cast<IdxT>(IndexGetter::grid_size);

    if ( len > (block_size * grid_size) ) {
      RAJA_ABORT_OR_THROW("len exceeds the size of the directly mapped index space");
    }

    internal::set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexGetter::block_size));
    internal::set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexGetter::grid_size));
  }
};

template<named_dim dim, int GRID_SIZE, typename Concretizer, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::Direct,
                                 ::RAJA::hip::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>,
                                 Concretizer,
                                 UniqueMarker>
{
  static_assert(GRID_SIZE > 0, "grid size must be > 0 or named_usage::unspecified with forall");

  using IndexGetter = ::RAJA::hip::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::hip::ConcretizerImpl<IdxT, Concretizer, UniqueMarker> concretizer{func, dynamic_shmem_size, len};

    const IdxT grid_size = static_cast<IdxT>(IndexGetter::grid_size);
    const IdxT block_size = concretizer.get_block_size_to_fit_len(grid_size);

    if ( block_size == IdxT(0) ) {
      RAJA_ABORT_OR_THROW("len exceeds the size of the directly mapped index space");
    }

    internal::set_hip_dim<dim>(dims.threads, block_size);
    internal::set_hip_dim<dim>(dims.blocks, grid_size);
  }
};

template<named_dim dim, int BLOCK_SIZE, typename Concretizer, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::Direct,
                                 ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>,
                                 Concretizer,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0 or named_usage::unspecified with forall");

  using IndexGetter = ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::hip::ConcretizerImpl<IdxT, Concretizer, UniqueMarker> concretizer{func, dynamic_shmem_size, len};

    const IdxT block_size = static_cast<IdxT>(IndexGetter::block_size);
    const IdxT grid_size = concretizer.get_grid_size_to_fit_len(block_size);

    internal::set_hip_dim<dim>(dims.threads, block_size);
    internal::set_hip_dim<dim>(dims.blocks, grid_size);
  }
};

template<named_dim dim, typename Concretizer, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::Direct,
                                 ::RAJA::hip::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>,
                                 Concretizer,
                                 UniqueMarker>
{
  using IndexGetter = ::RAJA::hip::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::hip::ConcretizerImpl<IdxT, Concretizer, UniqueMarker> concretizer{func, dynamic_shmem_size, len};

    const auto sizes = concretizer.get_block_and_grid_size_to_fit_len();

    internal::set_hip_dim<dim>(dims.threads, sizes.first);
    internal::set_hip_dim<dim>(dims.blocks, sizes.second);
  }
};

template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE, typename Concretizer, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
                                 ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>,
                                 Concretizer,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0 or named_usage::unspecified with forall");
  static_assert(GRID_SIZE > 0, "grid size must be > 0 or named_usage::unspecified with forall");

  using IndexGetter = ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT RAJA_UNUSED_ARG(len),
                             const void* RAJA_UNUSED_ARG(func), size_t RAJA_UNUSED_ARG(dynamic_shmem_size))
  {
    const IdxT block_size = static_cast<IdxT>(IndexGetter::block_size);
    const IdxT grid_size = static_cast<IdxT>(IndexGetter::grid_size);

    internal::set_hip_dim<dim>(dims.threads, block_size);
    internal::set_hip_dim<dim>(dims.blocks, grid_size);
  }
};

template<named_dim dim, int GRID_SIZE, typename Concretizer, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
                                 ::RAJA::hip::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>,
                                 Concretizer,
                                 UniqueMarker>
{
  static_assert(GRID_SIZE > 0, "grid size must be > 0 or named_usage::unspecified with forall");

  using IndexGetter = ::RAJA::hip::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::hip::ConcretizerImpl<IdxT, Concretizer, UniqueMarker> concretizer{func, dynamic_shmem_size, len};

    const IdxT grid_size = static_cast<IdxT>(IndexGetter::grid_size);
    const IdxT block_size = concretizer.get_block_size_to_fit_device(grid_size);

    internal::set_hip_dim<dim>(dims.threads, block_size);
    internal::set_hip_dim<dim>(dims.blocks, grid_size);
  }
};

template<named_dim dim, int BLOCK_SIZE, typename Concretizer, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
                                 ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>,
                                 Concretizer,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0 or named_usage::unspecified with forall");

  using IndexGetter = ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::hip::ConcretizerImpl<IdxT, Concretizer, UniqueMarker> concretizer{func, dynamic_shmem_size, len};

    const IdxT block_size = static_cast<IdxT>(IndexGetter::block_size);
    const IdxT grid_size = concretizer.get_grid_size_to_fit_device(block_size);

    internal::set_hip_dim<dim>(dims.threads, block_size);
    internal::set_hip_dim<dim>(dims.blocks, grid_size);
  }
};

template<named_dim dim, typename Concretizer, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::StridedLoop<named_usage::unspecified>,
                                 ::RAJA::hip::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>,
                                 Concretizer,
                                 UniqueMarker>
{
  using IndexGetter = ::RAJA::hip::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::hip::ConcretizerImpl<IdxT, Concretizer, UniqueMarker> concretizer{func, dynamic_shmem_size, len};

    const auto sizes = concretizer.get_block_and_grid_size_to_fit_device();

    internal::set_hip_dim<dim>(dims.threads, sizes.first);
    internal::set_hip_dim<dim>(dims.blocks, sizes.second);
  }
};

//
//////////////////////////////////////////////////////////////////////
//
// HIP kernel templates.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  HIP kernel forall template.
 *
 ******************************************************************************
 */
template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_base_of<iteration_mapping::DirectBase, IterationMapping>::value &&
                (IterationGetter::block_size > 0),
              size_t > BlockSize = IterationGetter::block_size>
__launch_bounds__(BlockSize, 1) __global__
void forall_hip_kernel(LOOP_BODY loop_body,
                       const Iterator idx,
                       IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = IterationGetter::template index<IndexType>();
  if (ii < length) {
    body(idx[ii]);
  }
}
///
template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_base_of<iteration_mapping::DirectBase, IterationMapping>::value &&
                (IterationGetter::block_size <= 0),
              size_t > RAJA_UNUSED_ARG(BlockSize) = 0>
__global__
void forall_hip_kernel(LOOP_BODY loop_body,
                       const Iterator idx,
                       IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = IterationGetter::template index<IndexType>();
  if (ii < length) {
    body(idx[ii]);
  }
}

template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_base_of<iteration_mapping::DirectBase, IterationMapping>::value &&
                (IterationGetter::block_size > 0),
              size_t > BlockSize = IterationGetter::block_size>
__launch_bounds__(BlockSize, 1) __global__
void forallp_hip_kernel(LOOP_BODY loop_body,
                        const Iterator idx,
                        IndexType length,
                        ForallParam f_params)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = IterationGetter::template index<IndexType>();
  if ( ii < length ) {
    RAJA::expt::invoke_body( f_params, body, idx[ii] );
  }
  RAJA::expt::ParamMultiplexer::combine<EXEC_POL>(f_params);
}
///
template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_base_of<iteration_mapping::DirectBase, IterationMapping>::value &&
                (IterationGetter::block_size <= 0),
              size_t > RAJA_UNUSED_ARG(BlockSize) = 0>
__global__
void forallp_hip_kernel(LOOP_BODY loop_body,
                        const Iterator idx,
                        IndexType length,
                        ForallParam f_params)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = IterationGetter::template index<IndexType>();
  if ( ii < length ) {
    RAJA::expt::invoke_body( f_params, body, idx[ii] );
  }
  RAJA::expt::ParamMultiplexer::combine<EXEC_POL>(f_params);
}

template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_base_of<iteration_mapping::StridedLoopBase, IterationMapping>::value &&
                std::is_base_of<iteration_mapping::UnsizedLoopBase, IterationMapping>::value &&
                (IterationGetter::block_size > 0),
              size_t > BlockSize = IterationGetter::block_size>
__launch_bounds__(BlockSize, 1) __global__
void forall_hip_kernel(LOOP_BODY loop_body,
                       const Iterator idx,
                       IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  for (auto ii = IterationGetter::template index<IndexType>();
       ii < length;
       ii += IterationGetter::template size<IndexType>()) {
    body(idx[ii]);
  }
}
///
template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_base_of<iteration_mapping::StridedLoopBase, IterationMapping>::value &&
                std::is_base_of<iteration_mapping::UnsizedLoopBase, IterationMapping>::value &&
                (IterationGetter::block_size <= 0),
              size_t > RAJA_UNUSED_ARG(BlockSize) = 0>
__global__
void forall_hip_kernel(LOOP_BODY loop_body,
                       const Iterator idx,
                       IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  for (auto ii = IterationGetter::template index<IndexType>();
       ii < length;
       ii += IterationGetter::template size<IndexType>()) {
    body(idx[ii]);
  }
}

///
template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_base_of<iteration_mapping::StridedLoopBase, IterationMapping>::value &&
                std::is_base_of<iteration_mapping::UnsizedLoopBase, IterationMapping>::value &&
                (IterationGetter::block_size > 0),
              size_t > BlockSize = IterationGetter::block_size>
__launch_bounds__(BlockSize, 1) __global__
void forallp_hip_kernel(LOOP_BODY loop_body,
                        const Iterator idx,
                        IndexType length,
                        ForallParam f_params)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  for (auto ii = IterationGetter::template index<IndexType>();
       ii < length;
       ii += IterationGetter::template size<IndexType>()) {
    RAJA::expt::invoke_body( f_params, body, idx[ii] );
  }
  RAJA::expt::ParamMultiplexer::combine<EXEC_POL>(f_params);
}
///
template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_base_of<iteration_mapping::StridedLoopBase, IterationMapping>::value &&
                std::is_base_of<iteration_mapping::UnsizedLoopBase, IterationMapping>::value &&
                (IterationGetter::block_size <= 0),
              size_t > RAJA_UNUSED_ARG(BlockSize) = 0>
__global__
void forallp_hip_kernel(LOOP_BODY loop_body,
                        const Iterator idx,
                        IndexType length,
                        ForallParam f_params)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  for (auto ii = IterationGetter::template index<IndexType>();
       ii < length;
       ii += IterationGetter::template size<IndexType>()) {
    RAJA::expt::invoke_body( f_params, body, idx[ii] );
  }
  RAJA::expt::ParamMultiplexer::combine<EXEC_POL>(f_params);
}

}  // namespace impl

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for HIP execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody,
          typename IterationMapping, typename IterationGetter,
          typename Concretizer, bool Async,
          typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Hip>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Hip hip_res,
            ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Concretizer, Async>const&,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Concretizer, Async>;
  using UniqueMarker = ::camp::list<IterationMapping, IterationGetter, LOOP_BODY, Iterator, ForallParam>;
  using DimensionCalculator = impl::ForallDimensionCalculator<IterationMapping, IterationGetter, Concretizer, UniqueMarker>;

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end = std::end(iter);
  IndexType len = std::distance(begin, end);

  // Only launch kernel if we have something to iterate over
  if (len > 0) {

    auto func = reinterpret_cast<const void*>(
        &impl::forall_hip_kernel<EXEC_POL, Iterator, LOOP_BODY, IndexType>);

    //
    // Setup shared memory buffers
    //
    size_t shmem = 0;

    //
    // Compute the kernel dimensions
    //
    internal::HipDims dims(1);
    DimensionCalculator::set_dimensions(dims, len, func, shmem);

    RAJA_FT_BEGIN;

    {
      //
      // Privatize the loop_body, using make_launch_body to setup reductions
      //
      LOOP_BODY body = RAJA::hip::make_launch_body(
          dims.blocks, dims.threads, shmem, hip_res, std::forward<LoopBody>(loop_body));

      //
      // Launch the kernels
      //
      void *args[] = {(void*)&body, (void*)&begin, (void*)&len};
      RAJA::hip::launch(func, dims.blocks, dims.threads, args, shmem, hip_res, Async);
    }

    RAJA_FT_END;
  }

  return resources::EventProxy<resources::Hip>(hip_res);
}


template <typename Iterable, typename LoopBody,
          typename IterationMapping, typename IterationGetter,
          typename Concretizer, bool Async,
          typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Hip>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  concepts::negate< RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>> >
forall_impl(resources::Hip hip_res,
            ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Concretizer, Async> const&,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam f_params)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Concretizer, Async>;
  using UniqueMarker = ::camp::list<IterationMapping, IterationGetter, LOOP_BODY, Iterator, ForallParam>;
  using DimensionCalculator = impl::ForallDimensionCalculator<IterationMapping, IterationGetter, Concretizer, UniqueMarker>;

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end = std::end(iter);
  IndexType len = std::distance(begin, end);

  // Only launch kernel if we have something to iterate over
  if (len > 0) {

    auto func = reinterpret_cast<const void*>(
        &impl::forallp_hip_kernel<EXEC_POL, Iterator, LOOP_BODY, IndexType, camp::decay<ForallParam>>);

    //
    // Setup shared memory buffers
    //
    size_t shmem = 0;

    //
    // Compute the kernel dimensions
    //
    internal::HipDims dims(1);
    DimensionCalculator::set_dimensions(dims, len, func, shmem);

    RAJA_FT_BEGIN;

    RAJA::hip::detail::hipInfo launch_info;
    launch_info.gridDim = dims.blocks;
    launch_info.blockDim = dims.threads;
    launch_info.res = hip_res;

    {
      RAJA::expt::ParamMultiplexer::init<EXEC_POL>(f_params, launch_info);

      //
      // Privatize the loop_body, using make_launch_body to setup reductions
      //
      LOOP_BODY body = RAJA::hip::make_launch_body(
          dims.blocks, dims.threads, shmem, hip_res, std::forward<LoopBody>(loop_body));

      //
      // Launch the kernels
      //
      void *args[] = {(void*)&body, (void*)&begin, (void*)&len, (void*)&f_params};
      RAJA::hip::launch(func, dims.blocks, dims.threads, args, shmem, hip_res, Async);

      RAJA::expt::ParamMultiplexer::resolve<EXEC_POL>(f_params, launch_info);
    }

    RAJA_FT_END;
  }

  return resources::EventProxy<resources::Hip>(hip_res);
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// using the explicitly named segment iteration policy and execute
// segments as HIP kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         HIP execution for segments.
 *
 ******************************************************************************
 */
template <typename LoopBody,
          typename IterationMapping, typename IterationGetter,
          typename Concretizer, bool Async,
          typename... SegmentTypes>
RAJA_INLINE resources::EventProxy<resources::Hip>
forall_impl(resources::Hip r,
            ExecPolicy<seq_segit, ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Concretizer, Async>>,
            const TypedIndexSet<SegmentTypes...>& iset,
            LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(r,
                     isi,
                     detail::CallForall(),
                     ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Concretizer, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) RAJA::hip::synchronize(r);
  return resources::EventProxy<resources::Hip>(r);
}

}  // namespace hip

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
