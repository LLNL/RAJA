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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
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

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"

#include "RAJA/index/IndexSet.hpp"

namespace RAJA
{
namespace policy
{
namespace hip
{

namespace impl
{

template<typename IterationMapping, typename IterationGetter, typename UniqueMarker>
struct ForallDimensionCalculator;

template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::Direct,
                                 ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE >= 0, "block size may not be ignored with forall");
  static_assert(GRID_SIZE >= 0, "grid size may not be ignored with forall");

  using IndexGetter = ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  using OccupancyCalculator = std::conditional_t<
        ( BLOCK_SIZE == named_usage::unspecified &&
          GRID_SIZE == named_usage::unspecified ),
      ::RAJA::hip::HipOccupancyCalculator<UniqueMarker>,
      ::RAJA::hip::HipOccupancyDefaults>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    if (IndexGetter::block_size == named_usage::unspecified &&
        IndexGetter::grid_size == named_usage::unspecified) {

      OccupancyCalculator oc(func);
      auto max_sizes = oc.get_max_block_size_and_grid_size(dynamic_shmem_size);

      internal::set_hip_dim<dim>(dims.threads, static_cast<IdxT>(max_sizes.first));
      internal::set_hip_dim<dim>(dims.blocks, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(max_sizes.first)));

    } else if (IndexGetter::block_size == named_usage::unspecified) {
      // BEWARE: if calculated block_size is too high then the kernel launch will fail
      internal::set_hip_dim<dim>(dims.threads, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexGetter::grid_size)));
      internal::set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexGetter::grid_size));

    } else if (IndexGetter::grid_size == named_usage::unspecified) {

      internal::set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexGetter::block_size));
      internal::set_hip_dim<dim>(dims.blocks, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexGetter::block_size)));

    } else {

      if ( len > (static_cast<IdxT>(IndexGetter::block_size) *
                  static_cast<IdxT>(IndexGetter::grid_size)) ) {
        RAJA_ABORT_OR_THROW("len exceeds the size of the directly mapped index space");
      }

      internal::set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexGetter::block_size));
      internal::set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexGetter::grid_size));
    }
  }
};

template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::StridedLoop,
                                 ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE >= 0, "block size may not be ignored with forall");
  static_assert(GRID_SIZE >= 0, "grid size may not be ignored with forall");

  using IndexMapper = ::RAJA::hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  using OccupancyCalculator = std::conditional_t<
        ( BLOCK_SIZE == named_usage::unspecified ||
          GRID_SIZE == named_usage::unspecified ),
      ::RAJA::hip::HipOccupancyCalculator<UniqueMarker>,
      ::RAJA::hip::HipOccupancyDefaults>;

  template < typename IdxT >
  static void set_dimensions(internal::HipDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    if (IndexMapper::block_size == named_usage::unspecified) {

      OccupancyCalculator oc(func);
      auto max_sizes = oc.get_max_block_size_and_grid_size(dynamic_shmem_size);

      if (IndexMapper::grid_size == named_usage::unspecified) {

        IdxT calculated_grid_size = std::min(
            RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(max_sizes.first)),
            static_cast<IdxT>(max_sizes.second));

        internal::set_hip_dim<dim>(dims.threads, static_cast<IdxT>(max_sizes.first));
        internal::set_hip_dim<dim>(dims.blocks, calculated_grid_size);

      } else {

        IdxT calculated_block_size = std::min(
            RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::grid_size)),
            static_cast<IdxT>(max_sizes.first));

        internal::set_hip_dim<dim>(dims.threads, calculated_block_size);
        internal::set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
      }

    } else if (IndexMapper::grid_size == named_usage::unspecified) {

      OccupancyCalculator oc(func);
      auto max_grid_size = oc.get_max_grid_size(dynamic_shmem_size,
                                                static_cast<IdxT>(IndexMapper::block_size));

      IdxT calculated_grid_size = std::min(
          RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::block_size)),
          static_cast<IdxT>(max_grid_size));

      internal::set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
      internal::set_hip_dim<dim>(dims.blocks, calculated_grid_size);

    } else {

      internal::set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
      internal::set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
    }
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
                std::is_same<IterationMapping, iteration_mapping::Direct>::value &&
                (IterationGetter::block_size > 0), size_t >
              BlockSize = IterationGetter::block_size>
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
                std::is_same<IterationMapping, iteration_mapping::Direct>::value &&
                (IterationGetter::block_size <= 0), size_t >
              = 0>
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
                std::is_same<IterationMapping, iteration_mapping::Direct>::value &&
                (IterationGetter::block_size > 0), size_t >
              BlockSize = IterationGetter::block_size>
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
                std::is_same<IterationMapping, iteration_mapping::Direct>::value &&
                (IterationGetter::block_size <= 0), size_t >
              = 0>
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
                std::is_same<IterationMapping, iteration_mapping::StridedLoop>::value &&
                (IterationGetter::block_size > 0), size_t >
              BlockSize = IterationGetter::block_size>
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
                std::is_same<IterationMapping, iteration_mapping::StridedLoop>::value &&
                (IterationGetter::block_size <= 0), size_t >
              = 0>
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
                std::is_same<IterationMapping, iteration_mapping::StridedLoop>::value &&
                (IterationGetter::block_size > 0), size_t >
              BlockSize = IterationGetter::block_size>
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
                std::is_same<IterationMapping, iteration_mapping::StridedLoop>::value &&
                (IterationGetter::block_size <= 0), size_t >
              = 0>
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
          typename IterationMapping, typename IterationGetter, bool Async,
          typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Hip>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Hip hip_res,
            ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Async>const&,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Async>;
  using UniqueMarker = ::camp::list<IterationMapping, IterationGetter, LOOP_BODY, Iterator, ForallParam>;
  using DimensionCalculator = impl::ForallDimensionCalculator<IterationMapping, IterationGetter, UniqueMarker>;

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
          typename IterationMapping, typename IterationGetter, bool Async,
          typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Hip>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  concepts::negate< RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>> >
forall_impl(resources::Hip hip_res,
            ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Async> const&,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam f_params)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Async>;
  using UniqueMarker = ::camp::list<IterationMapping, IterationGetter, LOOP_BODY, Iterator, ForallParam>;
  using DimensionCalculator = impl::ForallDimensionCalculator<IterationMapping, IterationGetter, UniqueMarker>;

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

      RAJA::expt::ParamMultiplexer::resolve<EXEC_POL>(f_params);
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
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE resources::EventProxy<resources::Hip>
forall_impl(resources::Hip r,
            ExecPolicy<seq_segit, ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, Async>>,
            const TypedIndexSet<SegmentTypes...>& iset,
            LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(r,
                     isi,
                     detail::CallForall(),
                     ::RAJA::policy::hip::hip_exec<IterationMapping, IterationGetter, true>(),
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
