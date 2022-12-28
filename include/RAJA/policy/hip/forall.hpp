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

namespace internal
{

template<int dim, hip_dim_member_t t_block_size, hip_dim_member_t t_grid_size>
struct HipForallDirect<::RAJA::internal::HipIndexGlobal<dim, t_block_size, t_grid_size>>
{
  using IndexMapper = ::RAJA::internal::HipIndexGlobal<dim, t_block_size, t_grid_size>;

  template < typename UniqueMarker >
  using OccupancyCalculator = std::conditional_t<
      (t_block_size == 0 && t_grid_size == 0),
      ::RAJA::hip::HipOccupancyCalculator<UniqueMarker>,
      ::RAJA::hip::HipOccupancyDefaults>;

  IndexMapper indexer;

  RAJA_HOST_DEVICE constexpr
  HipForallDirect(hip_dim_member_t _block_size = 0,
                  hip_dim_member_t _grid_size = 0)
    : indexer(_block_size, _grid_size)
  { }

  template < typename UniqueMarker >
  inline OccupancyCalculator<UniqueMarker> get_occupancy_calculator(const void* func) const
  {
    return {func};
  }

  template < typename IdxT, typename OC >
  inline void set_dimensions(HipDims& dims, IdxT len,
                             OC const& oc, size_t dynamic_shmem_size) const
  {
    if (indexer.block_size == 0 && indexer.grid_size == 0) {
      auto max_sizes = oc.get_max_block_size_and_grid_size(dynamic_shmem_size);
      set_hip_dim<dim>(dims.threads, static_cast<IdxT>(max_sizes.first));
      set_hip_dim<dim>(dims.blocks, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(max_sizes.first)));

    } else if (indexer.block_size == 0) {
      // BEWARE: if calculated block_size is too high then the kernel launch will fail
      set_hip_dim<dim>(dims.threads, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(indexer.grid_size)));
      set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(indexer.grid_size));

    } else if (indexer.grid_size == 0) {
      set_hip_dim<dim>(dims.threads, static_cast<IdxT>(indexer.block_size));
      set_hip_dim<dim>(dims.blocks, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(indexer.block_size)));

    } else {
      if ( len > (static_cast<IdxT>(indexer.block_size) *
                  static_cast<IdxT>(indexer.grid_size)) ) {
        RAJA_ABORT_OR_THROW("len exceeds the size of the directly mapped index space");
      }
      set_hip_dim<dim>(dims.threads, static_cast<IdxT>(indexer.block_size));
      set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(indexer.grid_size));
    }
  }
};

template<int dim, hip_dim_member_t t_block_size, hip_dim_member_t t_grid_size>
struct HipForallLoop<::RAJA::internal::HipIndexGlobal<dim, t_block_size, t_grid_size>>
{
  using IndexMapper = ::RAJA::internal::HipIndexGlobal<dim, t_block_size, t_grid_size>;

  template < typename UniqueMarker >
  using OccupancyCalculator = std::conditional_t<
      (t_block_size == 0 || t_grid_size == 0),
      ::RAJA::hip::HipOccupancyCalculator<UniqueMarker>,
      ::RAJA::hip::HipOccupancyDefaults>;

  IndexMapper indexer;

  RAJA_HOST_DEVICE constexpr
  HipForallLoop(hip_dim_member_t _block_size = 0,
                hip_dim_member_t _grid_size = 0)
    : indexer(_block_size, _grid_size)
  { }

  template < typename UniqueMarker >
  inline OccupancyCalculator<UniqueMarker> get_occupancy_calculator(const void* func) const
  {
    return {func};
  }

  template < typename IdxT, typename OC >
  inline void set_dimensions(HipDims& dims, IdxT len,
                             OC const& oc, size_t dynamic_shmem_size) const
  {
    if (indexer.block_size == 0) {
      auto max_sizes = oc.get_max_block_size_and_grid_size(dynamic_shmem_size);

      if (indexer.grid_size == 0) {
        IdxT calculated_grid_size = std::min(
            RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(max_sizes.first)),
            static_cast<IdxT>(max_sizes.second));
        set_hip_dim<dim>(dims.threads, static_cast<IdxT>(max_sizes.first));
        set_hip_dim<dim>(dims.blocks, calculated_grid_size);

      } else {
        IdxT calculated_block_size = std::min(
            RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(indexer.grid_size)),
            static_cast<IdxT>(max_sizes.first));
        set_hip_dim<dim>(dims.threads, calculated_block_size);
        set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(indexer.grid_size));
      }

    } else if (indexer.grid_size == 0) {
      auto max_grid_size = oc.get_max_grid_size(dynamic_shmem_size,
                                                static_cast<IdxT>(indexer.block_size));
      IdxT calculated_grid_size = std::min(
          RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(indexer.block_size)),
          static_cast<IdxT>(max_grid_size));
      set_hip_dim<dim>(dims.threads, static_cast<IdxT>(indexer.block_size));
      set_hip_dim<dim>(dims.blocks, calculated_grid_size);

    } else {
      set_hip_dim<dim>(dims.threads, static_cast<IdxT>(indexer.block_size));
      set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(indexer.grid_size));
    }
  }
};

} // namespace internal

namespace type_traits {

template <typename IndexMapper>
struct is_hip_direct_indexer<::RAJA::internal::HipForallDirect<IndexMapper>> : std::true_type {};
template <typename IndexMapper>
struct is_hip_loop_indexer<::RAJA::internal::HipForallLoop<IndexMapper>> : std::true_type {};

} // namespace type_traits


namespace policy
{
namespace hip
{

namespace impl
{

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
          typename Indexer = typename EXEC_POL::Indexer,
          typename IndexMapper = typename Indexer::IndexMapper,
          concepts::enable_if_t< size_t,
              type_traits::is_hip_block_size_known<IndexMapper>,
              type_traits::is_hip_direct_indexer<Indexer> > BlockSize = IndexMapper::block_size>
__launch_bounds__(BlockSize, 1) __global__
void forall_hip_kernel(LOOP_BODY loop_body,
                       const Iterator idx,
                       IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = IndexMapper::template index<IndexType>();
  if (ii < length) {
    body(idx[ii]);
  }
}
///
template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename Indexer = typename EXEC_POL::Indexer,
          typename IndexMapper = typename Indexer::IndexMapper,
          concepts::enable_if_t< size_t,
              type_traits::is_hip_block_size_known<IndexMapper>,
              type_traits::is_hip_loop_indexer<Indexer> > BlockSize = IndexMapper::block_size>
__launch_bounds__(BlockSize, 1) __global__
void forall_hip_kernel(LOOP_BODY loop_body,
                       const Iterator idx,
                       IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  for (auto ii = IndexMapper::template index<IndexType>();
       ii < length;
       ii += IndexMapper::template size<IndexType>()) {
    body(idx[ii]);
  }
}
///
template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename Indexer = typename EXEC_POL::Indexer,
          typename IndexMapper = typename Indexer::IndexMapper,
          concepts::enable_if_t< size_t,
              concepts::negate< type_traits::is_hip_block_size_known<IndexMapper> >,
              type_traits::is_hip_direct_indexer<Indexer> > = 0>
__global__
void forall_hip_kernel(LOOP_BODY loop_body,
                       const Iterator idx,
                       IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = IndexMapper::template index<IndexType>();
  if (ii < length) {
    body(idx[ii]);
  }
}
///
template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename Indexer = typename EXEC_POL::Indexer,
          typename IndexMapper = typename Indexer::IndexMapper,
          concepts::enable_if_t< size_t,
              concepts::negate< type_traits::is_hip_block_size_known<IndexMapper> >,
              type_traits::is_hip_loop_indexer<Indexer> > = 0>
__global__
void forall_hip_kernel(LOOP_BODY loop_body,
                       const Iterator idx,
                       IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  for (auto ii = IndexMapper::template index<IndexType>();
       ii < length;
       ii += IndexMapper::template size<IndexType>()) {
    body(idx[ii]);
  }
}

template <typename EXEC_POL,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam,
          typename Indexer = typename EXEC_POL::Indexer,
          typename IndexMapper = typename Indexer::IndexMapper,
          concepts::enable_if_t< size_t,
              type_traits::is_hip_block_size_known<IndexMapper>,
              type_traits::is_hip_direct_indexer<Indexer> > BlockSize = IndexMapper::block_size>
__launch_bounds__(BlockSize, 1) __global__
void forallp_hip_kernel(LOOP_BODY loop_body,
                        const Iterator idx,
                        IndexType length,
                        ForallParam f_params)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = IndexMapper::template index<IndexType>();
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
          typename Indexer = typename EXEC_POL::Indexer,
          typename IndexMapper = typename Indexer::IndexMapper,
          concepts::enable_if_t< size_t,
              type_traits::is_hip_block_size_known<IndexMapper>,
              type_traits::is_hip_loop_indexer<Indexer> > BlockSize = IndexMapper::block_size>
__launch_bounds__(BlockSize, 1) __global__
void forallp_hip_kernel(LOOP_BODY loop_body,
                        const Iterator idx,
                        IndexType length,
                        ForallParam f_params)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  for (auto ii = IndexMapper::template index<IndexType>();
       ii < length;
       ii += IndexMapper::template size<IndexType>()) {
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
          typename Indexer = typename EXEC_POL::Indexer,
          typename IndexMapper = typename Indexer::IndexMapper,
          concepts::enable_if_t< size_t,
              concepts::negate< type_traits::is_hip_block_size_known<IndexMapper> >,
              type_traits::is_hip_direct_indexer<Indexer> > = 0>
__global__
void forallp_hip_kernel(LOOP_BODY loop_body,
                        const Iterator idx,
                        IndexType length,
                        ForallParam f_params)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = IndexMapper::template index<IndexType>();
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
          typename Indexer = typename EXEC_POL::Indexer,
          typename IndexMapper = typename Indexer::IndexMapper,
          concepts::enable_if_t< size_t,
              concepts::negate< type_traits::is_hip_block_size_known<IndexMapper> >,
              type_traits::is_hip_loop_indexer<Indexer> > = 0>
__global__
void forallp_hip_kernel(LOOP_BODY loop_body,
                        const Iterator idx,
                        IndexType length,
                        ForallParam f_params)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  for (auto ii = IndexMapper::template index<IndexType>();
       ii < length;
       ii += IndexMapper::template size<IndexType>()) {
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

template <typename Iterable, typename LoopBody, typename Indexer, bool Async, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Hip>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Hip hip_res,
            ::RAJA::policy::hip::hip_exec<Indexer, Async>const& policy,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam)
{
  using ITERABLE = camp::decay<Iterable>;
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = ::RAJA::policy::hip::hip_exec<Indexer, Async>;
  using UniqueMarker = ::camp::list<EXEC_POL, ITERABLE, LOOP_BODY, ForallParam>;

  auto func = reinterpret_cast<const void*>(
      &impl::forall_hip_kernel<EXEC_POL, Iterator, LOOP_BODY, IndexType>);

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end = std::end(iter);
  IndexType len = std::distance(begin, end);

  // Only launch kernel if we have something to iterate over
  if (len > 0) {
    //
    // Setup shared memory buffers
    //
    size_t shmem = 0;

    //
    // Compute the kernel dimensions
    //
    internal::HipDims dims(1);
    auto oc = policy.template get_occupancy_calculator<UniqueMarker>(func);
    policy.set_dimensions(dims, len, oc, shmem);

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


template <typename Iterable, typename LoopBody, typename Indexer, bool Async, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Hip>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  concepts::negate< RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>> >
forall_impl(resources::Hip hip_res,
            ::RAJA::policy::hip::hip_exec<Indexer, Async> const& policy,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam f_params)
{
  using ITERABLE = camp::decay<Iterable>;
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = ::RAJA::policy::hip::hip_exec<Indexer, Async>;
  using UniqueMarker = ::camp::list<EXEC_POL, ITERABLE, LOOP_BODY, ForallParam>;

  auto func = reinterpret_cast<const void*>(
      &impl::forallp_hip_kernel< EXEC_POL, Iterator, LOOP_BODY, IndexType, camp::decay<ForallParam> >);

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end = std::end(iter);
  IndexType len = std::distance(begin, end);

  // Only launch kernel if we have something to iterate over
  if (len > 0) {
    //
    // Setup shared memory buffers
    //
    size_t shmem = 0;

    //
    // Compute the kernel dimensions
    //
    internal::HipDims dims(1);
    auto oc = policy.template get_occupancy_calculator<UniqueMarker>(func);
    policy.set_dimensions(dims, len, oc, shmem);

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
          typename Indexer,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE resources::EventProxy<resources::Hip>
forall_impl(resources::Hip r,
            ExecPolicy<seq_segit, ::RAJA::policy::hip::hip_exec<Indexer, Async>>,
            const TypedIndexSet<SegmentTypes...>& iset,
            LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(r,
                     isi,
                     detail::CallForall(),
                     ::RAJA::policy::hip::hip_exec<Indexer, true>(),
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
