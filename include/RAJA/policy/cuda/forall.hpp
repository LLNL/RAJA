/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via CUDA kernel launch.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_cuda_HPP
#define RAJA_forall_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <algorithm>

#include "RAJA/pattern/forall.hpp"

#include "RAJA/pattern/params/forall.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include "RAJA/index/IndexSet.hpp"

#include "RAJA/util/resource.hpp"

namespace RAJA
{
namespace policy
{
namespace cuda
{

namespace impl
{

/*!
 ******************************************************************************
 *
 * \brief  Cuda kernel block and grid dimension calculator template.
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
template<typename IterationMapping, typename IterationGetter, typename UniqueMarker>
struct ForallDimensionCalculator;

// The general cases handle fixed BLOCK_SIZE > 0 and/or GRID_SIZE > 0
// there are specializations for named_usage::unspecified
// but named_usage::ignored is not supported so no specializations are provided
// and static_asserts in the general case catch unsupported values
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::Direct,
                                 ::RAJA::cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0 or named_usage::unspecified with forall");
  static_assert(GRID_SIZE > 0, "grid size must be > 0 or named_usage::unspecified with forall");

  using IndexGetter = ::RAJA::cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(internal::CudaDims& dims, IdxT len,
                             const void* RAJA_UNUSED_ARG(func), size_t RAJA_UNUSED_ARG(dynamic_shmem_size))
  {
    if ( len > (static_cast<IdxT>(IndexGetter::block_size) *
                static_cast<IdxT>(IndexGetter::grid_size)) ) {
      RAJA_ABORT_OR_THROW("len exceeds the size of the directly mapped index space");
    }

    internal::set_cuda_dim<dim>(dims.threads, static_cast<IdxT>(IndexGetter::block_size));
    internal::set_cuda_dim<dim>(dims.blocks, static_cast<IdxT>(IndexGetter::grid_size));
  }
};

template<named_dim dim, int GRID_SIZE, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::Direct,
                                 ::RAJA::cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>,
                                 UniqueMarker>
{
  static_assert(GRID_SIZE > 0, "grid size must be > 0 or named_usage::unspecified with forall");

  using IndexGetter = ::RAJA::cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(internal::CudaDims& dims, IdxT len,
                             const void* RAJA_UNUSED_ARG(func), size_t RAJA_UNUSED_ARG(dynamic_shmem_size))
  {
    // BEWARE: if calculated block_size is too high then the kernel launch will fail
    internal::set_cuda_dim<dim>(dims.threads, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexGetter::grid_size)));
    internal::set_cuda_dim<dim>(dims.blocks, static_cast<IdxT>(IndexGetter::grid_size));
  }
};

template<named_dim dim, int BLOCK_SIZE, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::Direct,
                                 ::RAJA::cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0 or named_usage::unspecified with forall");

  using IndexGetter = ::RAJA::cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(internal::CudaDims& dims, IdxT len,
                             const void* RAJA_UNUSED_ARG(func), size_t RAJA_UNUSED_ARG(dynamic_shmem_size))
  {
    internal::set_cuda_dim<dim>(dims.threads, static_cast<IdxT>(IndexGetter::block_size));
    internal::set_cuda_dim<dim>(dims.blocks, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexGetter::block_size)));
  }
};

template<named_dim dim, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::Direct,
                                 ::RAJA::cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>,
                                 UniqueMarker>
{
  using IndexGetter = ::RAJA::cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(internal::CudaDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::cuda::CudaOccupancyCalculator<UniqueMarker> oc(func);
    auto max_sizes = oc.get_max_block_size_and_grid_size(dynamic_shmem_size);

    internal::set_cuda_dim<dim>(dims.threads, static_cast<IdxT>(max_sizes.first));
    internal::set_cuda_dim<dim>(dims.blocks, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(max_sizes.first)));
  }
};

template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::StridedLoop,
                                 ::RAJA::cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0 or named_usage::unspecified with forall");
  static_assert(GRID_SIZE > 0, "grid size must be > 0 or named_usage::unspecified with forall");

  using IndexMapper = ::RAJA::cuda::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(internal::CudaDims& dims, IdxT RAJA_UNUSED_ARG(len),
                             const void* RAJA_UNUSED_ARG(func), size_t RAJA_UNUSED_ARG(dynamic_shmem_size))
  {
    internal::set_cuda_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    internal::set_cuda_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
  }
};

template<named_dim dim, int GRID_SIZE, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::StridedLoop,
                                 ::RAJA::cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>,
                                 UniqueMarker>
{
  static_assert(GRID_SIZE > 0, "grid size must be > 0 or named_usage::unspecified with forall");

  using IndexMapper = ::RAJA::cuda::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(internal::CudaDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::cuda::CudaOccupancyCalculator<UniqueMarker> oc(func);
    auto max_sizes = oc.get_max_block_size_and_grid_size(dynamic_shmem_size);

    IdxT calculated_block_size = std::min(
        RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::grid_size)),
        static_cast<IdxT>(max_sizes.first));

    internal::set_cuda_dim<dim>(dims.threads, calculated_block_size);
    internal::set_cuda_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
  }
};

template<named_dim dim, int BLOCK_SIZE, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::StridedLoop,
                                 ::RAJA::cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>,
                                 UniqueMarker>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0 or named_usage::unspecified with forall");

  using IndexMapper = ::RAJA::cuda::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(internal::CudaDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::cuda::CudaOccupancyCalculator<UniqueMarker> oc(func);
    auto max_grid_size = oc.get_max_grid_size(dynamic_shmem_size,
                                              static_cast<IdxT>(IndexMapper::block_size));

    IdxT calculated_grid_size = std::min(
        RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::block_size)),
        static_cast<IdxT>(max_grid_size));

    internal::set_cuda_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    internal::set_cuda_dim<dim>(dims.blocks, calculated_grid_size);
  }
};

template<named_dim dim, typename UniqueMarker>
struct ForallDimensionCalculator<::RAJA::iteration_mapping::StridedLoop,
                                 ::RAJA::cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>,
                                 UniqueMarker>
{
  using IndexMapper = ::RAJA::cuda::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(internal::CudaDims& dims, IdxT len,
                             const void* func, size_t dynamic_shmem_size)
  {
    ::RAJA::cuda::CudaOccupancyCalculator<UniqueMarker> oc(func);
    auto max_sizes = oc.get_max_block_size_and_grid_size(dynamic_shmem_size);

    IdxT calculated_grid_size = std::min(
        RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(max_sizes.first)),
        static_cast<IdxT>(max_sizes.second));

    internal::set_cuda_dim<dim>(dims.threads, static_cast<IdxT>(max_sizes.first));
    internal::set_cuda_dim<dim>(dims.blocks, calculated_grid_size);
  }
};

//
//////////////////////////////////////////////////////////////////////
//
// CUDA kernel templates.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernel forall template.
 *
 ******************************************************************************
 */
template <typename EXEC_POL,
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_same<IterationMapping, iteration_mapping::Direct>::value &&
                (IterationGetter::block_size > 0),
              size_t > BlockSize = IterationGetter::block_size>
__launch_bounds__(BlockSize, BlocksPerSM) __global__
void forall_cuda_kernel(LOOP_BODY loop_body,
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
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_same<IterationMapping, iteration_mapping::Direct>::value &&
                (IterationGetter::block_size <= 0),
              size_t > RAJA_UNUSED_ARG(BlockSize) = 0>
__global__
void forall_cuda_kernel(LOOP_BODY loop_body,
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
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_same<IterationMapping, iteration_mapping::Direct>::value &&
                (IterationGetter::block_size > 0),
              size_t > BlockSize = IterationGetter::block_size>
__launch_bounds__(BlockSize, BlocksPerSM) __global__
void forallp_cuda_kernel(LOOP_BODY loop_body,
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
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_same<IterationMapping, iteration_mapping::Direct>::value &&
                (IterationGetter::block_size <= 0),
              size_t > RAJA_UNUSED_ARG(BlockSize) = 0>
__global__
void forallp_cuda_kernel(LOOP_BODY loop_body,
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
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_same<IterationMapping, iteration_mapping::StridedLoop>::value &&
                (IterationGetter::block_size > 0),
              size_t > BlockSize = IterationGetter::block_size>
__launch_bounds__(BlockSize, BlocksPerSM) __global__
void forall_cuda_kernel(LOOP_BODY loop_body,
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
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_same<IterationMapping, iteration_mapping::StridedLoop>::value &&
                (IterationGetter::block_size <= 0),
              size_t > RAJA_UNUSED_ARG(BlockSize) = 0>
__global__
void forall_cuda_kernel(LOOP_BODY loop_body,
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
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_same<IterationMapping, iteration_mapping::StridedLoop>::value &&
                (IterationGetter::block_size > 0),
              size_t > BlockSize = IterationGetter::block_size>
__launch_bounds__(BlockSize, BlocksPerSM) __global__
void forallp_cuda_kernel(LOOP_BODY loop_body,
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
          size_t BlocksPerSM,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType,
          typename ForallParam,
          typename IterationMapping = typename EXEC_POL::IterationMapping,
          typename IterationGetter = typename EXEC_POL::IterationGetter,
          std::enable_if_t<
                std::is_same<IterationMapping, iteration_mapping::StridedLoop>::value &&
                (IterationGetter::block_size <= 0),
              size_t > RAJA_UNUSED_ARG(BlockSize) = 0>
__global__
void forallp_cuda_kernel(LOOP_BODY loop_body,
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
// Function templates for CUDA execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody,
          typename IterationMapping, typename IterationGetter,
          size_t BlocksPerSM, bool Async,
          typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Cuda>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(resources::Cuda cuda_res,
            ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping, IterationGetter, BlocksPerSM, Async>const&,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping, IterationGetter, BlocksPerSM, Async>;
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
        &impl::forall_cuda_kernel<EXEC_POL, BlocksPerSM, Iterator, LOOP_BODY, IndexType>);

    //
    // Setup shared memory buffers
    //
    size_t shmem = 0;

    //
    // Compute the kernel dimensions
    //
    internal::CudaDims dims(1);
    DimensionCalculator::set_dimensions(dims, len, func, shmem);

    RAJA_FT_BEGIN;

    {
      //
      // Privatize the loop_body, using make_launch_body to setup reductions
      //
      LOOP_BODY body = RAJA::cuda::make_launch_body(
          dims.blocks, dims.threads, shmem, cuda_res, std::forward<LoopBody>(loop_body));

      //
      // Launch the kernels
      //
      void *args[] = {(void*)&body, (void*)&begin, (void*)&len};
      RAJA::cuda::launch(func, dims.blocks, dims.threads, args, shmem, cuda_res, Async);
    }

    RAJA_FT_END;
  }

  return resources::EventProxy<resources::Cuda>(cuda_res);
}


template <typename Iterable, typename LoopBody,
          typename IterationMapping, typename IterationGetter,
          size_t BlocksPerSM, bool Async,
          typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Cuda>,
  RAJA::expt::type_traits::is_ForallParamPack<ForallParam>,
  concepts::negate< RAJA::expt::type_traits::is_ForallParamPack_empty<ForallParam>> >
forall_impl(resources::Cuda cuda_res,
            ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping, IterationGetter, BlocksPerSM, Async> const&,
            Iterable&& iter,
            LoopBody&& loop_body,
            ForallParam f_params)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;
  using EXEC_POL = ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping, IterationGetter, BlocksPerSM, Async>;
  using UniqueMarker = ::camp::list<IterationMapping, IterationGetter, camp::num<BlocksPerSM>, LOOP_BODY, Iterator, ForallParam>;
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
        impl::forallp_cuda_kernel< EXEC_POL, BlocksPerSM, Iterator, LOOP_BODY, IndexType, camp::decay<ForallParam> >);

    //
    // Setup shared memory buffers
    //
    size_t shmem = 0;

    //
    // Compute the kernel dimensions
    //
    internal::CudaDims dims(1);
    DimensionCalculator::set_dimensions(dims, len, func, shmem);

    RAJA_FT_BEGIN;

    RAJA::cuda::detail::cudaInfo launch_info;
    launch_info.gridDim = dims.blocks;
    launch_info.blockDim = dims.threads;
    launch_info.res = cuda_res;

    {
      RAJA::expt::ParamMultiplexer::init<EXEC_POL>(f_params, launch_info);

      //
      // Privatize the loop_body, using make_launch_body to setup reductions
      //
      LOOP_BODY body = RAJA::cuda::make_launch_body(
          dims.blocks, dims.threads, shmem, cuda_res, std::forward<LoopBody>(loop_body));

      //
      // Launch the kernels
      //
      void *args[] = {(void*)&body, (void*)&begin, (void*)&len, (void*)&f_params};
      RAJA::cuda::launch(func, dims.blocks, dims.threads, args, shmem, cuda_res, Async);

      RAJA::expt::ParamMultiplexer::resolve<EXEC_POL>(f_params, launch_info);
    }

    RAJA_FT_END;
  }

  return resources::EventProxy<resources::Cuda>(cuda_res);
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// using the explicitly named segment iteration policy and execute
// segments as CUDA kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         CUDA execution for segments.
 *
 ******************************************************************************
 */
template <typename LoopBody,
          typename IterationMapping, typename IterationGetter,
          size_t BlocksPerSM, bool Async,
          typename... SegmentTypes>
RAJA_INLINE resources::EventProxy<resources::Cuda>
forall_impl(resources::Cuda r,
            ExecPolicy<seq_segit, ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping, IterationGetter, BlocksPerSM, Async>>,
            const TypedIndexSet<SegmentTypes...>& iset,
            LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(r,
                     isi,
                     detail::CallForall(),
                     ::RAJA::policy::cuda::cuda_exec_explicit<IterationMapping, IterationGetter, BlocksPerSM, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) RAJA::cuda::synchronize(r);
  return resources::EventProxy<resources::Cuda>(r);
}

}  // namespace cuda

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
