/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with HIP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_policy_hip_kernel_internal_HPP
#define RAJA_policy_hip_kernel_internal_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"


namespace RAJA
{

namespace internal
{

struct LaunchDims {

  HipDims dims;
  HipDims min_dims;

  LaunchDims() = default;
  LaunchDims(LaunchDims const&) = default;
  LaunchDims& operator=(LaunchDims const&) = default;

  RAJA_INLINE
  LaunchDims(HipDims _dims)
    : dims{_dims}
    , min_dims{}
  { }

  RAJA_INLINE
  LaunchDims(HipDims _dims, HipDims _min_dims)
    : dims{_dims}
    , min_dims{_min_dims}
  { }

  RAJA_INLINE
  LaunchDims max(LaunchDims const &c) const
  {
    LaunchDims result;

    result.dims.blocks.x = std::max(c.dims.blocks.x, dims.blocks.x);
    result.dims.blocks.y = std::max(c.dims.blocks.y, dims.blocks.y);
    result.dims.blocks.z = std::max(c.dims.blocks.z, dims.blocks.z);

    result.min_dims.blocks.x = std::max(c.min_dims.blocks.x, min_dims.blocks.x);
    result.min_dims.blocks.y = std::max(c.min_dims.blocks.y, min_dims.blocks.y);
    result.min_dims.blocks.z = std::max(c.min_dims.blocks.z, min_dims.blocks.z);

    result.dims.threads.x = std::max(c.dims.threads.x, dims.threads.x);
    result.dims.threads.y = std::max(c.dims.threads.y, dims.threads.y);
    result.dims.threads.z = std::max(c.dims.threads.z, dims.threads.z);

    result.min_dims.threads.x = std::max(c.min_dims.threads.x, min_dims.threads.x);
    result.min_dims.threads.y = std::max(c.min_dims.threads.y, min_dims.threads.y);
    result.min_dims.threads.z = std::max(c.min_dims.threads.z, min_dims.threads.z);

    return result;
  }

  RAJA_INLINE
  int num_blocks() const {
    return dims.num_blocks();
  }

  RAJA_INLINE
  int num_threads() const {
    return dims.num_threads();
  }


  RAJA_INLINE
  void clamp_to_min_blocks() {
    dims.blocks.x = std::max(min_dims.blocks.x, dims.blocks.x);
    dims.blocks.y = std::max(min_dims.blocks.y, dims.blocks.y);
    dims.blocks.z = std::max(min_dims.blocks.z, dims.blocks.z);
  };

  RAJA_INLINE
  void clamp_to_min_threads() {
    dims.threads.x = std::max(min_dims.threads.x, dims.threads.x);
    dims.threads.y = std::max(min_dims.threads.y, dims.threads.y);
    dims.threads.z = std::max(min_dims.threads.z, dims.threads.z);
  };

};


template <camp::idx_t cur_stmt, camp::idx_t num_stmts, typename StmtList>
struct HipStatementListExecutorHelper {

  using next_helper_t =
      HipStatementListExecutorHelper<cur_stmt + 1, num_stmts, StmtList>;

  using cur_stmt_t = camp::at_v<StmtList, cur_stmt>;

  template <typename Data>
  inline static RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // Execute stmt
    cur_stmt_t::exec(data, thread_active);

    // Execute next stmt
    next_helper_t::exec(data, thread_active);
  }


  template <typename Data>
  inline static LaunchDims calculateDimensions(Data &data)
  {
    // Compute this statements launch dimensions
    LaunchDims statement_dims = cur_stmt_t::calculateDimensions(data);

    // call the next statement in the list
    LaunchDims next_dims = next_helper_t::calculateDimensions(data);

    // Return the maximum of the two
    return statement_dims.max(next_dims);
  }
};

template <camp::idx_t num_stmts, typename StmtList>
struct HipStatementListExecutorHelper<num_stmts, num_stmts, StmtList> {

  template <typename Data>
  inline static RAJA_DEVICE void exec(Data &, bool)
  {
    // nop terminator
  }

  template <typename Data>
  inline static LaunchDims calculateDimensions(Data &)
  {
    return LaunchDims();
  }
};


template <typename Data, typename Policy, typename Types>
struct HipStatementExecutor;

template <typename Data, typename StmtList, typename Types>
struct HipStatementListExecutor;


template <typename Data, typename... Stmts, typename Types>
struct HipStatementListExecutor<Data, StatementList<Stmts...>, Types> {

  using enclosed_stmts_t =
      camp::list<HipStatementExecutor<Data, Stmts, Types>...>;

  static constexpr size_t num_stmts = sizeof...(Stmts);

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Execute statements in order with helper class
    HipStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::exec(data, thread_active);
  }



  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Compute this statements launch dimensions
    return HipStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::
        calculateDimensions(data);
  }
};


template <typename StmtList, typename Data, typename Types>
using hip_statement_list_executor_t = HipStatementListExecutor<
    Data,
    StmtList,
    Types>;


// specialization for direct sequential policies
template<typename kernel_indexer>
struct KernelDimensionCalculator;

// specialization for direct sequential policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct<>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>>>
{
  using IndexMapper = hip::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>;

  template < typename IdxT >
  static void set_dimensions(HipDims& RAJA_UNUSED_ARG(dims), HipDims& RAJA_UNUSED_ARG(min_dims), IdxT len)
  {
    if ( len > static_cast<IdxT>(1) ) {
      RAJA_ABORT_OR_THROW("len exceeds the size of the directly mapped index space");
    }
  }
};

// specialization for direct thread policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct<>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>>>
{
  using IndexMapper = hip::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    // BEWARE: if calculated block_size is too high then the kernel launch will fail
    set_hip_dim<dim>(dims.threads, static_cast<IdxT>(len));
    set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(len));
  }
};
///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct<>,
                                                    sync,
                                                    hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>>>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    if ( len > static_cast<IdxT>(IndexMapper::block_size) ) {
      RAJA_ABORT_OR_THROW("len exceeds the size of the directly mapped index space");
    }
    set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(IndexMapper::block_size));
  }
};

// specialization for direct block policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct<>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>>>
{
  using IndexMapper = hip::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(len));
    set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(len));
  }
};
///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct<>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0, "grid size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    if ( len > static_cast<IdxT>(IndexMapper::grid_size) ) {
      RAJA_ABORT_OR_THROW("len exceeds the size of the directly mapped index space");
    }
    set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
    set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
  }
};

// specialization for direct global policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct<>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>>>
{
  using IndexMapper = hip::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(HipDims& RAJA_UNUSED_ARG(dims), HipDims& RAJA_UNUSED_ARG(min_dims), IdxT len)
  {
    if (len > static_cast<IdxT>(0)) {
      RAJA_ABORT_OR_THROW("must know one of block_size or grid_size");
    }
  }
};
///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct<>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0, "grid size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    // BEWARE: if calculated block_size is too high then the kernel launch will fail
    set_hip_dim<dim>(dims.threads, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::grid_size)));
    set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
    set_hip_dim<dim>(min_dims.threads, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::grid_size)));
    set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
  }
};
///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct<>,
                                                    sync,
                                                    hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>>>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(dims.blocks, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::block_size)));
    set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(min_dims.blocks, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::block_size)));
  }
};
///
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct<>,
                                                    sync,
                                                    hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");
  static_assert(GRID_SIZE > 0, "grid size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    if ( len > (static_cast<IdxT>(IndexMapper::block_size) *
                static_cast<IdxT>(IndexMapper::grid_size)) ) {
      RAJA_ABORT_OR_THROW("len exceeds the size of the directly mapped index space");
    }
    set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
    set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
  }
};


// specialization for strided loop sequential policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop<named_usage::unspecified>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>>>
{
  using IndexMapper = hip::IndexGlobal<dim, named_usage::ignored, named_usage::ignored>;

  template < typename IdxT >
  static void set_dimensions(HipDims& RAJA_UNUSED_ARG(dims), HipDims& RAJA_UNUSED_ARG(min_dims), IdxT RAJA_UNUSED_ARG(len))
  {
  }
};

// specialization for strided loop thread policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop<named_usage::unspecified>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>>>
{
  using IndexMapper = hip::IndexGlobal<dim, named_usage::unspecified, named_usage::ignored>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    // BEWARE: if calculated block_size is too high then the kernel launch will fail
    set_hip_dim<dim>(dims.threads, static_cast<IdxT>(len));
    set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(1));
  }
};
///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop<named_usage::unspecified>,
                                                    sync,
                                                    hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>>>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::ignored>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT RAJA_UNUSED_ARG(len))
  {
    set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(IndexMapper::block_size));
  }
};

// specialization for strided loop block policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop<named_usage::unspecified>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>>>
{
  using IndexMapper = hip::IndexGlobal<dim, named_usage::ignored, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(len));
    set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(1));
  }
};
///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop<named_usage::unspecified>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0, "grid size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, named_usage::ignored, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT RAJA_UNUSED_ARG(len))
  {
    set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
    set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
  }
};

// specialization for strided loop global policies
template<named_dim dim, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop<named_usage::unspecified>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>>>
{
  using IndexMapper = hip::IndexGlobal<dim, named_usage::unspecified, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    if (len > static_cast<IdxT>(0)) {
      set_hip_dim<dim>(dims.threads, static_cast<IdxT>(1));
      set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(1));
      set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(1));
      set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(1));
    }
  }
};
///
template<named_dim dim, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop<named_usage::unspecified>,
                                                    sync,
                                                    hip::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>>>
{
  static_assert(GRID_SIZE > 0, "grid size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, named_usage::unspecified, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    // BEWARE: if calculated block_size is too high then the kernel launch will fail
    set_hip_dim<dim>(dims.threads, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::grid_size)));
    set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
    set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(1));
    set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
  }
};
///
template<named_dim dim, int BLOCK_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop<named_usage::unspecified>,
                                                    sync,
                                                    hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>>>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, BLOCK_SIZE, named_usage::unspecified>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT len)
  {
    set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(dims.blocks, RAJA_DIVIDE_CEILING_INT(len, static_cast<IdxT>(IndexMapper::block_size)));
    set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(1));
  }
};
///
template<named_dim dim, int BLOCK_SIZE, int GRID_SIZE, kernel_sync_requirement sync>
struct KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop<named_usage::unspecified>,
                                                    sync,
                                                    hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>>>
{
  static_assert(BLOCK_SIZE > 0, "block size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");
  static_assert(GRID_SIZE > 0, "grid size must be > 0, named_usage::unspecified, or named_usage::ignored with kernel");

  using IndexMapper = hip::IndexGlobal<dim, BLOCK_SIZE, GRID_SIZE>;

  template < typename IdxT >
  static void set_dimensions(HipDims& dims, HipDims& min_dims, IdxT RAJA_UNUSED_ARG(len))
  {
    set_hip_dim<dim>(dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
    set_hip_dim<dim>(min_dims.threads, static_cast<IdxT>(IndexMapper::block_size));
    set_hip_dim<dim>(min_dims.blocks, static_cast<IdxT>(IndexMapper::grid_size));
  }
};

}  // namespace internal

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
