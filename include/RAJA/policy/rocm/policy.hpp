/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA ROCM policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_rocm_HPP
#define RAJA_policy_rocm_HPP

#if defined(RAJA_ENABLE_ROCM)

#include "RAJA/config.hpp"
#include "RAJA/pattern/reduce.hpp"
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

#include <hc.hpp>
#include <hc_printf.hpp>
#define hipHostMallocDefault        0x0
#include <hip/hcc_detail/hip_runtime_api.h>


namespace RAJA
{

using rocm_dim_t = dim3;


///
/////////////////////////////////////////////////////////////////////
///
/// Generalizations of ROCM dim3 x, y and z used to describe
/// sizes and indices for threads and blocks.
///
/////////////////////////////////////////////////////////////////////
///

struct Dim3x {
  [[cpu]] [[hc]] inline unsigned int &operator()(rocm_dim_t &dim)
  {
    return dim.x;
  }

  [[cpu]] [[hc]] inline unsigned int operator()(rocm_dim_t const &dim)
  {
    return dim.x;
  }
};
///
struct Dim3y {
  [[cpu]] [[hc]] inline unsigned int &operator()(rocm_dim_t &dim)
  {
    return dim.y;
  }

  [[cpu]] [[hc]] inline unsigned int operator()(rocm_dim_t const &dim)
  {
    return dim.y;
  }
};
///
struct Dim3z {
  [[cpu]] [[hc]] inline unsigned int &operator()(rocm_dim_t &dim)
  {
    return dim.z;
  }

  [[cpu]] [[hc]] inline unsigned int operator()(rocm_dim_t const &dim)
  {
    return dim.z;
  }
};

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
namespace rocm
{

template <size_t BLOCK_SIZE, bool Async = false>
struct rocm_exec
    : public RAJA::
          make_policy_pattern_launch_platform_t<RAJA::Policy::rocm,
                                                RAJA::Pattern::forall,
                                                detail::get_launch<Async>::
                                                    value,
                                                RAJA::Platform::rocm> {
};


/*
 * Policy for on-device loops, akin to RAJA::loop_exec
 */
struct rocm_loop_exec
    : public RAJA::make_policy_pattern_launch_platform_t<RAJA::Policy::rocm,
                                                         RAJA::Pattern::forall,
                                                         RAJA::Launch::sync,
                                                         RAJA::Platform::rocm> {
};

//
// NOTE: There is no Index set segment iteration policy for ROCM
//

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction reduction policies
///
///////////////////////////////////////////////////////////////////////
///

template <size_t BLOCK_SIZE, bool Async = false, bool maybe_atomic = false>
struct rocm_reduce
    : public RAJA::
          make_policy_pattern_launch_platform_t<RAJA::Policy::rocm,
                                                RAJA::Pattern::reduce,
                                                detail::get_launch<Async>::
                                                    value,
                                                RAJA::Platform::rocm> {
};

template <size_t BLOCK_SIZE>
using rocm_reduce_async = rocm_reduce<BLOCK_SIZE, true, false>;

template <size_t BLOCK_SIZE>
using rocm_reduce_atomic = rocm_reduce<BLOCK_SIZE, false, true>;

template <size_t BLOCK_SIZE>
using rocm_reduce_atomic_async = rocm_reduce<BLOCK_SIZE, true, true>;


template <typename POL>
struct ROCmPolicy
    : public RAJA::
          make_policy_pattern_launch_platform_t<RAJA::Policy::rocm,
                                                RAJA::Pattern::forall,
                                                RAJA::Launch::undefined,
                                                RAJA::Platform::rocm> {

  using rocm_exec_policy = POL;
};

//
// Operations in the included files are parametrized using the following
// values for ROCM wavefront size and max block size.
//
constexpr const RAJA::Index_type WAVEFRONT_SIZE = 64;
constexpr const RAJA::Index_type MAX_BLOCK_SIZE = 1024;
constexpr const RAJA::Index_type MAX_WAVEFRONTS = MAX_BLOCK_SIZE / WAVEFRONT_SIZE;
static_assert(WAVEFRONT_SIZE >= MAX_WAVEFRONTS,
              "RAJA Assumption Broken: WAVEFRONT_SIZE < MAX_WAVEFRONTS");
static_assert(MAX_BLOCK_SIZE % WAVEFRONT_SIZE == 0,
              "RAJA Assumption Broken: MAX_BLOCK_SIZE not "
              "a multiple of WAVEFRONT_SIZE");

}  // end namespace rocm
}  // end namespace policy

using policy::rocm::rocm_exec;
using policy::rocm::rocm_loop_exec;
using policy::rocm::rocm_reduce;
using policy::rocm::rocm_reduce_async;
using policy::rocm::rocm_reduce_atomic;
using policy::rocm::rocm_reduce_atomic_async;
using policy::rocm::ROCmPolicy;

template<typename std::common_type<
    decltype(hc_get_group_id),
    decltype(hc_get_group_size),
    decltype(hc_get_num_groups),
    decltype(hc_get_workitem_id)>::type f>
class Coordinates {
    using R = decltype(f(0));

    struct X { __device__ operator R() const { return f(0); } };
    struct Y { __device__ operator R() const { return f(1); } };
    struct Z { __device__ operator R() const { return f(2); } };
public:
    static constexpr X x{};
    static constexpr Y y{};
    static constexpr Z z{};
};

static constexpr Coordinates<hc_get_group_size> blockDim;
static constexpr Coordinates<hc_get_group_id> blockIdx;
static constexpr Coordinates<hc_get_num_groups> gridDim;
static constexpr Coordinates<hc_get_workitem_id> threadIdx;

/*!
 * \brief Struct that contains two ROCM dim3's that represent the number of
 * thread block and the number of blocks.
 *
 * This is passed to the execution policies to setup the kernel launch.
 */
struct ROCmDim {
  rocm_dim_t num_threads;
  rocm_dim_t num_blocks;

  [[hc]] void print(void) const
  {
    hc::printf("<<< (%d,%d,%d), (%d,%d,%d) >>>\n",
           (int)num_blocks.x,
           (int)num_blocks.y,
           (int)num_blocks.z,
           (int)num_threads.x,
           (int)num_threads.y,
           (int)num_threads.z);
  }
  [[cpu]] void print(void) const
  {
    printf("<<< (%d,%d,%d), (%d,%d,%d) >>>\n",
           (int)num_blocks.x,
           (int)num_blocks.y,
           (int)num_blocks.z,
           (int)num_threads.x,
           (int)num_threads.y,
           (int)num_threads.z);
  }
};


RAJA_INLINE
constexpr RAJA::Index_type numBlocks(ROCmDim const &dim)
{
  return dim.num_blocks.x * dim.num_blocks.y * dim.num_blocks.z;
}

RAJA_INLINE
constexpr RAJA::Index_type numThreads(ROCmDim const &dim)
{
  return dim.num_threads.x * dim.num_threads.y * dim.num_threads.z;
}


template <typename POL, typename IDX>
struct ROCmIndexPair : public POL {
  template <typename IS>
  RAJA_INLINE constexpr ROCmIndexPair(ROCmDim &dims, IS const &is)
      : POL(dims, is)
  {
  }

  typedef IDX INDEX;
};

/** Provides a range from 0 to N_iter - 1
 *
 */
template <typename VIEWDIM, size_t threads_per_block>
struct ROCmThreadBlock {
  RAJA::Index_type distance;

  VIEWDIM view;

  template <typename Iterable>
  ROCmThreadBlock(ROCmDim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  [[hc]] inline RAJA::Index_type operator()(void)
  {
    RAJA::Index_type idx = (RAJA::Index_type)view(blockIdx) * (RAJA::Index_type)threads_per_block + (RAJA::Index_type)view(threadIdx);

    if (idx >= distance) {
      idx = RAJA::operators::limits<RAJA::Index_type>::min();
    }

    return idx;
  }

  void inline setDims(ROCmDim &dims)
  {
    RAJA::Index_type n = distance;
    if (n < threads_per_block) {
      view(dims.num_threads) = n;
      view(dims.num_blocks) = 1;
    } else {
      view(dims.num_threads) = threads_per_block;

      RAJA::Index_type blocks = n / threads_per_block;
      if (n % threads_per_block) {
        ++blocks;
      }
      view(dims.num_blocks) = blocks;
    }
  }
};

/*
 * These execution policies map a loop nest to the block and threads of a
 * given dimension with the number of THREADS per block specifies.
 */

template <size_t THREADS>
using rocm_threadblock_x_exec = ROCmPolicy<ROCmThreadBlock<Dim3x, THREADS>>;

template <size_t THREADS>
using rocm_threadblock_y_exec = ROCmPolicy<ROCmThreadBlock<Dim3y, THREADS>>;

template <size_t THREADS>
using rocm_threadblock_z_exec = ROCmPolicy<ROCmThreadBlock<Dim3z, THREADS>>;

template <typename VIEWDIM>
struct ROCmThread {
  RAJA::Index_type distance;

  VIEWDIM view;

  template <typename Iterable>
  ROCmThread(ROCmDim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  [[hc]] inline RAJA::Index_type operator()(void)
  {
    RAJA::Index_type idx = view(threadIdx);
    if (idx >= distance) {
      return RAJA::operators::limits<RAJA::Index_type>::min();
    }
    return idx;
  }

  void inline setDims(ROCmDim &dims) { view(dims.num_threads) = distance; }
};

/* These execution policies map the given loop nest to the threads in the
   specified dimensions (not blocks)
 */
using rocm_thread_x_exec = ROCmPolicy<ROCmThread<Dim3x>>;

using rocm_thread_y_exec = ROCmPolicy<ROCmThread<Dim3y>>;

using rocm_thread_z_exec = ROCmPolicy<ROCmThread<Dim3z>>;

template <typename VIEWDIM>
struct ROCmBlock {
  RAJA::Index_type distance;

  VIEWDIM view;

  template <typename Iterable>
  ROCmBlock(ROCmDim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  [[hc]] inline RAJA::Index_type operator()(void)
  {
    RAJA::Index_type idx = view(blockIdx);
    if (idx >= distance) {
      return RAJA::operators::limits<RAJA::Index_type>::min();
    }
    return idx;
  }

  void inline setDims(ROCmDim &dims) { view(dims.num_blocks) = distance; }
};

/* These execution policies map the given loop nest to the blocks in the
   specified dimensions (not threads)
 */
using rocm_block_x_exec = ROCmPolicy<ROCmBlock<Dim3x>>;

using rocm_block_y_exec = ROCmPolicy<ROCmBlock<Dim3y>>;

using rocm_block_z_exec = ROCmPolicy<ROCmBlock<Dim3z>>;

}  // closing brace for RAJA namespace

#endif  // RAJA_ENABLE_ROCM
#endif
