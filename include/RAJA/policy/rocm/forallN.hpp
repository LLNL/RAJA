/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
 *          traversals on GPU with ROCm.
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

#ifndef RAJA_forallN_rocm_HPP
#define RAJA_forallN_rocm_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_ROCM)

#include <cassert>
#include <climits>

#include "RAJA/util/types.hpp"

#include "RAJA/policy/rocm/MemUtils_ROCm.hpp"
#include "RAJA/policy/rocm/policy.hpp"

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{

/*!
 * \brief Functor that binds the first argument of a callable.
 *
 * This version has host-device constructor and device-only operator.
 */
template <typename BODY>
struct ForallN_BindFirstArg_Device {
  BODY const &body;
  size_t i;

  RAJA_INLINE
  RAJA_DEVICE
  constexpr ForallN_BindFirstArg_Device(BODY &b, size_t i0) : body(b), i(i0) {}

  template <typename... ARGS>
  RAJA_INLINE RAJA_DEVICE void operator()(ARGS... args) const
  {
    body(i, args...);
  }
};


template <typename ROCM_EXEC, typename Iterator>
struct ROCmIterableWrapper {
  ROCM_EXEC pol_;
  Iterator i_;
  constexpr ROCmIterableWrapper(const ROCM_EXEC &pol, const Iterator &i)
      : pol_(pol), i_(i)
  {
  }

  inline decltype(i_[0]) operator() [[hc]] ()
  {
    auto val = pol_();
    return val > INT_MIN ? i_[pol_()] : INT_MIN;
  }
};

template <typename ROCM_EXEC, typename Iterator>
auto make_rocm_iter_wrapper(const ROCM_EXEC &pol, const Iterator &i)
    -> ROCmIterableWrapper<ROCM_EXEC, Iterator>
{
  return ROCmIterableWrapper<ROCM_EXEC, Iterator>(pol, i);
}

/*!
 * \brief  Function to check indices for out-of-bounds
 *
 */
template <typename BODY, typename... ARGS>
RAJA_INLINE void rocmCheckBounds(BODY &body, int i, ARGS... args) [[hc]]
{
  if (i > INT_MIN) {
    ForallN_BindFirstArg_Device<BODY> bound(body, i);
    rocmCheckBounds(bound, args...);
  }
}

template <typename BODY>
RAJA_INLINE void rocmCheckBounds(BODY &body, int i) [[hc]]
{
  if (i > INT_MIN) {
    body(i);
  }
}

/*!
 * \brief Launcher that uses execution policies to map blockIdx and threadIdx to
 * map
 * to N-argument function
 */
template <typename BODY, typename... RARGS>
void rocmLauncherN(BODY loop_body, RARGS... rargs) [[hc]]
{
  // force reduction object copy constructors and destructors to run
  auto body = loop_body;

  // Compute indices and then pass through the bounds-checking mechanism
  rocmCheckBounds(body, (rargs())...);
}

template <bool device,
          typename RoARG0,
          typename ISET0,
          typename RoARG1,
          typename ISET1,
          typename... RoARGS,
          typename... ISETS>
struct ForallN_Executor<device,
                        ForallN_PolicyPair<ROCmPolicy<RoARG0>, ISET0>,
                        ForallN_PolicyPair<ROCmPolicy<RoARG1>, ISET1>,
                        ForallN_PolicyPair<ROCmPolicy<RoARGS>, ISETS>...> {
  ForallN_PolicyPair<ROCmPolicy<RoARG0>, ISET0> iset0;
  ForallN_PolicyPair<ROCmPolicy<RoARG1>, ISET1> iset1;
  std::tuple<ForallN_PolicyPair<ROCmPolicy<RoARGS>, ISETS>...> isets;

  ForallN_Executor(
      ForallN_PolicyPair<ROCmPolicy<RoARG0>, ISET0> const &iset0_,
      ForallN_PolicyPair<ROCmPolicy<RoARG1>, ISET1> const &iset1_,
      ForallN_PolicyPair<ROCmPolicy<RoARGS>, ISETS> const &... isets_)
      : iset0(iset0_), iset1(iset1_), isets(isets_...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    unpackIndexSets(body, VarOps::make_index_sequence<sizeof...(RoARGS)>{});
  }

  template <typename BODY, size_t... N>
  RAJA_INLINE void unpackIndexSets(BODY body,
                                   VarOps::index_sequence<N...>) const
  {
    ROCmDim dims;

    callLauncher(dims,
                 body,
                 make_rocm_iter_wrapper(RoARG0(dims, iset0), std::begin(iset0)),
                 make_rocm_iter_wrapper(RoARG1(dims, iset1), std::begin(iset1)),
                 make_rocm_iter_wrapper(RoARGS(dims, std::get<N>(isets)),
                                        std::begin(std::get<N>(isets)))...);
  }

  template <typename BODY, typename... RARGS>
  RAJA_INLINE void callLauncher(ROCmDim const &dims,
                                BODY loop_body,
                                RARGS const &... rargs) const
  {
    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {

      bool Async = true;
      rocmStream_t stream = 0;

//      auto ext = hc::extent<3>(dims.num_blocks,dims.num_threads,1)
//       .tile(dims.num_threads.x,dims.num_threads.y,dims.num_threads.z);
      auto ext = 
        hc::extent<3>( dims.num_blocks.x,dims.num_blocks.y,dims.num_blocks.z)
              .tile(dims.num_threads.x,dims.num_threads.y,dims.num_threads.z);
      auto fut = hc::parallel_for_each(ext,
                                       [=](const hc::tiled_index<3> & idx)
                                       [[hc]]{
      rocmLauncherN(
          RAJA::rocm::make_launch_body(dims.num_blocks,
                                       dims.num_threads,
                                       0,
                                       stream,
                                       std::move(loop_body)),
          rargs...);
      }).wait();

      RAJA::rocm::launch(stream);
      if (!Async) RAJA::rocm::synchronize(stream);
    }
  }
};

template <bool device, typename RoARG0, typename ISET0>
struct ForallN_Executor<device, ForallN_PolicyPair<ROCmPolicy<RoARG0>, ISET0>> {
  ISET0 iset0;

  ForallN_Executor(ForallN_PolicyPair<ROCmPolicy<RoARG0>, ISET0> const &iset0_)
      : iset0(iset0_)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY loop_body) const
  {
    ROCmDim dims;
    auto r0 = make_rocm_iter_wrapper(RoARG0(dims, iset0), std::begin(iset0));

    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {

      bool Async = true;
      rocmStream_t stream = 0;

      auto ext = 
        hc::extent<3>( dims.num_blocks.x,dims.num_blocks.y,dims.num_blocks.z)
              .tile(dims.num_threads.x,dims.num_threads.y,dims.num_threads.z);
      auto fut = hc::parallel_for_each(ext,
                                       [=](const hc::tiled_index<3> & idx)
                                       [[hc]]{
      rocmLauncherN(
          RAJA::rocm::make_launch_body(dims.num_blocks,
                                       dims.num_threads,
                                       0,
                                       stream,
                                       std::move(loop_body)),
          r0);
      }).wait();

      RAJA::rocm::launch(stream);
      if (!Async) RAJA::rocm::synchronize(stream);
    }
  }
};

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_ROCM guard

#endif  // closing endif for header file include guard
