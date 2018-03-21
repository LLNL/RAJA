/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_forallN_cuda_HPP
#define RAJA_forallN_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "RAJA/util/types.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

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


template <typename CUDA_EXEC, typename Iterator>
struct CudaIterableWrapper {
  CUDA_EXEC pol_;
  Iterator i_;
  constexpr CudaIterableWrapper(const CUDA_EXEC &pol, const Iterator &i)
      : pol_(pol), i_(i)
  {
  }

  __device__ inline decltype(i_[0]) operator()()
  {
    auto val = pol_();
    return val > INT_MIN ? i_[pol_()] : INT_MIN;
  }
};

template <typename CUDA_EXEC, typename Iterator>
auto make_cuda_iter_wrapper(const CUDA_EXEC &pol, const Iterator &i)
    -> CudaIterableWrapper<CUDA_EXEC, Iterator>
{
  return CudaIterableWrapper<CUDA_EXEC, Iterator>(pol, i);
}

/*!
 * \brief  Function to check indices for out-of-bounds
 *
 */
template <typename BODY, typename... ARGS>
RAJA_INLINE __device__ void cudaCheckBounds(BODY &body, int i, ARGS... args)
{
  if (i > INT_MIN) {
    ForallN_BindFirstArg_Device<BODY> bound(body, i);
    cudaCheckBounds(bound, args...);
  }
}

template <typename BODY>
RAJA_INLINE __device__ void cudaCheckBounds(BODY &body, int i)
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
template <typename BODY, typename... CARGS>
__global__ void cudaLauncherN(BODY loop_body, CARGS... cargs)
{
  // force reduction object copy constructors and destructors to run
  auto body = loop_body;

  // Compute indices and then pass through the bounds-checking mechanism
  cudaCheckBounds(body, (cargs())...);
}

template <bool device,
          typename CuARG0,
          typename ISET0,
          typename CuARG1,
          typename ISET1,
          typename... CuARGS,
          typename... ISETS>
struct ForallN_Executor<device,
                        ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0>,
                        ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1>,
                        ForallN_PolicyPair<CudaPolicy<CuARGS>, ISETS>...> {
  ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> iset0;
  ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1> iset1;
  std::tuple<ForallN_PolicyPair<CudaPolicy<CuARGS>, ISETS>...> isets;

  ForallN_Executor(
      ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> const &iset0_,
      ForallN_PolicyPair<CudaPolicy<CuARG1>, ISET1> const &iset1_,
      ForallN_PolicyPair<CudaPolicy<CuARGS>, ISETS> const &... isets_)
      : iset0(iset0_), iset1(iset1_), isets(isets_...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    unpackIndexSets(body, VarOps::make_index_sequence<sizeof...(CuARGS)>{});
  }

  template <typename BODY, size_t... N>
  RAJA_INLINE void unpackIndexSets(BODY body,
                                   VarOps::index_sequence<N...>) const
  {
    CudaDim dims;

    callLauncher(dims,
                 body,
                 make_cuda_iter_wrapper(CuARG0(dims, iset0), std::begin(iset0)),
                 make_cuda_iter_wrapper(CuARG1(dims, iset1), std::begin(iset1)),
                 make_cuda_iter_wrapper(CuARGS(dims, std::get<N>(isets)),
                                        std::begin(std::get<N>(isets)))...);
  }

  template <typename BODY, typename... CARGS>
  RAJA_INLINE void callLauncher(CudaDim const &dims,
                                BODY loop_body,
                                CARGS const &... cargs) const
  {
    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {

      bool Async = true;
      cudaStream_t stream = 0;

      cudaLauncherN<<<dims.num_blocks, dims.num_threads, 0, stream>>>(
          RAJA::cuda::make_launch_body(dims.num_blocks,
                                       dims.num_threads,
                                       0,
                                       stream,
                                       std::move(loop_body)),
          cargs...);
      RAJA::cuda::peekAtLastError();

      RAJA::cuda::launch(stream);
      if (!Async) RAJA::cuda::synchronize(stream);
    }
  }
};

template <bool device, typename CuARG0, typename ISET0>
struct ForallN_Executor<device, ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0>> {
  ISET0 iset0;

  ForallN_Executor(ForallN_PolicyPair<CudaPolicy<CuARG0>, ISET0> const &iset0_)
      : iset0(iset0_)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY loop_body) const
  {
    CudaDim dims;
    auto c0 = make_cuda_iter_wrapper(CuARG0(dims, iset0), std::begin(iset0));

    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {

      bool Async = true;
      cudaStream_t stream = 0;

      cudaLauncherN<<<dims.num_blocks, dims.num_threads, 0, stream>>>(
          RAJA::cuda::make_launch_body(dims.num_blocks,
                                       dims.num_threads,
                                       0,
                                       stream,
                                       std::move(loop_body)),
          c0);
      RAJA::cuda::peekAtLastError();

      RAJA::cuda::launch(stream);
      if (!Async) RAJA::cuda::synchronize(stream);
    }
  }
};

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
