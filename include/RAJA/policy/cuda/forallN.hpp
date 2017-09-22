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
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
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


RAJA_INLINE
constexpr int numBlocks(CudaDim const &dim)
{
  return dim.num_blocks.x * dim.num_blocks.y * dim.num_blocks.z;
}

RAJA_INLINE
constexpr int numThreads(CudaDim const &dim)
{
  return dim.num_threads.x * dim.num_threads.y * dim.num_threads.z;
}

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
                                BODY body,
                                CARGS const &... cargs) const
  {
    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {
      cudaLauncherN<<<RAJA_CUDA_LAUNCH_PARAMS(dims.num_blocks,
                                              dims.num_threads)>>>(body,
                                                                   cargs...);
    }

    RAJA_CUDA_CHECK_AND_SYNC(true);
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
  RAJA_INLINE void operator()(BODY body) const
  {
    CudaDim dims;
    auto c0 = make_cuda_iter_wrapper(CuARG0(dims, iset0), std::begin(iset0));

    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {
      cudaLauncherN<<<RAJA_CUDA_LAUNCH_PARAMS(dims.num_blocks,
                                              dims.num_threads)>>>(body, c0);
    }

    RAJA_CUDA_CHECK_AND_SYNC(true);
  }
};

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
