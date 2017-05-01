/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA transform-reduce declarations.
*
******************************************************************************
*/

#ifndef RAJA_transform_reduce_openmp_HXX
#define RAJA_transform_reduce_openmp_HXX

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
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

#include "RAJA/config.hpp"

#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/defines.hpp"

#include "RAJA/policy/sequential/transform-reduce.hpp"

#include <tuple>

namespace RAJA
{
namespace detail
{

template <typename Iterable,
          typename Transformer,
          typename Reducer,
          typename... Args>
void transform_reduce(RAJA::omp_parallel_for_exec,
                      Iterable&& iterable,
                      Transformer&& transformer,
                      Reducer&& reducer,
                      Args&&... args)
{
  auto begin = iterable.begin();
  auto end = iterable.end();
  #pragma omp parallel
  {
    std::tuple<Args...> local;

    #pragma omp for
    for (auto i = begin; i < end; ++i) {
      reduce(RAJA::seq_exec{},
             VarOps::forward(reducer),
             local,
             VarOps::forward(transformer(i)),
             VarOps::index_sequence_for<Args...>());
    }

    #pragma omp critical(omp_transform_reduce_aggregate)
    reduce(RAJA::seq_exec{},
           VarOps::forward(reducer),
           std::tie(args...),
           VarOps::forward(local),
           VarOps::index_sequence_for<Args...>());
  }
}

}  // namespace detail

}  // namespace RAJA

#endif
