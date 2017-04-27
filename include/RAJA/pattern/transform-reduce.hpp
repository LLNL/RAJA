/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA transform-reduce declarations.
*
******************************************************************************
*/

#ifndef RAJA_transform_reduce_HXX
#define RAJA_transform_reduce_HXX

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
#include "RAJA/internal/type_traits.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"

#include <tuple>
#include <type_traits>

namespace RAJA
{
namespace detail
{

template <typename Reducer, typename... Args, size_t... Ids>
void validate_reduce(Reducer&& reducer, VarOps::index_sequence<Ids...>, Args&&...)
{
  using ReducerFn = typename std::decay<decltype(reducer)>::type;
  using ReducerArgs = typename function_traits<ReducerFn>::argument_types;
  constexpr const int NArgs = sizeof...(Args);
  static_assert(NArgs * 2 == std::tuple_size<ReducerArgs>::value,
                "Reducer -- function has unexpected number of arguments");
  static_assert(
      all_of<std::is_reference<
        extract_arg_t<Ids, ReducerFn>>::value...>::value,
      "Reducer -- First N arguments should be reference types");
  static_assert(
      all_of<!std::is_reference<
        extract_arg_t<Ids + NArgs,
          ReducerFn>>::value...>::value,
      "Reducer -- Last N arguments should be value types (optionally const)");
  static_assert(
      all_of<std::is_same<
          typename std::remove_reference<
            extract_arg_t<Ids, ReducerFn>>::type,
          typename std::remove_const<
            extract_arg_t<Ids + NArgs, ReducerFn>>::type>::value...>::value,
      "Reducer -- Argument type mismatch: (i) and (i + N) should have same type");
}

template <typename ExecPolicy,
          typename Iterable,
          typename Transformer,
          typename Reducer,
          typename... Args>
void transform_reduce_invoke(Iterable&& iterable,
                             Transformer&& transformer,
                             Reducer&& reducer,
                             Args&&... args)
{
  using TransformerReturn = decltype(transformer(iterable.begin()));
  static_assert(
      detail::is_specialization_of<TransformerReturn, std::tuple>::value,
      "Transformer -- Callable should return a tuple of r-values");
  static_assert(
      std::is_same<TransformerReturn,
                   std::tuple<typename std::remove_reference<Args>::type...>>::value,
      "Transformer -- Output parameters mismatch transformer return");
  validate_reduce(VarOps::forward(reducer),
                  VarOps::index_sequence_for<Args...>{},
                  VarOps::forward(args)...);
  transform_reduce(ExecPolicy{},
                   VarOps::forward(iterable),
                   VarOps::forward(transformer),
                   VarOps::forward(reducer),
                   VarOps::forward(args)...);
}

template <typename ExecPolicy,
          typename Iterable,
          typename TupleArgs,
          size_t... First,
          size_t... Last>
void transform_reduce(Iterable&& iterable,
                      TupleArgs&& args,
                      VarOps::index_sequence<First...>,
                      VarOps::index_sequence<Last...>)
{
  transform_reduce_invoke<ExecPolicy>(VarOps::forward(iterable),
                                      VarOps::forward(std::get<Last>(args))...,
                                      VarOps::forward(std::get<First>(args))...);
}
}

template <typename ExecPolicy, typename Iterable, typename... Args>
void transform_reduce(Iterable&& iter, Args&&... args)
{
  static_assert(detail::is_exec_policy<ExecPolicy>::value,
                "ExecPolicy parameter does not model ExecutionPolicy");
  static_assert(detail::is_iterable<Iterable>::value,
                "Iterable parameter does not model Iterable");
  constexpr const size_t Nminus2 = sizeof...(Args) - 2;
  constexpr const size_t Nminus1 = sizeof...(Args) - 1;
  detail::transform_reduce<ExecPolicy>(VarOps::forward<Iterable>(iter),
                                       std::forward_as_tuple(args...),
                                       VarOps::make_index_sequence<Nminus2>(),
                                       VarOps::index_sequence<Nminus2, Nminus1>());
}

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
