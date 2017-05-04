/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA-specific type traits
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

#ifndef RAJA_type_traits_HXX
#define RAJA_type_traits_HXX

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

#include <tuple>
#include <type_traits>
#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{
namespace detail
{

template <typename T>
struct function_traits : function_traits<decltype(&T::operator())> {
};

// free function
template <typename R, typename... Args>
struct function_traits<R(Args...)> {
  using result_type = R;
  using argument_types = std::tuple<Args...>;
};

// pointer to function
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
  using result_type = R;
  using argument_types = std::tuple<Args...>;
};

// member function
template <typename T, typename R, typename... Args>
struct function_traits<R (T::*)(Args...)> {
  using result_type = R;
  using argument_types = std::tuple<Args...>;
};

// const member function
template <typename T, typename R, typename... Args>
struct function_traits<R (T::*)(Args...) const> {
  using result_type = R;
  using argument_types = std::tuple<Args...>;
};

// extract 0-based argument type
template <size_t I, typename Fn>
using extract_arg =
    std::tuple_element<I, typename function_traits<Fn>::argument_types>;

}  // closing brace for detail namespace

template <size_t I, typename Fn>
using extract_arg_t = typename detail::extract_arg<I, Fn>::type;

template <typename T>
struct is_wrapper_policy : public std::false_type {
};

template <template <typename...> class Outer, typename... Tags>
struct is_wrapper_policy<Outer<Tags...>>
    : public std::integral_constant<bool,
                                    std::is_same<WrapperPolicy<Tags...>,
                                                 Outer<Tags...>>::value> {
};

template <typename T>
struct is_policy
    : public std::integral_constant<bool,
                                    is_wrapper_policy<T>::value
                                        || std::is_base_of<T,
                                                           PolicyBase>::value> {
};

namespace detail
{
template <typename T, T A, T B>
struct is_enum_same : public std::false_type {
};
template <typename T, T A>
struct is_enum_same<T, A, A> : public std::true_type {
};

template <typename P_, Policy P>
struct models_policy : public std::integral_constant<bool, is_enum_same<Policy, P_::policy, P>::value> {
};

template <typename P_, Launch L>
struct models_launch : public std::integral_constant<bool, is_enum_same<Launch, P_::launch, L>::value> {
};

template <typename P_, Pattern P>
struct models_pattern : public std::integral_constant<bool, is_enum_same<Pattern, P_::pattern, P>::value> {
};
}

template <typename P>
struct is_sequential_policy
    : public detail::models_policy<P, Policy::sequential> {
};
template <typename P>
struct is_simd_policy : public detail::models_policy<P, Policy::simd> {
};
template <typename P>
struct is_openmp_policy : public detail::models_policy<P, Policy::openmp> {
};
template <typename P>
struct is_cuda_policy : public detail::models_policy<P, Policy::cuda> {
};
template <typename P>
struct is_cilk_policy : public detail::models_policy<P, Policy::cilk> {
};

template <typename L>
struct is_sync_launch : public detail::models_launch<L, Launch::sync> {
};
template <typename L>
struct is_async_launch : public detail::models_launch<L, Launch::async> {
};

template <typename P>
struct is_forall_pattern : public detail::models_pattern<P, Pattern::forall> {
};
template <typename P>
struct is_reduce_pattern : public detail::models_pattern<P, Pattern::reduce> {
};

template <Policy P>
struct forall_for : public make_policy_pattern<P, Pattern::forall> {
};
template <Policy P>
struct reduce_for : public make_policy_pattern<P, Pattern::reduce> {
};

template <Pattern P>
struct sequential_for : public make_policy_pattern<Policy::sequential, P> {
};
template <Pattern P>
struct simd_for : public make_policy_pattern<Policy::simd, P> {
};
template <Pattern P>
struct openmp_for : public make_policy_pattern<Policy::openmp, P> {
};
template <Pattern P>
struct cuda_for : public make_policy_pattern<Policy::cuda, P> {
};
template <Pattern P>
struct cuda_async_for
    : public make_policy_launch_pattern<Policy::cuda, Launch::async, P> {
};
template <Pattern P>
struct cilk_for : public make_policy_pattern<Policy::cilk, P> {
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
