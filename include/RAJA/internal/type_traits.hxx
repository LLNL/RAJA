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

template <size_t I, typename Fn>
using extract_arg_t = typename extract_arg<I, Fn>::type;

template <typename Exec>
struct is_exec_policy
  : public std::integral_constant<
      bool, std::is_base_of<RAJA::PolicyBase, Exec>::value> {
};

template <typename T, typename = void>
struct is_iterable : std::false_type {
};

template <typename... Ts>
struct is_iterable_helper {
};

template <typename T>
struct is_iterable<
  T,
  typename std::conditional<
    false,
    is_iterable_helper<
      decltype(std::declval<T>().begin()),
      decltype(std::declval<T>().end())>,
    void>::type> : public std::true_type {
};

template <typename T, template <typename...> class Template>
struct is_specialization_of : std::false_type {
};

template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template<Args...>, Template> : std::true_type {
};

template <bool...>
struct bool_list;

template <bool... V>
struct all_of
  : public std::integral_constant<
      bool, std::is_same<bool_list<true, V...>,
                         bool_list<V..., true>>::value> {
};

}  // closing brace for detail namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
