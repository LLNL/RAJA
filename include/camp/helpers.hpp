#ifndef CAMP_HELPERS_HPP
#define CAMP_HELPERS_HPP

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
// For additional details, please also read RAJA/README.
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

#include <cstddef>
#include <iterator>
#include <utility>

#include "camp/defines.hpp"

namespace camp
{

/// metafunction to get instance of pointer type
template <typename T>
T* declptr();

/// metafunction to get instance of value type
template <typename T>
auto val() noexcept -> decltype(std::declval<T>());

/// metafunction to get instance of const type
template <typename T>
auto cval() noexcept -> decltype(std::declval<T const>());

/// metafunction to expand a parameter pack and ignore result
template <typename... Ts>
CAMP_HOST_DEVICE void sink(Ts...)
{
}


// bring common utility routines into scope to allow ADL
using std::begin;
using std::swap;

namespace type
{
  namespace ref
  {
    template <class T>
    struct rem_s {
      using type = T;
    };
    template <class T>
    struct rem_s<T&> {
      using type = T;
    };
    template <class T>
    struct rem_s<T&&> {
      using type = T;
    };

    /// remove reference from T
    template <class T>
    using rem = typename rem_s<T>::type;

    /// add remove reference to T
    template <class T>
    using add = T&;
  }  // end namespace ref

  namespace rvref
  {
    /// add rvalue reference to T
    template <class T>
    using add = T&&;
  }  // end namespace rvref

  namespace c
  {
    template <class T>
    struct rem_s {
      using type = T;
    };
    template <class T>
    struct rem_s<const T> {
      using type = T;
    };

    /// remove const qualifier from T
    template <class T>
    using rem = typename rem_s<T>::type;

    /// add const qualifier to T
    template <class T>
    using add = const T;
  }  // end namespace ref

  namespace v
  {
    template <class T>
    struct rem_s {
      using type = T;
    };
    template <class T>
    struct rem_s<volatile T> {
      using type = T;
    };

    /// remove volatile qualifier from T
    template <class T>
    using rem = typename rem_s<T>::type;

    /// add volatile qualifier to T
    template <class T>
    using add = volatile T;
  }  // end namespace ref

  namespace cv
  {
    template <class T>
    struct rem_s {
      using type = T;
    };
    template <class T>
    struct rem_s<const T> {
      using type = T;
    };
    template <class T>
    struct rem_s<volatile T> {
      using type = T;
    };
    template <class T>
    struct rem_s<const volatile T> {
      using type = T;
    };

    /// remove const and volatile qualifiers from T
    template <class T>
    using rem = typename rem_s<T>::type;

    /// add const and volatile qualifiers to T
    template <class T>
    using add = const volatile T;
  }  // end namespace ref
}  // end namespace type

template <typename T>
using decay = type::cv::rem<type::ref::rem<T>>;

template <typename T>
using plain = type::ref::rem<T>;

template <typename T>
using diff_from = decltype(val<plain<T>>() - val<plain<T>>());
template <typename T, typename U>
using diff_between = decltype(val<plain<T>>() - val<plain<U>>());

template <typename T>
using iterator_from = decltype(begin(val<plain<T>>()));

template <class T>
CAMP_HOST_DEVICE constexpr T&& forward(type::ref::rem<T>& t) noexcept
{
  return static_cast<T&&>(t);
}
template <class T>
CAMP_HOST_DEVICE constexpr T&& forward(type::ref::rem<T>&& t) noexcept
{
  return static_cast<T&&>(t);
}

template <typename T>
CAMP_HOST_DEVICE void safe_swap(T& t1, T& t2)
{
#if defined(__CUDA_ARCH__)
  T temp{std::move(t1)};
  t1 = std::move(t2);
  t2 = std::move(temp);
#else
  using std::swap;
  swap(t1, t2);
#endif
}

template <typename T, typename = decltype(sink(swap(val<T>(), val<T>())))>
CAMP_HOST_DEVICE void safe_swap(T& t1, T& t2)
{
  using std::swap;
  swap(t1, t2);
}
}

#endif /* CAMP_HELPERS_HPP */
