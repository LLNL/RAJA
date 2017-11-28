#ifndef CAMP_LAMBDA_HPP
#define CAMP_LAMBDA_HPP

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

#include <type_traits>

#include "camp/defines.hpp"
#include "camp/list/at.hpp"
#include "camp/list/list.hpp"


namespace camp
{

template <template <typename...> class Expr>
struct lambda {
  template <typename... Ts>
  using expr = typename Expr<Ts...>::type;
};

template <typename Lambda, typename Seq>
struct apply_l;
template <typename Lambda, typename... Args>
struct apply_l<Lambda, list<Args...>> {
  using type = typename Lambda::template expr<Args...>::type;
};

template <typename Lambda, typename... Args>
struct invoke_l {
  using type = typename Lambda::template expr<Args...>::type;
};

template <idx_t n>
struct arg {
  template <typename... Ts>
  using expr = typename at<list<Ts...>, num<n - 1>>::type;
};

using _1 = arg<1>;
using _2 = arg<2>;
using _3 = arg<3>;
using _4 = arg<4>;
using _5 = arg<5>;
using _6 = arg<6>;
using _7 = arg<7>;
using _8 = arg<8>;
using _9 = arg<9>;

namespace detail
{
  template <typename T, typename... Args>
  struct get_bound_arg {
    using type = T;
  };
  template <idx_t i, typename... Args>
  struct get_bound_arg<arg<i>, Args...> {
    using type = typename arg<i>::template expr<Args...>;
  };
}

template <template <typename...> class Expr, typename... ArgBindings>
struct bind {
  using bindings = list<ArgBindings...>;
  template <typename... Ts>
  using expr = typename Expr<
      typename detail::get_bound_arg<ArgBindings, Ts...>::type...>::type;
  using type = bind;
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((invoke_l<bind<list, _1, int, _2>, float, double>),
              (list<float, int, double>));
}
#endif

template <template <typename...> class Expr, typename... BoundArgs>
struct bind_front {
  template <typename... Ts>
  using expr = typename Expr<BoundArgs..., Ts...>::type;
  using type = bind_front;
};

CAMP_MAKE_L(bind_front);

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((invoke_l<bind_front<list, int>, float, double>),
              (list<int, float, double>));
}
#endif

}  // end namespace camp

#endif /* CAMP_LAMBDA_HPP */
