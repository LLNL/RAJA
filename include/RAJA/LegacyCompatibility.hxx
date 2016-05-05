#ifndef KRIPKE_LEGAGY_COMPATIBILITY_HXX
#define KRIPKE_LEGAGY_COMPATIBILITY_HXX 1

static_assert(__cplusplus >= 201103L, "C++ standards below 2011 are not supported");

#include <type_traits>
#include <utility>
#include <iostream>
#include <functional>
#include <cstdint>

namespace VarOps {

// Basics, using c++14 semantics in a c++11 compatible way, credit to libc++
template <class T> struct remove_reference {
    typedef T type;
};
template <class T> struct remove_reference<T&> {
    typedef T type;
};
template <class T> struct remove_reference<T&&> {
    typedef T type;
};
template <class T>
constexpr T&& forward(typename remove_reference<T>::type& t) noexcept{
    return static_cast<T&&>(t);
}
template <class T>
constexpr T&& forward(typename remove_reference<T>::type&& t) noexcept{
    return static_cast<T&&>(t);
}

// FoldL
template<typename Op, typename ...Rest>
struct foldl_impl;

template<typename Op, typename Arg1>
struct foldl_impl<Op, Arg1> {
    using Ret = Arg1;
};

template<typename Op, typename Arg1, typename Arg2>
struct foldl_impl<Op, Arg1, Arg2> {
    using Ret = typename std::result_of<Op(Arg1, Arg2)>::type;
};

template<typename Op, typename Arg1, typename Arg2, typename Arg3, typename ...Rest>
struct foldl_impl<Op, Arg1, Arg2, Arg3, Rest...> {
    using Ret = typename
        foldl_impl<Op, typename std::result_of<Op(typename std::result_of<Op(Arg1, Arg2)>::type, Arg3)>::type, Rest...>::Ret;
};

template <typename Op, typename Arg1>
constexpr auto foldl(Op&& operation, Arg1&& arg) -> typename foldl_impl<Op, Arg1>::Ret
{
    return forward<Arg1&&>(arg);
}

template <typename Op, typename Arg1, typename Arg2>
constexpr auto foldl(Op&& operation, Arg1&& arg1, Arg2&& arg2) ->  typename foldl_impl<Op, Arg1, Arg2>::Ret
{
    return forward<Op&&>(operation)(
        forward<Arg1&&>(arg1), forward<Arg2&&>(arg2));
}

template <typename Op, typename Arg1, typename Arg2, typename Arg3,
    typename... Rest>
constexpr auto foldl(
    Op&& operation, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Rest&&... rest) ->  typename foldl_impl<Op, Arg1, Arg2, Arg3, Rest...>::Ret
{
    return foldl(forward<Op&&>(operation),
        forward<Op&&>(operation)(forward<Op&&>(operation)(
                                     forward<Arg1&&>(arg1), forward<Arg2&&>(
                                         arg2)),
                             forward<Arg3&&>(arg3)),
        forward<Rest&&>(rest)...);
}

// Convenience folds
template<typename Result, typename ... Args>
Result sum(Args...args)
{
    return foldl([](Result l, Result r){ return l + r; }, args...);
}

// template<typename Result, size_t N>
// struct product_first_n;
//
// template<typename Result>
// struct product_first_n<Result, 0>{
//     static Result value = 1;
//     template<typename ... Args>
//     constexpr product_first_n(Args...args) : value{1} { }
// };
//
// template<typename Result, size_t N>
// struct product_first_n{
//     static Result value = product_first_n<Result, N-1>(args...)::value;
//     template<typename FirstArg, typename ... Args>
//     constexpr product_first_n(FirstArg arg1, Args...args)
//     : value() { }
// };

// Index sequence

template<size_t... Ints>
struct integer_sequence{
    using type = integer_sequence;
    static constexpr size_t size = sizeof...(Ints);
};

template<template <class...> class Seq, class First, class ...Ints>
constexpr auto rotate_left_one(const Seq<First, Ints...>) -> Seq<Ints..., First>
{
    return Seq<Ints..., First>{};
}

template<size_t... Ints>
constexpr size_t integer_sequence<Ints...>::size;

namespace integer_sequence_detail {
// using aliases for cleaner syntax
template<class T> using Invoke = typename T::type;

template<typename T, class S1, class S2> struct concat;

template<typename T, T... I1, T... I2>
struct concat<T, integer_sequence<I1...>, integer_sequence<I2...>>
  : integer_sequence<I1..., (sizeof...(I1)+I2)...>{};

template<typename T, class S1, class S2>
using Concat = Invoke<concat<T, S1, S2>>;

template<size_t N> struct gen_seq;
template<size_t N> using GenSeq = Invoke<gen_seq<N>>;

template<size_t N>
struct gen_seq : integer_sequence_detail::Concat<size_t,
                 integer_sequence_detail::GenSeq<N/2>,
                 integer_sequence_detail::GenSeq<N - N/2>>{};

template<> struct gen_seq<0> : integer_sequence<>{};
template<> struct gen_seq<1> : integer_sequence<0>{};

}

template<size_t Upper>
using make_index_sequence = typename integer_sequence_detail::gen_seq<Upper>::type;

template<size_t... Ints>
using index_sequence = integer_sequence<Ints...>;

// Invoke

template<typename Fn, size_t ...Sequence, typename TupleLike>
constexpr auto invoke_with_order(TupleLike&& t, Fn&& f, index_sequence<Sequence...>) -> decltype(f(std::get<Sequence>(t)...)) {
    return f(std::get<Sequence>(t)...);
}

template<typename Fn, typename TupleLike>
constexpr auto invoke(TupleLike&& t, Fn&& f)
    -> decltype(invoke_with_order(t, f, make_index_sequence<std::tuple_size<TupleLike>::value>{})) {
    return invoke_with_order(t, f, make_index_sequence<std::tuple_size<TupleLike>::value>{});
}

// Ignore helper
template<typename ... Args>
void ignore_args(Args...) {}

// Assign

template<size_t ... To,
         size_t ... From,
         typename ToT,
         typename FromT>
void assign(ToT dst, FromT src, index_sequence<To...>, index_sequence<From...>) {
    ignore_args((dst[To] = src[From])...);
}

template<size_t ... To,
         typename ToT,
         typename... Args>
void assign_args(ToT dst, index_sequence<To...>, Args...args) {
    ignore_args((dst[To] = args)...);
}

}

#endif
