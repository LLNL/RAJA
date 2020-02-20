/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA sort templates.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_sort_HPP
#define RAJA_util_sort_HPP

#include "RAJA/config.hpp"

#include <iterator>

#include "RAJA/pattern/detail/algorithm.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

namespace RAJA
{

/*!
    \brief swap values lhs and rhs
*/
template <typename T>
RAJA_HOST_DEVICE RAJA_INLINE
void
swap(T& lhs, T& rhs)
{
  T tmp = std::move(lhs);
  lhs = std::move(rhs);
  rhs = std::move(tmp);
}

/*!
    \brief swap values at iterators lhs and rhs
*/
template <typename Iter>
RAJA_HOST_DEVICE RAJA_INLINE
void
iter_swap(Iter lhs, Iter rhs)
{
  using RAJA::swap;
  swap(*lhs, *rhs);
}

/*!
    \brief returns iterator to next item
*/
template <typename Iter>
RAJA_HOST_DEVICE RAJA_INLINE
Iter
next(Iter it)
{
  ++it;
  return it;
}

/*!
    \brief returns iterator to next item
*/
template <typename Iter>
RAJA_HOST_DEVICE RAJA_INLINE
Iter
prev(Iter it)
{
  --it;
  return it;
}

namespace detail
{

/*!
    \brief evaluate log base 2 of N rounded down to the nearest integer >= 0
*/
RAJA_HOST_DEVICE RAJA_INLINE
unsigned
ulog2(size_t N)
{
  unsigned val = 0;

  while (N > 1) {
    val += 1;
    N >>= 1;
  }

  return val;
}

/*!
    \brief unstable partition given range inplace using predicate function
    and using O(N) predicate evaluations and O(1) memory
*/
template <typename Iter, typename Predicate>
RAJA_HOST_DEVICE RAJA_INLINE
Iter
partition(Iter begin,
          Iter end,
          Predicate pred)
{
  using ::RAJA::iter_swap;

  if (begin == end) {
    return;
  }

  // advance to first false
  Iter first_false = begin;
  for (; first_false != end; ++first_false) {

    if (!pred(*first_false)) {
      break;
    }
  }

  // return if none were false
  if (first_false == end) {
    return first_false;
  }

  // advance through rest of list to find the next true
  for (Iter next_true = RAJA::next(first_false); next_true != end; ++next_true) {

    // find the end of a range of falses [first_false, next_true)
    if (pred(*next_true)) {

      // shift the known range of falses forward
      // by swapping the true to the beginning of the range
      iter_swap(first_false, next_true);
      ++first_false;
    }
  }

  return first_false;
}

/*!
    \brief stable insertion sort given range inplace using comparison function
    and using O(N^2) comparisons and O(1) memory
*/
template <typename Iter, typename Compare>
RAJA_HOST_DEVICE RAJA_INLINE
void
insertion_sort(Iter begin,
               Iter end,
               Compare comp)
{
  using ::RAJA::iter_swap;

  if (begin == end) {
    return;
  }

  // for each unsorted item in the right side of the range
  for (Iter next_unsorted = RAJA::next(begin); next_unsorted != end; ++next_unsorted) {

    // insert unsorted item into the sorted left side of the range
    for (Iter to_insert = next_unsorted; to_insert != begin; --to_insert) {

      Iter next_sorted = RAJA::prev(to_insert);

      // compare with next item to left
      if (comp(*to_insert, *next_sorted)) {

        // swap down if should be before
        iter_swap(next_sorted, to_insert);

      } else {

        // stop if in correct position
        break;
      }
    }
  }
}

/*!
    \brief unstable heap sort given range inplace using comparison function
    and using O(N*lg(N)) comparisons and O(1) memory
*/
template <typename Iter, typename Compare>
RAJA_HOST_DEVICE RAJA_INLINE
void
heap_sort(Iter begin,
          Iter end,
          Compare comp)
{
  static_assert(!type_traits::is_iterator<Iter>::value, "unimplemented");
}

/*!
    \brief unstable intro sort given range inplace using comparison function
    and using O(N*lg(N)) comparisons and O(lg(N)) memory
*/
template <typename Iter, typename Compare>
RAJA_HOST_DEVICE RAJA_INLINE
void
intro_sort(Iter begin,
           Iter end,
           Compare comp,
           unsigned depth)
{
  auto N = end - begin;

  // cutoff to use insertion sort
  static constexpr camp::decay<decltype(N)> insertion_sort_cutoff = 16;

  if (N < 2) {

    // already sorted

  } else if (N < insertion_sort_cutoff) {

    // use insertion sort for small inputs
    insertion_sort(begin, end, comp);

  } else if (depth == 0) {

    // use heap sort if recurse too deep
    heap_sort(begin, end, comp);

  } else {

    // use quick sort
    // choose pivot with median of 3
    Iter mid = begin + N/2;
    Iter last = end-1;
    Iter pivot = comp(*begin, *mid)
                    ? ( comp(*mid, *last)
                           ? mid
                           : ( comp(*begin, *last)
                                  ? last
                                  : begin ) )
                    : ( comp(*mid, *last)
                           ? ( comp(*begin, *last)
                                  ? begin
                                  : last )
                           : mid );

    IterVal<Iter> pivot_val = *pivot;
    // partition
    pivot = partition(begin, end, [&](Iter it){ return comp(*it, pivot_val); });

    // recurse to sort first and second parts, ignoring already sorted pivot
    // by construction pivot is always in the range [begin, end)
    intro_sort(begin, pivot, comp, depth-1);
    intro_sort(RAJA::next(pivot), end, comp, depth-1);
  }

}

/*!
    \brief stable merge sort given range inplace using comparison function
    and using O(N*lg(N)) comparisons and O(N) memory
*/
template <typename Iter, typename Compare>
RAJA_HOST_DEVICE RAJA_INLINE
void
merge_sort(Iter begin,
           Iter end,
           Compare comp)
{
  static_assert(!type_traits::is_iterator<Iter>::value, "unimplemented");
}

}  // namespace detail


/*!
    \brief unstable partition given range inplace using predicate function
    and using O(N) predicate evaluations and O(1) memory
*/
template <typename Iter,
          typename Predicate>
RAJA_HOST_DEVICE RAJA_INLINE
Iter
partition(Iter begin,
          Iter end,
          Predicate pred)
{
  return detail::partition(begin, end, pred);
}

/*!
    \brief unstable insertion sort given range inplace using comparison function
    and using O(N^2) comparisons and O(1) memory
*/
template <typename Iter,
          typename Compare = operators::less<detail::IterVal<Iter>>>
RAJA_HOST_DEVICE RAJA_INLINE
concepts::enable_if<type_traits::is_iterator<Iter>>
insertion_sort(Iter begin,
               Iter end,
               Compare comp = Compare{})
{
  auto N = end - begin;

  if (N > 1) {

    detail::insertion_sort(begin, end, comp);
  }
}

/*!
    \brief unstable heap sort given range inplace using comparison function
    and using O(N*lg(N)) comparisons and O(1) memory
*/
template <typename Iter,
          typename Compare = operators::less<detail::IterVal<Iter>>>
RAJA_HOST_DEVICE RAJA_INLINE
concepts::enable_if<type_traits::is_iterator<Iter>>
heap_sort(Iter begin,
          Iter end,
          Compare comp = Compare{})
{
  auto N = end - begin;

  if (N > 1) {

    detail::heap_sort(begin, end, comp);
  }
}

/*!
    \brief unstable intro sort given range inplace using comparison function
    and using O(N*lg(N)) comparisons and O(lg(N)) memory
*/
template <typename Iter,
          typename Compare = operators::less<detail::IterVal<Iter>>>
RAJA_HOST_DEVICE RAJA_INLINE
concepts::enable_if<type_traits::is_iterator<Iter>>
intro_sort(Iter begin,
           Iter end,
           Compare comp = Compare{})
{
  auto N = end - begin;

  if (N > 1) {

    // sset max depth to 2*lg(N)
    unsigned max_depth = 2*detail::ulog2(N);

    detail::intro_sort(begin, end, comp, max_depth);
  }
}

/*!
    \brief stable merge sort given range inplace using comparison function
    and using O(N*lg(N)) comparisons and O(N) memory
*/
template <typename Iter,
          typename Compare = operators::less<detail::IterVal<Iter>>>
RAJA_HOST_DEVICE RAJA_INLINE
concepts::enable_if<type_traits::is_iterator<Iter>>
merge_sort(Iter begin,
           Iter end,
           Compare comp = Compare{})
{
  auto N = end - begin;

  if (N > 1) {

    detail::merge_sort(begin, end, comp);
  }
}

}  // namespace RAJA

#endif
