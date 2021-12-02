/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA NDto1DHolder.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_NDto1DHolder_HPP
#define RAJA_NDto1DHolder_HPP

#include <type_traits>

#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/Span.hpp"

namespace RAJA
{

/*!
 * @brief A holder for adapting lambdas meant for multidimensional index spaces
 *        to allow their use in 1-dimensional index spaces.
 *
 * Creates callable object that when called with a 1-dimensional index, converts
 * that index to a multi-dimensional index and calls the given lambda.
 * This allows lambdas meant for use in multi-dimensional loop abstractions to
 * be used in 1-dimensional loop abstractions.
 * The 1-dimensional index is of the type of the IndexType template parameter.
 *
 * For example:
 *
 *     // lambda for use in 3-d loop
 *     auto lambda = [=](int i, int j, int k) {...};
 *
 *     // example lambda usage
 *     for (int i = ibegin; i < iend; ++i) {
 *       for (int j = jbegin; j < jend; ++j) {
 *         for (int k = kbegin; k < kend; ++k) {
 *           lambda(i, j, k);
 *         }
 *       }
 *     }
 *
 *     // ranges for 3D loop bounds
 *     RAJA::TypedRangeSegment<int> irange(ibegin, iend);
 *     RAJA::TypedRangeSegment<int> jrange(jbegin, jend);
 *     RAJA::TypedRangeSegment<int> krange(kbegin, kend);
 *
 *     // Create a NDto1DHolder object for lambda and the ranges
 *     // NOTE Use make_NDto1DHolder
 *     RAJA::NDto1DHolder<decltype(lambda), int, RAJA::TypedRangeSegment<int>, RAJA::TypedRangeSegment<int>, RAJA::TypedRangeSegment<int>>
 *         holder(lambda, irange, jrange, krange);
 *
 *     // Use with RAJA forall
 *     RAJA::forall<policy>(holder.getRange(), holder);
 *
 *     // Use with c++-style loop
 *     auto range = holder.getRange();
 *     for (auto it = begin(range); it < end(range); ++it) {
 *       holder(*it)
 *     }
 *
 */
template <typename Lambda, typename IndexType, typename ... Spans>
struct NDto1DHolder
{
  static_assert(sizeof...(Spans) > 0, "Must use one or more spans");

  using range_type = RAJA::TypedRangeSegment<IndexType>;
  using tuple_type = camp::tuple<Spans...>;

  template < camp::idx_t I >
  using span_type = typename camp::tuple_element<I, tuple_type>::type;
  template < camp::idx_t I >
  using iterator = typename span_type<I>::iterator;
  template < camp::idx_t I >
  using element_type = typename std::iterator_traits<iterator<I>>::difference_type;
  template < camp::idx_t I >
  using difference_type = camp::decay<element_type<I>>;

  static_assert(type_traits::is_integral<IndexType>::value,
                "IndexType must model Integral");

  // constructor from lambda and spans
  // NOTE this uses C_Span0, C_Spans... to avoid conflicting with the
  // copy constructor
  template < typename C_LAMBDA, typename C_Span0, typename ... C_Spans >
  RAJA_HOST_DEVICE NDto1DHolder(C_LAMBDA&& lambda, C_Span0&& span0, C_Spans&&... spans)
      : m_lambda(std::forward<C_LAMBDA>(lambda))
      , m_spans(std::forward<C_Span0>(span0), std::forward<C_Spans>(spans)...)
  {
  }

  // Call operator taking 1-dimensional index argument
  // Note: this may not be called with an out of bounds index
  RAJA_HOST_DEVICE RAJA_INLINE auto operator()(IndexType idx) const
  {
    return call_helper(idx, camp::make_idx_seq_t<sizeof...(Spans)-1>{});
  }

  // Get the total size of the index space
  RAJA_HOST_DEVICE RAJA_INLINE IndexType size() const
  {
    return size_helper(camp::make_idx_seq_t<sizeof...(Spans)>{});
  }

  // Check if the index space is empty
  RAJA_HOST_DEVICE RAJA_INLINE bool empty() const
  {
    return size() == static_cast<IndexType>(0);
  }

  // Get the 1-dimensional range representing the index space.
  RAJA_HOST_DEVICE RAJA_INLINE range_type getRange() const
  {
    return range_type(static_cast<IndexType>(0), size());
  }

private:
  Lambda m_lambda;
  tuple_type m_spans;

  template < camp::idx_t ... Is >
  RAJA_HOST_DEVICE inline IndexType size_helper(camp::idx_seq<Is...>) const
  {
    IndexType prod = 1;
    // use fold-like expression guaranteed to evaluate in order
    using fold_helper = int[];
    (void)fold_helper{((void)(prod *= static_cast<IndexType>(RAJA::get<Is>(m_spans).size())), 0)...};
    return prod;
  }

  // Note that here sizeof...(Is) == sizeof...(Spans)-1.
  // This is done to avoid extra calculations when getting the left most index.
  template < camp::idx_t ... Is >
  RAJA_HOST_DEVICE inline auto call_helper(IndexType idx, camp::idx_seq<Is...>) const
  {
    // Calculate the indices starting with the rightmost (stride 1) index.
    // Do this iteratively, slicing off one dimension at a time until all
    // indices but one are generated. What remains is the leftmost index.

    // get index by moding by size, remove this dimension by dividing by idx
    auto expr = [&](IndexType size) {
      IndexType mod = idx % size;
      IndexType div = idx / size;
      idx = div;
      return mod;
    };

    // use fold-like expression guaranteed to evaluate in order
    IndexType indices_backwards[] = {
      static_cast<IndexType>(0), // ensure indices_backwards array length > 0
      expr(static_cast<IndexType>(RAJA::get<sizeof...(Spans)-1-Is>(m_spans).size()))...};
    return m_lambda(
        RAJA::get<0>(m_spans).begin()[static_cast<difference_type<0>>(idx)],
        RAJA::get<Is+1>(m_spans).begin()[static_cast<difference_type<Is+1>>(indices_backwards[sizeof...(Spans)-1-Is])]...);
  }
};

/*!
 * @brief Creates a NDto1DHolder class from a lambda and segments.
 * @param lambda functional object
 * @param segs iterable objects defining the multi-dimensional index space
 * @return Returns a NDto1DHolder for the given lambda and segments
 *
 * Creates a NDto1DHolder object given a lambda and iterables.
 *
 * NOTE: the stride 1 index is the right-most index
 *
 * For example:
 *
 *     // lambda to be used in 2D loop
 *     auto lambda = [](int i, int j) {...};
 *
 *     // example lambda usage
 *     for (int i = ibegin; i < iend; ++i) {
 *       for (int j = jbegin; j < jend; ++j) {
 *         lambda(i, j);
 *       }
 *     }
 *
 *     // ranges for 2D loop bounds
 *     RAJA::TypedRangeSegment<int> irange(ibegin, iend);
 *     RAJA::TypedRangeSegment<int> jrange(jbegin, jend);
 *
 *     // holder class
 *     auto holder = RAJA::make_NDto1DHolder(lambda, irange, jrange);
 *
 *     // Use with RAJA forall
 *     RAJA::forall<policy>(holder.getRange(), holder);
 *
 *     // Use with c++-style loop
 *     auto range = holder.getRange();
 *     for (auto it = begin(range); it < end(range); ++it) {
 *       holder(*it)
 *     }
 *
 */
template <typename Lambda, typename ... Segments>
RAJA_HOST_DEVICE RAJA_INLINE
auto make_NDto1DHolder(Lambda&& lambda, Segments&&... segs)
{
  using std::begin; using std::end; using std::distance;
  using IndexType = typename std::common_type<
      typename std::iterator_traits<decltype(begin(segs))>::difference_type... >::type;
  return NDto1DHolder<typename std::decay<Lambda>::type,
                      IndexType,
                      decltype(RAJA::make_span(begin(segs), distance(begin(segs), end(segs))))...>(
      std::forward<Lambda>(lambda),
      RAJA::make_span(begin(segs), distance(begin(segs), end(segs)))...);
}

}  // end namespace RAJA

#endif /* RAJA_NDto1DHolder_HPP */
