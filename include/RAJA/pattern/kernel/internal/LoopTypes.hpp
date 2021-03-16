/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for loop kernel internals: LoopData structure and
 *          related helper functions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_internal_LoopTypes_HPP
#define RAJA_pattern_kernel_internal_LoopTypes_HPP

#include "RAJA/config.hpp"

#include "camp/camp.hpp"


namespace RAJA
{
namespace internal
{

namespace detail
{
// Helper class to convert a camp::idx_t into some type T
// used in template expansion in ListOfNHelper
template <typename T, camp::idx_t>
struct SeqToType
{
  using type = T;
};

template <typename T, typename SEQ>
struct ListOfNHelper;

template <typename T, camp::idx_t ... SEQ>
struct ListOfNHelper<T, camp::idx_seq<SEQ...> >
{
  using type = camp::list<typename SeqToType<T, SEQ>::type...>;
};
} // namespace detail

/*
 *  This creates a camp::list with N types, each one being T.
 *
 *  That is, list_of_n<T, 4>  ==  camp::list<T, T, T, T>
 *
 */
template <typename T, camp::idx_t N>
using list_of_n = typename detail::ListOfNHelper<T, camp::make_idx_seq_t<N>>::type;


template <typename SegmentTypes,
          typename OffsetTypes>
struct LoopTypes;

template <typename ... SegmentTypes,
          typename ... OffsetTypes>
struct LoopTypes<camp::list<SegmentTypes...>, camp::list<OffsetTypes...>> {

  using Self = LoopTypes<camp::list<SegmentTypes...>, camp::list<OffsetTypes...>>;

  static constexpr size_t s_num_segments = sizeof...(SegmentTypes);

  // This ensures that you don't double-loop over a segment within the same
  // loop nesting
  static_assert(s_num_segments == sizeof...(OffsetTypes),
      "Number of segments and offsets must match");

  using segment_types_t = camp::list<SegmentTypes...>;
  using offset_types_t = camp::list<OffsetTypes...>;
};


template<typename Data>
using makeInitialLoopTypes =
    LoopTypes<list_of_n<void, camp::tuple_size<typename Data::segment_tuple_t>::value>,
              list_of_n<void, camp::tuple_size<typename Data::segment_tuple_t>::value>>;


template<typename Types, camp::idx_t Segment, typename T, typename Seq>
struct SetSegmentTypeHelper;

template<typename Types,
         camp::idx_t Segment,
         typename T,
         camp::idx_t ... SEQ>
struct SetSegmentTypeHelper<Types, Segment, T, camp::idx_seq<SEQ...>>
{
    using segment_list = typename Types::segment_types_t;
    using offset_list = typename Types::offset_types_t;

    static_assert(std::is_same<camp::at_v<segment_list, Segment>, void>::value,
        "Segment was already assigned: Probably looping over same segment in loop nest");

    using type = LoopTypes<
        camp::list<typename std::conditional<SEQ == Segment, T, camp::at_v<segment_list, SEQ>>::type...>,
        camp::list<typename std::conditional<SEQ == Segment, T, camp::at_v<segment_list, SEQ>>::type...>>;

};


template<typename Types, camp::idx_t Segment, typename T>
using setSegmentType =
    typename SetSegmentTypeHelper<Types, Segment, T, camp::make_idx_seq_t<Types::s_num_segments>>::type;

template<typename Types, camp::idx_t Segment, typename Data>
using setSegmentTypeFromData =
    setSegmentType<Types, Segment, camp::at_v<typename camp::decay<Data>::index_tuple_t::TList, Segment>>;


}  // end namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_internal_LoopData_HPP */
