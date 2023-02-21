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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_kernel_internal_Template_HPP
#define RAJA_pattern_kernel_internal_Template_HPP

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


template <typename T, typename SEQ>
struct TupleOfNHelper;

template <typename T, camp::idx_t ... SEQ>
struct TupleOfNHelper<T, camp::idx_seq<SEQ...> >
{
  using type = camp::tuple<typename SeqToType<T, SEQ>::type...>;
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


/*
 *  This creates a camp::tuple with N types, each one being T.
 *
 *  That is, tuple_of_n<T, 4>  ==  camp::tuple<T, T, T, T>
 *
 */
template <typename T, camp::idx_t N>
using tuple_of_n = typename detail::TupleOfNHelper<T, camp::make_idx_seq_t<N>>::type;



}  // end namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_internal_LoopData_HPP */
