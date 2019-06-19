/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel lambda executor.
 *
 ******************************************************************************
 */


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_pattern_kernel_arghelper_HPP
#define RAJA_pattern_kernel_arghelper_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/Param.hpp"
#include "RAJA/pattern/kernel/LambdaArgs.hpp"

namespace RAJA
{

namespace internal
{

using RAJA::statement::seg_t;
using RAJA::statement::param_t;
using RAJA::statement::offset_t;
using RAJA::statement::LambdaArgs;

//Concatanate lists
template<typename ListA, typename ListB>
struct merge_list
{};

template<typename...itemsA, typename...itemsB>
struct merge_list<camp::list<itemsA...>, camp::list<itemsB...>>
{
  using type = typename camp::list<itemsA...,itemsB...>;
};

//List Maker
template<typename Arg>
struct listMaker
{
  using type = typename camp::list<>;
};

//Convert LambdaArgs<T, 1, 2, 3> - > camp::list<LambdaArgs<T, 1>, LambdaArgs<T, 2>, LambdaArgs<T, 3> >
template<typename T, camp::idx_t... elems>
struct listMaker<LambdaArgs<T, elems...>>
{
  using type = camp::list<LambdaArgs<T, elems>...>;
};

template<typename List>
struct parser{};

template<>
struct parser<camp::list<>>
{
  using type = camp::list<>;
};

template <typename Head, typename... Tail>
struct parser<camp::list<Head, Tail...>>
{
  using type = typename merge_list<typename listMaker<Head>::type,
				typename parser<camp::list<Tail...>>::type
				>::type;
};

//Extracts arguments from segments, and parameters
template<typename T>
struct extractor;

template<camp::idx_t id>
struct extractor<LambdaArgs<offset_t, id>>
{

  template<typename Data>
  RAJA_HOST_DEVICE
  static auto extract_arg(Data &&data) ->
    decltype(camp::get<id>(data.offset_tuple))
  {
    return camp::get<id>(data.offset_tuple);
  }

};

template<camp::idx_t id>
struct extractor<LambdaArgs<seg_t, id>>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  static auto extract_arg(Data &&data) ->
    decltype(camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)])
  {
    return camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)];
  }
};

template<camp::idx_t id>
struct extractor<LambdaArgs<param_t, id>>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  static auto extract_arg(Data &&data)->
    typename std::add_lvalue_reference<camp::tuple_element_t<id,typename camp::decay<Data>::param_tuple_t>>::type
  {
    return camp::get<id>(data.param_tuple);
  }
};

template<typename List>
struct call_extractor
{};

template<typename ...Args>
struct call_extractor<camp::list<Args...>>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  static auto make_tuple(Data &&data)
    -> camp::tuple<decltype(extractor<Args>::extract_arg(data))...>
  {
    return camp::tuple<decltype(extractor<Args>::extract_arg(data))...>{extractor<Args>::extract_arg(data)...};
  }
};

}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
