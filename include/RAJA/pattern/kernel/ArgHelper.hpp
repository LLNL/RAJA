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

#include "RAJA/pattern/kernel/OffSet.hpp"
#include "RAJA/pattern/kernel/Param.hpp"
#include "RAJA/pattern/kernel/Seg.hpp"

namespace RAJA
{

namespace internal
{

using RAJA::statement::SegList;
using RAJA::statement::Seg;

using RAJA::statement::OffSet;
using RAJA::statement::OffSetList;

using RAJA::statement::ParamList;
using RAJA::statement::Param;

//Concatanate lists
template<typename ListA, typename ListB>
struct catList
{};

template<typename...itemsA, typename...itemsB>
struct catList<camp::list<itemsA...>, camp::list<itemsB...>>
{
  using type = typename camp::list<itemsA...,itemsB...>;
};

template<typename...itemsA>
struct catList<camp::list<itemsA...>, camp::list<>>
{
  using type  = typename camp::list<itemsA... >;
};

//List Maker
template<typename Arg>
struct listMaker
{
  using type = typename camp::list<>;
};

  
//Converts SegList<1,2,3> -> list<Seg<0>, Seg<1>, Seg<2>>
template<camp::idx_t head, camp::idx_t... tail>
struct listMaker<SegList<head,tail...>>
{
  using type = typename catList<camp::list<Seg<head>>,
	        typename listMaker<SegList<tail...> >::type>::type;
};

//Converts Seg<id> -> list<Seg<id>>
template<camp::idx_t id>
struct listMaker<Seg<id>>
{
  using type = typename camp::list<Seg<id>>::type;
};


//Converts OffSetList<1,2,3> -> list<OffSet<0>, OffSet<1>, OffSet<2>>
template<camp::idx_t head, camp::idx_t... tail>
struct listMaker<OffSetList<head,tail...>>
{
  using type = typename catList<camp::list<OffSet<head>>,
	        typename listMaker<OffSetList<tail...> >::type>::type;
};

//Converts OffSet<id> -> list<OffSet<id>>
template<camp::idx_t id>
struct listMaker<OffSet<id>>
{
  using type = typename camp::list<OffSet<id>>::type;
};

//Converts ParamList<1,2,3> -> list<Param<0>, Param<1>, Param<2>>
template<camp::idx_t head, camp::idx_t... tail>
struct listMaker<ParamList<head,tail...>>
{
  using type = typename catList<camp::list<Param<head>>,
	        typename listMaker<ParamList<tail...> >::type>::type;
};

//Converts Param<id> -> list<Param<id>>
template<camp::idx_t id>
struct listMaker<Param<id>>
{
  using type = typename camp::list<Param<id>>::type;
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
  using type = typename catList<typename listMaker<Head>::type,
				typename parser<camp::list<Tail...>>::type
				>::type;
};


//Extracts arguments from segments, and parameters
template<typename Head, typename...Tail>
struct extractor
{};

template<camp::idx_t id, typename...Tail>
struct extractor<RAJA::statement::OffSet<id>, Tail...>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  static auto extract_arg(Data &&data) ->
    decltype(camp::get<id>(data.offset_tuple))
  {
    return camp::get<id>(data.offset_tuple);
  }
};

template<camp::idx_t id, typename...Tail>
struct extractor<RAJA::statement::Seg<id>, Tail...>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  static auto extract_arg(Data &&data) ->
    decltype(camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)])
  {
    return camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)];
  }
};

template<camp::idx_t id, typename...Tail>
struct extractor<RAJA::statement::Param<id>, Tail...>
{
  template<typename Data>
  RAJA_HOST_DEVICE
  static auto extract_arg(Data &&data)->
    decltype(camp::get<id>(data.param_tuple))
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
    -> decltype(camp::make_tuple(extractor<Args>::extract_arg(data) ...))
  {
    return camp::make_tuple(extractor<Args>::extract_arg(data) ...);
  }
};
  
}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
