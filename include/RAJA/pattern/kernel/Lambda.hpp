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


#ifndef RAJA_pattern_kernel_Lambda_HPP
#define RAJA_pattern_kernel_Lambda_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/internal.hpp"
#include "RAJA/pattern/kernel/Param.hpp"
#include "RAJA/pattern/kernel/Seg.hpp"

namespace RAJA
{

namespace statement
{


/*!
 * A RAJA::kernel statement that invokes a lambda function.
 *
 * The lambda is specified by its index in the sequence of lambda arguments
 * to a RAJA::kernel method.
 *
 * for example:
 * RAJA::kernel<exec_pol>(make_tuple{s0, s1, s2}, lambda0, lambda1);
 *
 */
template <camp::idx_t BodyIdx, typename... Args >
struct Lambda : internal::Statement<camp::nil> {
  const static camp::idx_t loop_body_index = BodyIdx;
};

}  // end namespace statement

namespace internal
{

template <camp::idx_t LoopIndex>
struct StatementExecutor<statement::Lambda<LoopIndex>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    invoke_lambda<LoopIndex>(std::forward<Data>(data));
  }
};

//...extractor of arguments
template<typename Head, typename...Tail>
struct extractor
{};

template<camp::idx_t id, typename...Tail>
struct extractor<RAJA::statement::Seg<id>, Tail...>
{
  template<typename Data>
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
  static auto extract_arg(Data &&data)->
    decltype(camp::get<id>(data.param_tuple))
  {
    return camp::get<id>(data.param_tuple);
  }
};

//Tools to concatenate lists
template<typename ListA, typename ListB>
struct catList
{};

template<typename...itemsA, typename...itemsB>
struct catList<camp::list<itemsA...>, camp::list<itemsB...>> {

  static auto makeList(camp::list<itemsA...> const &, camp::list<itemsB...> const &) ->
    camp::list<itemsA...,itemsB...>
  {    
    return camp::list<itemsA...,itemsB...> {};
  }

};

template<typename...itemsA>
struct catList<camp::list<itemsA...>, camp::list<>> {

  static auto makeList(camp::list<itemsA...> const &, camp::list<> const &) ->
    camp::list<itemsA...>
  {    
    return camp::list<itemsA...> {};
  }

};

using RAJA::statement::SegList;
using RAJA::statement::Seg;

//TODO listmaker
template<typename Arg>
struct listMaker
{
  static auto genList()
    -> camp::list<>
  {
    return camp::list<>{};
  }
};
//Need to be able to convert SegList<1,2,3> -> list<Seg<0>, Seg<1>, Seg<2>>

template<camp::idx_t head, camp::idx_t... tail>
struct listMaker<SegList<head,tail...>>
{
  static auto genList() 
    -> decltype(catList<camp::list<Seg<head>>,
            decltype(listMaker<SegList<tail...>>::genList())>::makeList(
            camp::list<Seg<head>>{}, 
            listMaker<SegList<tail...>>::genList()))
  {
    std::cout<<"list maker "<<head<<std::endl;

#if 0
    catList<camp::list<Seg<head>>,
            decltype(listMaker<SegList<tail...>>::genList())>::makeList(
            camp::list<Seg<head>>{}, 
            listMaker<SegList<tail...>>::genList());    
#endif

    return catList<camp::list<Seg<head>>,
            decltype(listMaker<SegList<tail...>>::genList())>::makeList(
            camp::list<Seg<head>>{}, 
            listMaker<SegList<tail...>>::genList());

  }
};

//Lambda with custom args
template <camp::idx_t LoopIndex,typename... Args>
struct StatementExecutor<statement::Lambda<LoopIndex, Args...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    printf("Entered executor \n");
    
    listMaker<SegList<1, 2>>::genList();

#if 0
    camp::list<double> a;    
    camp::list<int> b;
    auto c = catList<decltype(a), decltype(b) >::makeList(a, b);

    catList<camp::list<Seg<1>>, camp::list<Seg<2>> >::makeList(camp::list<Seg<1>>{}, 
                                                               camp::list<Seg<2>>{});
#endif    

#if 0    
    const int size = sizeof...(Args); //Total number of arguments
    auto myTuple = camp::make_tuple(extractor<Args>::extract_arg(data) ...);
    qinvoke_lambda<LoopIndex>(std::forward<Data>(data), myTuple,camp::make_idx_seq_t<size>{});
#endif
  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
