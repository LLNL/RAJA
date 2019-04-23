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

using RAJA::statement::ParamList;
using RAJA::statement::Param;

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

//Converts SegList<1,2,3> -> list<Seg<0>, Seg<1>, Seg<2>>
template<camp::idx_t head, camp::idx_t... tail>
struct listMaker<SegList<head,tail...>>
{
  static auto genList()
    -> decltype(catList<camp::list<Seg<head>>,
            decltype(listMaker<SegList<tail...>>::genList())>::makeList(
            camp::list<Seg<head>>{},
            listMaker<SegList<tail...>>::genList()))
  {
    std::cout<<"SegList -> Seg "<<head<<std::endl;

    return catList<camp::list<Seg<head>>,
            decltype(listMaker<SegList<tail...>>::genList())>::makeList(
            camp::list<Seg<head>>{},
            listMaker<SegList<tail...>>::genList());
  }
};

//Converts Seg<id> -> list<Seg<id>>
template<camp::idx_t id>
struct listMaker<Seg<id>>
{
  static auto genList()
    -> camp::list<Seg<id>>
  {
    std::cout<<"Seg<id> -> Seg<id>"<<std::endl;
    return camp::list<Seg<id>>{};
  }
};

//Converts ParamList<1,2,3> -> list<Param<0>, Param<1>, Param<2>>
template<camp::idx_t head, camp::idx_t... tail>
struct listMaker<ParamList<head,tail...>>
{
  static auto genList()
    -> decltype(catList<camp::list<Param<head>>,
            decltype(listMaker<ParamList<tail...>>::genList())>::makeList(
            camp::list<Param<head>>{},
            listMaker<ParamList<tail...>>::genList()))
  {
    std::cout<<"ParamList -> Param "<<head<<std::endl;

    return catList<camp::list<Param<head>>,
            decltype(listMaker<ParamList<tail...>>::genList())>::makeList(
            camp::list<Param<head>>{},
            listMaker<ParamList<tail...>>::genList());
  }
};


//Converts Param<id> -> list<Param<id>>
template<camp::idx_t id>
struct listMaker<Param<id>>
{
  static auto genList()
    -> camp::list<Param<id>>
  {
    std::cout<<"Param<id> -> Param<id>"<<std::endl;
    return camp::list<Param<id>>{};
  }
};





//Helper to unroll list
template <typename List>
struct printList
{};

template<>
struct printList<camp::list<>>
{
  static void display() {};
};

template<typename Head, typename...Tail>
struct printList<camp::list<Head, Tail...>>
{
  static void display()
  {
    std::cout<<"Seg Id "<<Head::seg_idx<<std::endl;
    printList<camp::list<Tail...>>::display();
  }
};

//Helper structs to parse through the arguments
template<typename List>
struct parser{};

template<>
struct parser<camp::list<>>
{
  static auto checkArgs()
    -> camp::list<>
  {
    printf("last parser \n");
    return camp::list<> {};
  }
};

template <typename Head, typename... Tail>
struct parser<camp::list<Head, Tail...>>
{

  static auto checkArgs()
    -> decltype(catList<decltype(listMaker<Head>::genList()),
                decltype(parser<camp::list<Tail...> >::checkArgs())>
                ::makeList(listMaker<Head>::genList(), parser<camp::list<Tail...> >::checkArgs()))
  {

    printf("Parsing argument ... \n");


    return catList<decltype(listMaker<Head>::genList()),
                   decltype(parser<camp::list<Tail...> >::checkArgs())>
      ::makeList(listMaker<Head>::genList(), parser<camp::list<Tail...> >::checkArgs());
  }
};

template<typename List>
struct call_extractor
{};

template<typename ...Args>
struct call_extractor<camp::list<Args...>>
{
  template<typename Data>
  static auto make_tuple(Data &&data)
    -> decltype(camp::make_tuple(extractor<Args>::extract_arg(data) ...))
  {
    return camp::make_tuple(extractor<Args>::extract_arg(data) ...);
  }
};


//Lambda with custom args
template <camp::idx_t LoopIndex,typename... Args>
struct StatementExecutor<statement::Lambda<LoopIndex, Args...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    printf("Entered executor \n");

    parser<camp::list<Args...>>::checkArgs();

    const int size = sizeof...(Args); //Total number of arguments
    auto myTuple =
      call_extractor<decltype(parser<camp::list<Args...>>::checkArgs())>::make_tuple(data);

    qinvoke_lambda<LoopIndex>(std::forward<Data>(data), myTuple,camp::make_idx_seq_t<size>{});
    //call_extractor<camp::list<Seg<0>>>::make_tuple(data);

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
