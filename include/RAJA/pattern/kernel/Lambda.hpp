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

//Base case
template <camp::idx_t BodyIdx, typename SegIdx, typename ParamIdx>
struct tLambda : internal::Statement<camp::nil> {
  const static camp::idx_t loop_body_index = BodyIdx;
};

//Variadic
template <camp::idx_t BodyIdx, camp::idx_t... SegIdx, camp::idx_t ... ParamIdx>
struct tLambda<BodyIdx, camp::idx_seq<SegIdx...>, camp::idx_seq<ParamIdx...> > : internal::Statement<camp::nil> {
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

template<camp::idx_t Pos, typename... Args>
struct inspector
{
  template<typename T>
  static void make_args(T tuple){
    std::cout<<"Kicking off with "<<Pos<<std::endl;
    inspector<Pos, Args...>::make_args(tuple);
  }
};

template<camp::idx_t Pos, camp::idx_t id, typename... Tail>
struct inspector<Pos, RAJA::statement::Param<id>, Tail...>
{
  template<typename T>
  static void make_args(T tuple){
    std::cout<<"Param with "<<Pos<<std::endl;
    inspector<Pos-1, Tail...>::make_args(tuple);
  }
};

template<camp::idx_t Pos, camp::idx_t id, typename...Tail>
struct inspector<Pos, RAJA::statement::Seg<id>, Tail...>
{
  template<typename T>
  static void make_args(T tuple){
    std::cout<<"Seg with "<<Pos<<std::endl;
    inspector<Pos-1, Tail...>::make_args(tuple);
  }
};

template<camp::idx_t id>
struct inspector<1, RAJA::statement::Param<id>>
{
  template<typename T>
  static void make_args(T tuple){
    std::cout<<"Last param call"<<std::endl;
  }
};

template<camp::idx_t id>
struct inspector<1, RAJA::statement::Seg<id>>
{
  template<typename T>
  static void make_args(T tuple){
    std::cout<<"Last seg call "<<std::endl;
  }
};

template <camp::idx_t LoopIndex,typename... Args>
struct StatementExecutor<statement::Lambda<LoopIndex, Args...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    printf("\n Building up lambda args \n");

    const int size = sizeof...(Args);
    auto myTuple = camp::make_tuple(); //empty tuple
    inspector<size, Args...>::make_args(myTuple);
    std::cout<<"\n \n"<<std::endl;

    //RAJA::statement::Param<0> a;
    //inspector<size, Args...>::make_arg(myTuple);


    auto seg_tuple = data.segment_tuple;
    auto off_tuple = data.offset_tuple;
    auto par_tuple = data.param_tuple;

    auto argTuple = camp::make_tuple(camp::get<0>(seg_tuple).begin()[camp::get<0>(off_tuple)]);

    qinvoke_lambda<LoopIndex>(std::forward<Data>(data));
  }
};



template <camp::idx_t LoopIndex, camp::idx_t... SegIdx, camp::idx_t... ParamIdx>
struct StatementExecutor<statement::tLambda<LoopIndex, camp::idx_seq<SegIdx...>, camp::idx_seq<ParamIdx...> > >
 {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    tinvoke_lambda<LoopIndex>
      (std::forward<Data>(data),camp::idx_seq<SegIdx...>{}, camp::idx_seq<ParamIdx...>{});

  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
