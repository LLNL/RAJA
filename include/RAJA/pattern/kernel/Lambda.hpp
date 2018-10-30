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
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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
#include "RAJA/util/ShmemTile.hpp"

#include "RAJA/pattern/kernel/internal.hpp"
#include <typeinfo>

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
template <camp::idx_t BodyIdx>
struct Lambda : internal::Statement<camp::nil> {
  const static camp::idx_t loop_body_index = BodyIdx;
};


//Create Shared Memory Statement V2.0
template<typename Indices,typename... EnclosedStmts>
struct CreateShmem : public internal::Statement<camp::nil> {
};

template<camp::idx_t... Indices,typename... EnclosedStmts>
struct CreateShmem<camp::idx_seq<Indices...>, EnclosedStmts...> : public internal::Statement<camp::nil> {
};


}// end namespace statement

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

//Shared memory creator version 2.0
template<camp::idx_t... Indices, typename... EnclosedStmts>
struct StatementExecutor<statement::CreateShmem<camp::idx_seq<Indices...>, EnclosedStmts...> >{

  //
  //Here we are out of objects that need to be intialized
  //
  template<class Data>
  static void RAJA_INLINE createShared(Data && data)
  {
    //Execute Statement List
    execute_statement_list<camp::list<EnclosedStmts...>>(data);
  }

  template<camp::idx_t Pos, camp::idx_t... others, class Data>
  static void RAJA_INLINE createShared(Data && data)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::type;
    varType SharedM;
    camp::get<Pos>(data.param_tuple).m_MemObj = &SharedM;
    createShared<others...>(data);
  }

  //Set pointer to null
  template<class Data>
  static void RAJA_INLINE setPtrToNull(Data &&) {} 

  template<camp::idx_t Pos, camp::idx_t... others, class Data>
  static void RAJA_INLINE setPtrToNull(Data && data)
  {
    camp::get<Pos>(data.param_tuple).m_MemObj = nullptr;
    setPtrToNull<others...>(data);
  }

  template<typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    //Initalize shared memory + launch loops
    createShared<Indices...>(data);

    //Set wrapper pointers to null
    setPtrToNull<Indices...>(data);
  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
