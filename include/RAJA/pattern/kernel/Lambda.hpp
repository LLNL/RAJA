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

//This statement will create shared memory
template<typename... EnclosedStmts>
struct CreateShmem : public internal::Statement<camp::nil> {
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

//Helper struct to help us count.
template<camp::idx_t> struct int_{};
 
template<typename... EnclosedStmts>
struct StatementExecutor<statement::CreateShmem<EnclosedStmts...>>{

  template<class Data, camp::idx_t Pos>
  static RAJA_INLINE void createShared(Data &&data, int_<Pos>){

    using varType = typename camp::tuple_element_t<Pos-1, typename camp::decay<Data>::param_tuple_t>::type;
    varType SharedM;
    camp::get<Pos-1>(data.param_tuple).SharedMem = &SharedM;

    createShared( data, int_<Pos-1>());
  }
  
  template<class Data>
  static void RAJA_INLINE createShared(Data &&data, int_<static_cast<camp::idx_t>(1)>){
    
    using varType = typename camp::tuple_element_t<0, typename camp::decay<Data>::param_tuple_t>::type;
    varType SharedM;
    
    camp::get<0>(data.param_tuple).SharedMem = &SharedM;

    //Execute Statement List
    execute_statement_list<camp::list<EnclosedStmts...>>(data);
  }  
  
  template<typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {    
    const camp::idx_t N = camp::tuple_size<typename camp::decay<Data>::param_tuple_t>::value;
    createShared(data,int_<N>());
  }
};

}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
