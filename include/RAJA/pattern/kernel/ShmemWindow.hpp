/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for shared memory window.
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


#ifndef RAJA_pattern_kernel_ShmemWindow_HPP
#define RAJA_pattern_kernel_ShmemWindow_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/util/ShmemTile.hpp"
#include "RAJA/util/StaticLayout.hpp"

namespace RAJA
{

namespace statement
{


/*!
 * A RAJA::kernel statement that sets the shared memory window.
 *
 *
 */

template <typename... EnclosedStmts>
struct SetShmemWindow
    : public internal::Statement<camp::nil, EnclosedStmts...> {
};

/*!
 * Kernel statements intialize local arrays
 * The statement takes parameter sequence to intialize
 * memory from the parameter tuple and additional statements.
 *
 * For example:
 * IntiLocalMem<Pol, RAJA::param_idx<0>, statements...>
 * Will intialize the 0th array in the param tuple
 */
template<typename Pol, typename Indices, typename... EnclosedStmts>
struct InitLocalMem : public internal::Statement<camp::nil> {
};

//Policy Specialization
template<camp::idx_t... Indices, typename... EnclosedStmts>
struct InitLocalMem<RAJA::cpu_tile_mem, camp::idx_seq<Indices...>, EnclosedStmts...> : public internal::Statement<camp::nil> {
};


}  // end namespace statement

namespace internal
{

//Statement executor to initalize RAJA local array
template<camp::idx_t... Indices, typename... EnclosedStmts>
struct StatementExecutor<statement::InitLocalMem<RAJA::cpu_tile_mem,camp::idx_seq<Indices...>, EnclosedStmts...> >{
  
  //Execute statement list
  template<class Data>
  static void RAJA_INLINE initMem(Data && data)
  {
    execute_statement_list<camp::list<EnclosedStmts...>>(data);
  }
  
  //Intialize local array
  //Identifies type + number of elements needed
  template<camp::idx_t Pos, camp::idx_t... others, class Data>
  static void RAJA_INLINE initMem(Data && data)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::element_t;
    const camp::idx_t NumElem = camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::NumElem;
    
    varType Array[NumElem];
    camp::get<Pos>(data.param_tuple).m_arrayPtr = Array;
    initMem<others...>(data);
  }
  
  //Set pointer to null
  template<class Data>
  static void RAJA_INLINE setPtrToNull(Data &&) {}
  
  template<camp::idx_t Pos, camp::idx_t... others, class Data>
  static void RAJA_INLINE setPtrToNull(Data && data)
  {
    camp::get<Pos>(data.param_tuple).m_arrayPtr = nullptr;
    setPtrToNull<others...>(data);
  }
  
  template<typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    //Initalize local arrays + execute statements
    initMem<Indices...>(data);
    
    //set array pointers to null
    setPtrToNull<Indices...>(data);
  }
  
};

template <typename... EnclosedStmts>
struct StatementExecutor<statement::SetShmemWindow<EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    // Call setWindow on all of our shmem objects
    shmem_set_windows(data.param_tuple, data.get_minimum_index_tuple());

    // Invoke the enclosed statements
    execute_statement_list<camp::list<EnclosedStmts...>>(
        std::forward<Data>(data));
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
